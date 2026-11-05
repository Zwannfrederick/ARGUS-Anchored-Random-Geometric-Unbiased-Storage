// S1, GPU side: decode partials over GPU-resident KV cells, then one merge per row.
// A block is 8 warps over one (KV head, chunk of cells, group of up to 8 query rows), one
// warp per row. Tiles of K, then of V, are staged in shared memory with coalesced loads;
// a lane computes whole dot products of its cell (no per-cell reductions), the softmax is
// updated once per tile, and each lane owns D/32 output elements. A second kernel merges
// the chunks of a row in order. Accumulation is F32 throughout.
#include "ggml_split_attention.h"
#include "ggml_cuda_external.h"
#include "ggml_decode_attention.h"
#include "argus_config.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
constexpr int rows_per_block = 8;  // one warp per query row
constexpr int chunk_cells = 512;

// Cells per tile, and lanes per cell so that a warp covers a tile.
template <int D> constexpr int tile_cells = D >= 512 ? 16 : 32;
// half2 words per shared tile row: odd, so lanes reading different rows at the same column
// fall in different banks.
template <int D> constexpr int row_words = D / 2 + 1;

void check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string("ARGUS split attention: ") + cudaGetErrorString(status)); }
}

struct GpuKeyBoxSheet {
    const char * k_base = nullptr;
    int heads_kv = 0;
    int dk = 0;
    int max_blocks = 0;
    uint64_t * d_tags = nullptr;
    __half2 * d_boxes = nullptr;
    __half2 * d_v_boxes = nullptr;
    float * d_tail_norms = nullptr;

    GpuKeyBoxSheet(const char * base, int h_kv, int d, int blocks)
        : k_base(base), heads_kv(h_kv), dk(d), max_blocks(blocks) {
        const size_t n_slots = static_cast<size_t>(h_kv) * blocks;
        check(cudaMalloc(&d_tags, n_slots * sizeof(uint64_t)));
        check(cudaMemset(d_tags, 0, n_slots * sizeof(uint64_t)));
        check(cudaMalloc(&d_boxes, n_slots * static_cast<size_t>(d) * sizeof(__half2)));
        check(cudaMalloc(&d_v_boxes, n_slots * static_cast<size_t>(d / 2) * sizeof(__half2)));
        check(cudaMalloc(&d_tail_norms, n_slots * sizeof(float)));
    }
    GpuKeyBoxSheet(const GpuKeyBoxSheet &) = delete;
    GpuKeyBoxSheet & operator=(const GpuKeyBoxSheet &) = delete;
    ~GpuKeyBoxSheet() {
        if (d_tags) { cudaFree(d_tags); }
        if (d_boxes) { cudaFree(d_boxes); }
        if (d_v_boxes) { cudaFree(d_v_boxes); }
        if (d_tail_norms) { cudaFree(d_tail_norms); }
    }
};

std::mutex gpu_sheet_lock;
std::deque<GpuKeyBoxSheet> gpu_box_sheets;
unsigned long long * h_mapped_stats = nullptr;
unsigned long long * d_mapped_stats = nullptr;
std::atomic<uint64_t> gpu_kernel_ns[ARGUS_MAX_TRACKED_LAYERS];

ArgusK1Stats get_gpu_layer_stats(int il) {
    if (!h_mapped_stats || il < 0 || il >= ARGUS_MAX_TRACKED_LAYERS) { return {}; }
    const unsigned long long * s = h_mapped_stats + il * 5;
    return {
        static_cast<uint64_t>(s[0]),
        static_cast<uint64_t>(s[1]),
        static_cast<uint64_t>(s[2]),
        static_cast<uint64_t>(s[3]),
        static_cast<uint64_t>(s[4]),
        gpu_kernel_ns[il].load(std::memory_order_relaxed),
    };
}

void reset_gpu_stats_and_sheets() {
    std::lock_guard<std::mutex> guard(gpu_sheet_lock);
    if (h_mapped_stats) {
        cudaDeviceSynchronize();
        std::memset(h_mapped_stats, 0, ARGUS_MAX_TRACKED_LAYERS * 5 * sizeof(unsigned long long));
    }
    for (int i = 0; i < ARGUS_MAX_TRACKED_LAYERS; ++i) {
        gpu_kernel_ns[i].store(0, std::memory_order_relaxed);
    }
}

unsigned long long * ensure_mapped_stats() {
    std::lock_guard<std::mutex> guard(gpu_sheet_lock);
    if (!h_mapped_stats) {
        const size_t bytes = ARGUS_MAX_TRACKED_LAYERS * 5 * sizeof(unsigned long long);
        if (cudaHostAlloc(&h_mapped_stats, bytes, cudaHostAllocMapped) == cudaSuccess) {
            std::memset(h_mapped_stats, 0, bytes);
            cudaHostGetDevicePointer(&d_mapped_stats, h_mapped_stats, 0);
            argus_register_external_stats(get_gpu_layer_stats, reset_gpu_stats_and_sheets);
        }
    }
    return d_mapped_stats;
}

GpuKeyBoxSheet * get_gpu_box_sheet(const char * k_base, int heads_kv, int dk, int cells, int tile_t) {
    const int needed = std::max(256, (cells + tile_t - 1) / tile_t);
    std::lock_guard<std::mutex> guard(gpu_sheet_lock);
    for (auto & s : gpu_box_sheets) {
        if (s.k_base == k_base && s.heads_kv == heads_kv && s.dk == dk && s.max_blocks >= needed) {
            return &s;
        }
    }
    if (gpu_box_sheets.size() >= 64) {
        gpu_box_sheets.pop_front();
    }
    gpu_box_sheets.emplace_back(k_base, heads_kv, dk, std::max(needed, 1024));
    return &gpu_box_sheets.back();
}

struct Problem {
    const char * q; size_t q_head, q_token;
    const char * k; size_t k_head, k_cell;
    const char * v; size_t v_head, v_cell;
    const char * mask; size_t mask_token; bool mask_f16; long long mask_cell0;
    int cells, rows, group, heads_kv;
    float scale, skip_eps;
    bool multipole, top_r;
    int layer_il, max_blocks;
    uint64_t * box_tags;
    __half2 * boxes;
    __half2 * v_boxes;
    float * tail_norms;
    unsigned long long * stats_out;
};

__device__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) { x += __shfl_xor_sync(0xffffffffu, x, offset); }
    return x;
}

__device__ float warp_max(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) { x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset)); }
    return x;
}

__device__ float bias_at(const Problem & p, int token, int cell) {
    const char * row = p.mask + token * p.mask_token;
    const long long column = p.mask_cell0 + cell;
    return p.mask_f16 ? __half2float(reinterpret_cast<const __half *>(row)[column])
                      : reinterpret_cast<const float *>(row)[column];
}

__device__ uint64_t gpu_block_tag(const char * base, size_t stride_bytes, int n) {
    const uint64_t w0 = *reinterpret_cast<const uint64_t *>(base);
    const uint64_t wn = *reinterpret_cast<const uint64_t *>(base + (n - 1) * stride_bytes);
    const uint64_t h = (w0 ^ ((wn << 13) | (wn >> 51))) * 0x9e3779b97f4a7c15ULL;
    return ((h >> 16) | 1ULL) | (static_cast<uint64_t>(n) << 48);
}

// n rows of one KV head, from cell `first`, into the shared tile (16-byte global loads).
template <int D>
__device__ void load_tile(const char * base, size_t head_stride, size_t cell_stride, int head_kv, int first, int n,
                          __half2 (*tile)[row_words<D>]) {
    constexpr int units = D / 8;  // 16-byte units per row
    for (int i = threadIdx.x; i < n * units; i += blockDim.x) {
        const int c = i / units, u = i % units;
        const uint4 raw = *reinterpret_cast<const uint4 *>(base + head_kv * head_stride + (first + c) * cell_stride + u * 16);
        const __half2 * pairs = reinterpret_cast<const __half2 *>(&raw);
        #pragma unroll
        for (int j = 0; j < 4; ++j) { tile[c][4 * u + j] = pairs[j]; }
    }
}

template <int D>
__device__ void load_tile_masked(const char * base, size_t head_stride, size_t cell_stride, int head_kv, int first, int n,
                                 uint32_t active_mask, __half2 (*tile)[row_words<D>]) {
    constexpr int units = D / 8;
    for (int i = threadIdx.x; i < n * units; i += blockDim.x) {
        const int c = i / units, u = i % units;
        if (!(active_mask & (1u << c))) { continue; }
        const uint4 raw = *reinterpret_cast<const uint4 *>(base + head_kv * head_stride + (first + c) * cell_stride + u * 16);
        const __half2 * pairs = reinterpret_cast<const __half2 *>(&raw);
        #pragma unroll
        for (int j = 0; j < 4; ++j) { tile[c][4 * u + j] = pairs[j]; }
    }
}

// grid (chunks, heads_kv, row groups), block 8 warps. partials[chunk][head_kv * rows + row][D + 2].
template <int D>
__global__ void __launch_bounds__(rows_per_block * 32) decode_partials(Problem p, float * partials) {
    constexpr int T = tile_cells<D>, lanes_per_cell = 32 / T, E = D / 32;
    constexpr int part_words = D / 2 / lanes_per_cell;
    __shared__ float q_shared[rows_per_block][D];
    __shared__ __half2 tile[T][row_words<D>];
    __shared__ float weights[rows_per_block][T];
    __shared__ float cell_floors[rows_per_block];
    __shared__ float shared_r_sq[rows_per_block];
    __shared__ uint32_t shared_active_mask;
    __shared__ int shared_last_valid_cell;

    const int lane = threadIdx.x % 32, r = threadIdx.x / 32;
    const int chunk = blockIdx.x, head_kv = blockIdx.y, row0 = blockIdx.z * rows_per_block;
    const int rows = min(rows_per_block, p.rows - row0);
    const bool active = r < rows;
    const int row = row0 + r, token = active ? row / p.group : 0;
    for (int i = threadIdx.x; i < rows * D; i += blockDim.x) {
        const int rr = i / D, d = i % D, rw = row0 + rr;
        const int t = rw / p.group, head = head_kv * p.group + rw % p.group;
        q_shared[rr][d] = reinterpret_cast<const float *>(p.q + head * p.q_head + t * p.q_token)[d];
    }

    if (r == 0) {
        int found = 0;
        const int min_tok = row0 / p.group;
        const int max_tok = (row0 + rows - 1) / p.group;
        for (int c_base = p.cells - 1; c_base >= 0; c_base -= 32) {
            const int c_idx = c_base - lane;
            bool ok = false;
            if (c_idx >= 0) {
                for (int tk = min_tok; tk <= max_tok; ++tk) {
                    if (bias_at(p, tk, c_idx) != -INFINITY) { ok = true; break; }
                }
            }
            const uint32_t m_ok = __ballot_sync(0xffffffffu, ok);
            if (m_ok != 0u) {
                found = c_base - (__ffs(m_ok) - 1);
                break;
            }
        }
        if (lane == 0) { shared_last_valid_cell = found; }
    }
    __syncthreads();
    const int last_valid_cell = shared_last_valid_cell;
    const int last_blk = last_valid_cell / T;

    float m = -INFINITY, l = 0.0f, acc[E];
    #pragma unroll
    for (int e = 0; e < E; ++e) { acc[e] = 0.0f; }
    const int c0 = chunk * chunk_cells, c1 = min(min(c0 + chunk_cells, p.cells), last_valid_cell + 1);
    const int cell = lane / lanes_per_cell, part = lane % lanes_per_cell;

    float wm_m = -INFINITY, wm_l = 0.0f;
    if (p.skip_eps > 0.0f && last_valid_cell >= 2 * T) {
        const int last_ref_t0 = last_blk * T;
        int refs[3] = {0, last_ref_t0 - T, last_ref_t0};
        int n_refs = 0;
        refs[n_refs++] = 0;
        if (last_blk >= 2) { refs[n_refs++] = (last_blk - 1) * T; }
        if (last_blk >= 1) { refs[n_refs++] = last_blk * T; }
        for (int ri = 0; ri < n_refs; ++ri) {
            const int t_ref = refs[ri], n_ref = min(T, last_valid_cell + 1 - t_ref);
            __syncthreads();
            load_tile<D>(p.k, p.k_head, p.k_cell, head_kv, t_ref, n_ref, tile);
            __syncthreads();
            if (active) {
                float dot = 0.0f;
                if (cell < n_ref) {
                    const float2 * q2 = reinterpret_cast<const float2 *>(q_shared[r]);
                    #pragma unroll 8
                    for (int w = part * part_words; w < (part + 1) * part_words; ++w) {
                        const float2 k2 = __half22float2(tile[cell][w]);
                        const float2 qq = q2[w];
                        dot = fmaf(qq.x, k2.x, fmaf(qq.y, k2.y, dot));
                    }
                }
                #pragma unroll
                for (int offset = lanes_per_cell / 2; offset > 0; offset /= 2) { dot += __shfl_xor_sync(0xffffffffu, dot, offset); }
                float score = -INFINITY;
                if (cell < n_ref) {
                    const float bias = bias_at(p, token, t_ref + cell);
                    if (bias != -INFINITY) { score = fmaf(dot, p.scale, bias); }
                }
                const float top = fmaxf(wm_m, warp_max(score));
                const float weight = score == -INFINITY ? 0.0f : expf(score - top);
                const float keep = wm_m == -INFINITY ? 0.0f : expf(wm_m - top);
                wm_l = wm_l * keep + warp_sum(part == 0 ? weight : 0.0f);
                wm_m = top;
            }
        }
    }

    int best_e = 0;
    float q_tail_norm = 0.0f;
    if (active && p.skip_eps > 0.0f) {
        float max_abs = -1.0f, tail_sq = 0.0f;
        #pragma unroll
        for (int e = 0; e < E; ++e) {
            const float qv = q_shared[r][e * 32 + lane];
            const float aq = fabsf(qv);
            tail_sq = fmaf(qv, qv, tail_sq);
            if (aq > max_abs) { max_abs = aq; best_e = e; }
        }
        tail_sq = fmaxf(0.0f, tail_sq - max_abs * max_abs);
        q_tail_norm = sqrtf(warp_sum(tail_sq));
    }

    unsigned int k_tot = 0, k_skip = 0, v_blk_skip = 0, v_tot = 0, v_skip = 0;

    for (int t0 = c0; t0 < c1; t0 += T) {
        const int n = min(T, c1 - t0);
        const int blk = t0 / T;
        if (threadIdx.x == 0) {
            ++k_tot;
            v_tot += static_cast<unsigned int>(n);
        }
        uint64_t tag = 0;
        size_t slot = 0;
        const bool can_box = (p.skip_eps > 0.0f) && (0 < blk) && (blk < last_blk - 1) && (n == T) &&
                             (p.box_tags != nullptr) && (blk < p.max_blocks);
        bool has_box = false;
        if (can_box) {
            slot = static_cast<size_t>(head_kv) * p.max_blocks + blk;
            tag = gpu_block_tag(p.k + head_kv * p.k_head + t0 * p.k_cell, p.k_cell, n);
            has_box = (p.box_tags[slot] == tag);
            if (has_box) {
                bool mono_ok = true;
                float c_dot = 0.0f, mean_score = -INFINITY, exp_mean = 0.0f, mono_mass = 0.0f, max_bias = -INFINITY;
                if (active) {
                    const float b_val = cell < n ? bias_at(p, token, t0 + cell) : -INFINITY;
                    max_bias = warp_max(b_val);
                    if (max_bias != -INFINITY) {
                        const __half2 * box_mu = p.boxes + slot * D;
                        const float2 * q2 = reinterpret_cast<const float2 *>(q_shared[r]);
                        #pragma unroll
                        for (int e2 = 0; e2 < E / 2; ++e2) {
                            const float2 mv = __half22float2(box_mu[e2 * 32 + lane]);
                            const float2 qv = q2[e2 * 32 + lane];
                            c_dot = fmaf(qv.x, mv.x, fmaf(qv.y, mv.y, c_dot));
                        }
                        c_dot = warp_sum(c_dot);
                        mean_score = fmaf(c_dot, p.scale, max_bias);
                        exp_mean = expf(mean_score - wm_m);
                        mono_mass = float(n) * exp_mean;
                        mono_ok = (wm_l > 0.0f && mean_score < wm_m && mono_mass <= 16.0f * p.skip_eps * wm_l);
                    }
                }
                if (__syncthreads_and(!active || max_bias == -INFINITY || mono_ok)) {
                    bool row_skip = true;
                    if (active && max_bias != -INFINITY) {
                        const __half2 * box_rad = p.boxes + slot * D + (D / 2);
                        float ub_dot = c_dot;
                        if (p.top_r) {
                            const int top_idx = best_e * 32 + lane;
                            const float q_top = q_shared[r][top_idx];
                            const float r_top = __half2float(reinterpret_cast<const __half *>(box_rad)[top_idx]);
                            const float l1_top = warp_sum(fabsf(q_top * r_top));
                            const float tail_bound = p.tail_norms ? q_tail_norm * p.tail_norms[slot] : INFINITY;
                            ub_dot = c_dot + l1_top + tail_bound;
                        } else {
                            const float2 * q2 = reinterpret_cast<const float2 *>(q_shared[r]);
                            float l1_r = 0.0f, l2_r = 0.0f;
                            #pragma unroll
                            for (int e2 = 0; e2 < E / 2; ++e2) {
                                const float2 rv = __half22float2(box_rad[e2 * 32 + lane]);
                                const float2 qv = q2[e2 * 32 + lane];
                                const float qr_x = fabsf(qv.x * rv.x);
                                const float qr_y = fabsf(qv.y * rv.y);
                                l1_r += qr_x + qr_y;
                                l2_r = fmaf(qr_x, qr_x, fmaf(qr_y, qr_y, l2_r));
                            }
                            l1_r = warp_sum(l1_r);
                            l2_r = warp_sum(l2_r);
                            ub_dot = c_dot + fminf(l1_r, 1.15f * sqrtf(l2_r));
                        }
                        const float ub_score = fmaf(ub_dot, p.scale, max_bias);
                        const float n_tail = float(min(4, n));
                        const float exp_ub = expf(ub_score - wm_m);
                        const float dipole_mass = n_tail * fmaxf(0.0f, exp_ub - exp_mean);
                        row_skip = (ub_score < wm_m && dipole_mass <= p.skip_eps * wm_l);
                    }
                    if (__syncthreads_and(!active || max_bias == -INFINITY || row_skip)) {
                        if (active && p.multipole && max_bias != -INFINITY && mono_mass > 1e-6f * p.skip_eps * wm_l && p.v_boxes != nullptr) {
                            const float top = fmaxf(m, mean_score);
                            const float keep = m == -INFINITY ? 0.0f : expf(m - top);
                            const float w_mp = float(n) * expf(mean_score - top);
                            l = l * keep + w_mp;
                            m = top;
                            const __half2 * v_box = p.v_boxes + slot * (D / 2);
                            #pragma unroll
                            for (int e2 = 0; e2 < E / 2; ++e2) {
                                const float2 mv = __half22float2(v_box[e2 * 32 + lane]);
                                acc[2 * e2 + 0] = fmaf(w_mp, mv.x, acc[2 * e2 + 0] * keep);
                                acc[2 * e2 + 1] = fmaf(w_mp, mv.y, acc[2 * e2 + 1] * keep);
                            }
                        }
                        if (threadIdx.x == 0) {
                            ++k_skip;
                            ++v_blk_skip;
                            v_skip += static_cast<unsigned int>(n);
                        }
                        continue;
                    }
                }
            }
        }
        const bool need_build_box = can_box && !has_box;
        __syncthreads();  // the previous tile's V is no longer read
        load_tile<D>(p.k, p.k_head, p.k_cell, head_kv, t0, n, tile);
        __syncthreads();
        if (need_build_box) {
            __half2 * box_mu  = p.boxes + slot * D;
            __half2 * box_rad = box_mu + (D / 2);
            const float inv_n = 1.0f / float(n);
            float r_sq_local = 0.0f;
            for (int w = threadIdx.x; w < D / 2; w += blockDim.x) {
                float2 vmin = __half22float2(tile[0][w]);
                float2 vmax = vmin;
                float2 vsum = vmin;
                for (int c = 1; c < n; ++c) {
                    const float2 kv = __half22float2(tile[c][w]);
                    vmin.x = fminf(vmin.x, kv.x); vmin.y = fminf(vmin.y, kv.y);
                    vmax.x = fmaxf(vmax.x, kv.x); vmax.y = fmaxf(vmax.y, kv.y);
                    vsum.x += kv.x;               vsum.y += kv.y;
                }
                const float mu_x = vsum.x * inv_n, rad_x = fmaxf(vmax.x - mu_x, mu_x - vmin.x);
                const float mu_y = vsum.y * inv_n, rad_y = fmaxf(vmax.y - mu_y, mu_y - vmin.y);
                box_mu[w]  = __floats2half2_rn(mu_x, mu_y);
                box_rad[w] = __floats2half2_rn(rad_x, rad_y);
                r_sq_local = fmaf(rad_x, rad_x, fmaf(rad_y, rad_y, r_sq_local));
            }
            r_sq_local = warp_sum(r_sq_local);
            if (lane == 0) { shared_r_sq[r] = r_sq_local; }
            __syncthreads();
            if (threadIdx.x == 0 && p.tail_norms != nullptr) {
                float tot_r_sq = 0.0f;
                for (int rr = 0; rr < rows_per_block; ++rr) { tot_r_sq += shared_r_sq[rr]; }
                p.tail_norms[slot] = sqrtf(tot_r_sq);
            }
        }
        float score = -INFINITY;
        float tile_max = -INFINITY;
        if (active) {
            float dot = 0.0f;
            if (cell < n) {
                const float2 * q2 = reinterpret_cast<const float2 *>(q_shared[r]);
                #pragma unroll 8
                for (int w = part * part_words; w < (part + 1) * part_words; ++w) {
                    const float2 k2 = __half22float2(tile[cell][w]);
                    const float2 qq = q2[w];
                    dot = fmaf(qq.x, k2.x, fmaf(qq.y, k2.y, dot));
                }
            }
            #pragma unroll
            for (int offset = lanes_per_cell / 2; offset > 0; offset /= 2) { dot += __shfl_xor_sync(0xffffffffu, dot, offset); }
            if (cell < n) {
                const float bias = bias_at(p, token, t0 + cell);
                if (bias != -INFINITY) { score = fmaf(dot, p.scale, bias); }
            }
            tile_max = warp_max(score);
        }
        if (p.skip_eps > 0.0f && !need_build_box) {
            const bool v_tile_skip = !active || (tile_max == -INFINITY) ||
                (wm_l > 0.0f && tile_max < wm_m && float(n) * expf(tile_max - wm_m) <= p.skip_eps * wm_l);
            if (__syncthreads_and(v_tile_skip)) {
                if (active && p.multipole && has_box && tile_max != -INFINITY && p.v_boxes != nullptr) {
                    const float top = fmaxf(m, tile_max);
                    const float weight = score == -INFINITY ? 0.0f : expf(score - top);
                    const float keep = m == -INFINITY ? 0.0f : expf(m - top);
                    const float tile_sum = warp_sum(part == 0 ? weight : 0.0f);
                    if (tile_sum > 0.0f) {
                        l = l * keep + tile_sum;
                        m = top;
                        const __half2 * v_box = p.v_boxes + slot * (D / 2);
                        #pragma unroll
                        for (int e2 = 0; e2 < E / 2; ++e2) {
                            const float2 mv = __half22float2(v_box[e2 * 32 + lane]);
                            acc[2 * e2 + 0] = fmaf(tile_sum, mv.x, acc[2 * e2 + 0] * keep);
                            acc[2 * e2 + 1] = fmaf(tile_sum, mv.y, acc[2 * e2 + 1] * keep);
                        }
                    }
                }
                if (threadIdx.x == 0) {
                    ++v_blk_skip;
                    v_skip += static_cast<unsigned int>(n);
                }
                continue;
            }
        }
        float row_tile_sum = 0.0f;
        if (active) {
            const float top = fmaxf(m, tile_max);
            const float weight = score == -INFINITY ? 0.0f : expf(score - top);
            const float keep = m == -INFINITY ? 0.0f : expf(m - top);
            const float tile_sum = warp_sum(part == 0 ? weight : 0.0f);
            row_tile_sum = tile_sum;
            l = l * keep + tile_sum;
            m = top;
            #pragma unroll
            for (int e = 0; e < E; ++e) { acc[e] *= keep; }
            if (part == 0) { weights[r][cell] = weight; }
            if (p.skip_eps > 0.0f) {
                if (blk > 0 && blk < last_blk - 1) {
                    const float wm_top = fmaxf(wm_m, tile_max);
                    wm_l = wm_l * (wm_m == -INFINITY ? 0.0f : expf(wm_m - wm_top)) +
                           tile_sum * (top == -INFINITY ? 0.0f : expf(top - wm_top));
                    wm_m = wm_top;
                }
                if (lane == 0) {
                    const float denom = fmaxf(l, wm_l * expf(wm_m - m));
                    cell_floors[r] = (p.skip_eps / float(T)) * denom;
                }
            }
        }
        __syncthreads();  // every warp is done with K
        if (p.skip_eps > 0.0f && T == 32 && !need_build_box) {
            if (r == 0) {
                bool cell_hot = false;
                if (lane < n) {
                    for (int rr = 0; rr < rows; ++rr) {
                        if (weights[rr][lane] > cell_floors[rr]) { cell_hot = true; break; }
                    }
                }
                const uint32_t mask = __ballot_sync(0xffffffffu, cell_hot);
                if (lane == 0) { shared_active_mask = mask; }
            }
            __syncthreads();
            const uint32_t active_mask = shared_active_mask;
            if (active_mask == 0u) {
                if (active && p.multipole && has_box && row_tile_sum > 0.0f && p.v_boxes != nullptr) {
                    const __half2 * v_box = p.v_boxes + slot * (D / 2);
                    #pragma unroll
                    for (int e2 = 0; e2 < E / 2; ++e2) {
                        const float2 mv = __half22float2(v_box[e2 * 32 + lane]);
                        acc[2 * e2 + 0] = fmaf(row_tile_sum, mv.x, acc[2 * e2 + 0]);
                        acc[2 * e2 + 1] = fmaf(row_tile_sum, mv.y, acc[2 * e2 + 1]);
                    }
                }
                if (threadIdx.x == 0) {
                    ++v_blk_skip;
                    v_skip += static_cast<unsigned int>(n);
                }
                continue;
            }
            if (threadIdx.x == 0) {
                v_skip += static_cast<unsigned int>(n - __popc(active_mask));
            }
            load_tile_masked<D>(p.v, p.v_head, p.v_cell, head_kv, t0, n, active_mask, tile);
            __syncthreads();
            if (active) {
                float hot_sum = 0.0f;
                uint32_t mask_iter = active_mask;
                while (mask_iter) {
                    const int c = __ffs(mask_iter) - 1;
                    mask_iter &= mask_iter - 1;
                    const float weight = weights[r][c];
                    if (weight == 0.0f) { continue; }
                    hot_sum += weight;
                    #pragma unroll
                    for (int e2 = 0; e2 < E / 2; ++e2) {
                        const float2 v2 = __half22float2(tile[c][e2 * 32 + lane]);
                        acc[2 * e2 + 0] = fmaf(weight, v2.x, acc[2 * e2 + 0]);
                        acc[2 * e2 + 1] = fmaf(weight, v2.y, acc[2 * e2 + 1]);
                    }
                }
                const float skipped_sum = fmaxf(0.0f, row_tile_sum - hot_sum);
                if (p.multipole && has_box && skipped_sum > 0.0f && p.v_boxes != nullptr) {
                    const __half2 * v_box = p.v_boxes + slot * (D / 2);
                    #pragma unroll
                    for (int e2 = 0; e2 < E / 2; ++e2) {
                        const float2 mv = __half22float2(v_box[e2 * 32 + lane]);
                        acc[2 * e2 + 0] = fmaf(skipped_sum, mv.x, acc[2 * e2 + 0]);
                        acc[2 * e2 + 1] = fmaf(skipped_sum, mv.y, acc[2 * e2 + 1]);
                    }
                }
            }
            continue;
        }
        load_tile<D>(p.v, p.v_head, p.v_cell, head_kv, t0, n, tile);
        __syncthreads();
        if (need_build_box) {
            if (p.v_boxes != nullptr) {
                __half2 * v_box = p.v_boxes + slot * (D / 2);
                const float inv_n = 1.0f / float(n);
                for (int w = threadIdx.x; w < D / 2; w += blockDim.x) {
                    float2 vsum = __half22float2(tile[0][w]);
                    for (int c = 1; c < n; ++c) {
                        const float2 vv = __half22float2(tile[c][w]);
                        vsum.x += vv.x;
                        vsum.y += vv.y;
                    }
                    v_box[w] = __floats2half2_rn(vsum.x * inv_n, vsum.y * inv_n);
                }
            }
            __threadfence();
            if (threadIdx.x == 0) { p.box_tags[slot] = tag; }
        }
        if (active) {
            for (int c = 0; c < n; ++c) {
                const float weight = weights[r][c];
                if (weight == 0.0f) { continue; }
                #pragma unroll
                for (int e2 = 0; e2 < E / 2; ++e2) {
                    const float2 v2 = __half22float2(tile[c][e2 * 32 + lane]);
                    acc[2 * e2 + 0] = fmaf(weight, v2.x, acc[2 * e2 + 0]);
                    acc[2 * e2 + 1] = fmaf(weight, v2.y, acc[2 * e2 + 1]);
                }
            }
        }
    }
    if (threadIdx.x == 0 && p.stats_out && p.layer_il >= 0 && p.layer_il < ARGUS_MAX_TRACKED_LAYERS) {
        unsigned long long * dst_stats = p.stats_out + p.layer_il * 5;
        atomicAdd(&dst_stats[0], static_cast<unsigned long long>(k_tot));
        atomicAdd(&dst_stats[1], static_cast<unsigned long long>(k_skip));
        atomicAdd(&dst_stats[2], static_cast<unsigned long long>(v_blk_skip));
        atomicAdd(&dst_stats[3], static_cast<unsigned long long>(v_tot));
        atomicAdd(&dst_stats[4], static_cast<unsigned long long>(v_skip));
    }
    if (active) {
        float * out = partials + (static_cast<size_t>(chunk) * p.heads_kv * p.rows + head_kv * p.rows + row) * (D + 2);
        if (lane == 0) { out[0] = m; out[1] = l; }
        #pragma unroll
        for (int e2 = 0; e2 < E / 2; ++e2) {
            const int base_d = 2 * (e2 * 32 + lane);
            out[2 + base_d + 0] = acc[2 * e2 + 0];
            out[2 + base_d + 1] = acc[2 * e2 + 1];
        }
    }
}

// grid (rows total), block 256: merge the chunks of one row in chunk order. With group > 0
// the row is normalized and written to its (head, token) place in [dv, n_head, n_tokens];
// otherwise it stays a partial in [rows][dv + 2].
__global__ void reduce_partials(const float * partials, float * out, int chunks, int rows_total, int dv,
                                int group, int n_head, int rows_per_head) {
    const int row = blockIdx.x, stride = dv + 2;
    float max = -INFINITY, sum = 0.0f;
    float acc[2] = {0.0f, 0.0f};  // dv <= 512 over 256 threads
    for (int j = 0; j < chunks; ++j) {
        const float * partial = partials + (static_cast<size_t>(j) * rows_total + row) * stride;
        if (partial[1] == 0.0f) { continue; }
        const float merged = fmaxf(max, partial[0]);
        const float keep = expf(max - merged), add = expf(partial[0] - merged);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int d = threadIdx.x + i * blockDim.x;
            if (d < dv) { acc[i] = acc[i] * keep + partial[2 + d] * add; }
        }
        sum = sum * keep + partial[1] * add;
        max = merged;
    }
    if (group > 0) {
        const int head_kv = row / rows_per_head, r = row % rows_per_head;
        const int token = r / group, head = head_kv * group + r % group;
        const float inverse = sum == 0.0f ? 0.0f : 1.0f / sum;
        float * result = out + (static_cast<size_t>(token) * n_head + head) * dv;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int d = threadIdx.x + i * blockDim.x;
            if (d < dv) { result[d] = acc[i] * inverse; }
        }
        return;
    }
    float * result = out + static_cast<size_t>(row) * stride;
    if (threadIdx.x == 0) { result[0] = max; result[1] = sum; }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int d = threadIdx.x + i * blockDim.x;
        if (d < dv) { result[2 + d] = acc[i]; }
    }
}

// grid (rows, ceil(width / 256)), block 256: row r of `rows` (F32) goes, rounded to F16 like
// GGML's CPU conversion (to nearest, ties to even), to mirror row min(indices[r], cells)
// and to host row indices[r] (F16, Q8_0, or Q4_0).
__global__ void tier_append(const char * rows, size_t row_stride, const int64_t * indices, int width,
                            char * host, size_t host_row_bytes, int host_type, __half * mirror, long long cells) {
    const int r = blockIdx.x, e = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= width) { return; }
    const long long target = indices[r];
    const float x = reinterpret_cast<const float *>(rows + r * row_stride)[e];
    const __half value = __float2half_rn(x);
    mirror[min(target, cells) * width + e] = value;
    if (host_type == GGML_TYPE_F16) {
        reinterpret_cast<__half *>(host + target * host_row_bytes)[e] = value;
        return;
    }
    const int lane = threadIdx.x % 32, blk = e / 32;
    if (host_type == GGML_TYPE_Q8_0) {
        const float amax = warp_max(fabsf(x));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        char * dst = host + target * host_row_bytes + blk * 34;
        if (lane == 0) { *reinterpret_cast<__half *>(dst) = __float2half_rn(d); }
        reinterpret_cast<int8_t *>(dst + 2)[lane] = static_cast<int8_t>(roundf(x * id));
    } else if (host_type == GGML_TYPE_Q4_0) {
        float amax = fabsf(x), vmax = x;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            const float o_amax = __shfl_xor_sync(0xffffffffu, amax, offset);
            const float o_vmax = __shfl_xor_sync(0xffffffffu, vmax, offset);
            const int o_lane = lane ^ offset;
            if (o_amax > amax || (o_amax == amax && o_lane < lane)) { amax = o_amax; vmax = o_vmax; }
        }
        const float d = vmax / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const float x_hi = __shfl_down_sync(0xffffffffu, x, 16);
        char * dst = host + target * host_row_bytes + blk * 18;
        if (lane == 0) { *reinterpret_cast<__half *>(dst) = __float2half_rn(d); }
        if (lane < 16) {
            const uint8_t q0 = static_cast<uint8_t>(min(15, max(0, static_cast<int>(x * id + 8.5f))));
            const uint8_t q1 = static_cast<uint8_t>(min(15, max(0, static_cast<int>(x_hi * id + 8.5f))));
            reinterpret_cast<uint8_t *>(dst + 2)[lane] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
}

Problem problem_of(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    const ArgusConfig cfg = argus_get_config();
    const float skip_eps = cfg.skip_eps;
    const int layer_il = argus_parse_layer_il(k);
    const int cells = static_cast<int>(k->ne[2]);
    const int heads_kv = static_cast<int>(k->ne[1]);
    const int dk = static_cast<int>(k->ne[0]);
    const int tile_t = dk >= 512 ? 16 : 32;
    GpuKeyBoxSheet * sheet = skip_eps > 0.0f
        ? get_gpu_box_sheet(static_cast<const char *>(k->data), heads_kv, dk, cells, tile_t)
        : nullptr;
    unsigned long long * stats_out = ensure_mapped_stats();
    return {static_cast<const char *>(q->data), q->nb[1], q->nb[2],
            static_cast<const char *>(k->data), k->nb[1], k->nb[2],
            static_cast<const char *>(v->data), v->nb[1], v->nb[2],
            static_cast<const char *>(mask->data), mask->nb[1], mask->type == GGML_TYPE_F16,
            static_cast<long long>(argus_cuda_external_extra(dst)),
            cells, static_cast<int>(q->ne[2] * (q->ne[1] / k->ne[1])),
            static_cast<int>(q->ne[1] / k->ne[1]), heads_kv,
            argus_cuda_external_scale(dst), skip_eps, cfg.multipole_enabled, cfg.top_r_enabled, layer_il,
            sheet ? sheet->max_blocks : 0,
            sheet ? sheet->d_tags : nullptr,
            sheet ? sheet->d_boxes : nullptr,
            sheet ? sheet->d_v_boxes : nullptr,
            sheet ? sheet->d_tail_norms : nullptr,
            stats_out};
}

template <int D>
void launch_partials(const Problem & p, float * partials, cudaStream_t stream) {
    const dim3 grid((p.cells + chunk_cells - 1) / chunk_cells, p.heads_kv, (p.rows + rows_per_block - 1) / rows_per_block);
    decode_partials<D><<<grid, rows_per_block * 32, 0, stream>>>(p, partials);
}
} // namespace

void argus_cuda_split_compute(ggml_tensor * dst, int device, void * raw_stream) {
    (void) device;
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    if (argus_cuda_external_kind(dst) == argus_kind_tier_append) {
        // src[0] the mirror, src[1] the F32 rows, src[2] their I64 cell indices (on the device);
        // extra the host tensor, in pinned memory the device writes through its mapping.
        const ggml_tensor * mirror = dst->src[0], * rows = dst->src[1], * indices = dst->src[2];
        const auto * host = reinterpret_cast<const ggml_tensor *>(argus_cuda_external_extra(dst));
        const int width = static_cast<int>(mirror->ne[0]);
        const dim3 grid(static_cast<unsigned>(rows->ne[1]), static_cast<unsigned>((width + 255) / 256));
        tier_append<<<grid, 256, 0, stream>>>(static_cast<const char *>(rows->data), rows->nb[1],
            static_cast<const int64_t *>(indices->data), width, static_cast<char *>(host->data), host->nb[1],
            static_cast<int>(host->type), static_cast<__half *>(mirror->data), static_cast<long long>(mirror->ne[1] - 1));
        check(cudaGetLastError());
        return;
    }
    if (argus_cuda_external_kind(dst) == argus_kind_partials_reduce) {
        // extra holds the query group of a final (normalized) result, 0 for a partial.
        const ggml_tensor * partials = dst->src[0];
        const int group = static_cast<int>(argus_cuda_external_extra(dst));
        const int rows_total = static_cast<int>(partials->ne[1]), dv = static_cast<int>(partials->ne[0] - 2);
        const int n_head = group ? static_cast<int>(dst->ne[1]) : 0;
        const int rows_per_head = group ? rows_total / (n_head / group) : 0;
        const char * prof = std::getenv("ARGUS_PROFILE_KERNELS");
        const bool do_prof = prof && std::strcmp(prof, "1") == 0;
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        if (do_prof) {
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev0, stream);
        }
        reduce_partials<<<static_cast<unsigned>(rows_total), 256, 0, stream>>>(
            static_cast<const float *>(partials->data), static_cast<float *>(dst->data),
            static_cast<int>(partials->ne[2]), rows_total, dv, group, n_head, rows_per_head);
        check(cudaGetLastError());
        if (do_prof) {
            cudaEventRecord(ev1, stream);
            cudaEventSynchronize(ev1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
            const int il = partials->src[1] ? argus_parse_layer_il(partials->src[1]) : 0;
            if (il >= 0 && il < ARGUS_MAX_TRACKED_LAYERS) {
                gpu_kernel_ns[il].fetch_add(static_cast<uint64_t>(ms * 1e6f), std::memory_order_relaxed);
            }
        }
        return;
    }
    const Problem p = problem_of(dst);
    auto * partials = static_cast<float *>(dst->data);
    const char * prof = std::getenv("ARGUS_PROFILE_KERNELS");
    const bool do_prof = prof && std::strcmp(prof, "1") == 0;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (do_prof) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream);
    }
    switch (dst->src[1]->ne[0]) {
        case 64: launch_partials<64>(p, partials, stream); break;
        case 128: launch_partials<128>(p, partials, stream); break;
        case 256: launch_partials<256>(p, partials, stream); break;
        case 512: launch_partials<512>(p, partials, stream); break;
        default: throw std::runtime_error("ARGUS split attention: unsupported head dimension");
    }
    check(cudaGetLastError());
    if (do_prof) {
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        if (p.layer_il >= 0 && p.layer_il < ARGUS_MAX_TRACKED_LAYERS) {
            gpu_kernel_ns[p.layer_il].fetch_add(static_cast<uint64_t>(ms * 1e6f), std::memory_order_relaxed);
        }
    }
}

namespace {
ggml_tensor * decode_graph(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
                           int64_t mask_cell0, float scale, bool final) {
    const int64_t d = k->ne[0];
    const bool supported =
        (d == 64 || d == 128 || d == 256 || d == 512) && v->ne[0] == d && q->ne[0] == d &&
        q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
        q->nb[0] == sizeof(float) && k->nb[0] == sizeof(ggml_fp16_t) && v->nb[0] == sizeof(ggml_fp16_t) &&
        q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1 && k->ne[1] == v->ne[1] && k->ne[2] == v->ne[2] &&
        k->ne[2] > 0 && q->ne[1] % k->ne[1] == 0 &&
        (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32) && mask->ne[1] >= q->ne[2] &&
        mask_cell0 >= 0 && mask->ne[0] >= mask_cell0 + k->ne[2] && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        k->nb[1] % 16 == 0 && k->nb[2] % 16 == 0 && v->nb[1] % 16 == 0 && v->nb[2] % 16 == 0;
    if (!supported) { return nullptr; }
    const int64_t rows_total = q->ne[1] * q->ne[2], group = q->ne[1] / k->ne[1];
    const int64_t chunks = (k->ne[2] + chunk_cells - 1) / chunk_cells;
    ggml_tensor * inputs[] = {q, k, v, mask};
    ggml_tensor * partials = argus_cuda_external_op(ctx, d + 2, rows_total, chunks, inputs, 4,
                                                    argus_kind_decode_partials, scale, mask_cell0);
    ggml_tensor * reduce_inputs[] = {partials};
    if (final) {
        return argus_cuda_external_op(ctx, d, q->ne[1], q->ne[2], reduce_inputs, 1, argus_kind_partials_reduce, 0.0f, group);
    }
    return argus_cuda_external_op(ctx, d + 2, rows_total, 1, reduce_inputs, 1, argus_kind_partials_reduce, 0.0f, 0);
}
} // namespace

ggml_tensor * argus_ggml_cuda_decode_partials(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                              ggml_tensor * mask, int64_t mask_cell0, float scale) {
    return decode_graph(ctx, q, k, v, mask, mask_cell0, scale, false);
}

ggml_tensor * argus_ggml_cuda_decode_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                               ggml_tensor * mask, float scale) {
    return decode_graph(ctx, q, k, v, mask, 0, scale, true);
}

ggml_tensor * argus_ggml_cuda_tier_append(ggml_context * ctx, ggml_tensor * mirror, ggml_tensor * rows,
                                          ggml_tensor * indices, const ggml_tensor * host) {
    const bool valid_host = host->type == GGML_TYPE_F16 ||
                            ((host->type == GGML_TYPE_Q8_0 || host->type == GGML_TYPE_Q4_0) && host->ne[0] % 32 == 0);
    if (mirror->type != GGML_TYPE_F16 || !valid_host || rows->type != GGML_TYPE_F32 ||
        indices->type != GGML_TYPE_I64 || rows->ne[0] != mirror->ne[0] || host->ne[0] != mirror->ne[0] ||
        rows->nb[0] != sizeof(float) || !ggml_is_contiguous(indices) || indices->ne[0] != rows->ne[1] ||
        rows->ne[2] != 1 || rows->ne[3] != 1 || host->nb[1] != ggml_row_size(host->type, host->ne[0])) {
        return nullptr;
    }
    ggml_tensor * args[] = {rows, indices};
    return argus_cuda_external_inplace(ctx, mirror, args, 2, argus_kind_tier_append, 0.0f,
                                       static_cast<int64_t>(reinterpret_cast<intptr_t>(host)));
}

namespace {
__global__ void dequant_mirror_kernel(const char * host, size_t host_row_bytes, int host_type, int width,
                                      __half * mirror) {
    const int c = blockIdx.x, e = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= width) { return; }
    const int blk = e / 32, lane = e % 32;
    const char * src = host + static_cast<size_t>(c) * host_row_bytes;
    float val = 0.0f;
    if (host_type == GGML_TYPE_Q8_0) {
        const char * b = src + blk * 34;
        const float d = __half2float(*reinterpret_cast<const __half *>(b));
        val = d * static_cast<float>(reinterpret_cast<const int8_t *>(b + 2)[lane]);
    } else if (host_type == GGML_TYPE_Q4_0) {
        const char * b = src + blk * 18;
        const float d = __half2float(*reinterpret_cast<const __half *>(b));
        const uint8_t byte = reinterpret_cast<const uint8_t *>(b + 2)[lane % 16];
        const int q = (lane < 16 ? (byte & 0x0F) : (byte >> 4)) - 8;
        val = d * static_cast<float>(q);
    }
    mirror[static_cast<size_t>(c) * width + e] = __float2half_rn(val);
}
} // namespace

bool argus_cuda_sync_mirror_from_host(const ggml_tensor * host, ggml_tensor * device, int64_t cells) {
    if (cells <= 0 || !host || !device || device->type != GGML_TYPE_F16 || host->ne[0] % 32 != 0 ||
        (host->type != GGML_TYPE_Q8_0 && host->type != GGML_TYPE_Q4_0)) {
        return false;
    }
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, host->data) != cudaSuccess || !attr.devicePointer) {
        cudaGetLastError();
        return false;
    }
    const int width = static_cast<int>(host->ne[0]);
    const dim3 grid(static_cast<unsigned>(cells), static_cast<unsigned>((width + 255) / 256));
    dequant_mirror_kernel<<<grid, 256>>>(static_cast<const char *>(attr.devicePointer), host->nb[1],
                                         static_cast<int>(host->type), width, static_cast<__half *>(device->data));
    check(cudaDeviceSynchronize());
    return true;
}
