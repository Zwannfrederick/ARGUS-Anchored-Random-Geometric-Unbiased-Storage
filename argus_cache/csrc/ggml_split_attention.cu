// S1, GPU side: decode partials over GPU-resident KV cells, then one merge per row.
// A block is 8 warps over one (KV head, chunk of cells, group of up to 8 query rows), one
// warp per row. Tiles of K, then of V, are staged in shared memory with coalesced loads;
// a lane computes whole dot products of its cell (no per-cell reductions), the softmax is
// updated once per tile, and each lane owns D/32 output elements. A second kernel merges
// the chunks of a row in order. Accumulation is F32 throughout.
#include "ggml_split_attention.h"
#include "ggml_cuda_external.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {
constexpr int rows_per_block = 8;  // one warp per query row
constexpr int chunk_cells = 256;

// Cells per tile, and lanes per cell so that a warp covers a tile.
template <int D> constexpr int tile_cells = D >= 512 ? 16 : 32;
// half2 words per shared tile row: odd, so lanes reading different rows at the same column
// fall in different banks.
template <int D> constexpr int row_words = D / 2 + 1;

void check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string("ARGUS split attention: ") + cudaGetErrorString(status)); }
}

struct Problem {
    const char * q; size_t q_head, q_token;
    const char * k; size_t k_head, k_cell;
    const char * v; size_t v_head, v_cell;
    const char * mask; size_t mask_token; bool mask_f16; long long mask_cell0;
    int cells, rows, group, heads_kv;
    float scale;
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

// grid (chunks, heads_kv, row groups), block 8 warps. partials[chunk][head_kv * rows + row][D + 2].
template <int D>
__global__ void __launch_bounds__(rows_per_block * 32) decode_partials(Problem p, float * partials) {
    constexpr int T = tile_cells<D>, lanes_per_cell = 32 / T, E = D / 32;
    constexpr int part_words = D / 2 / lanes_per_cell;
    __shared__ float q_shared[rows_per_block][D];
    __shared__ __half2 tile[T][row_words<D>];
    __shared__ float weights[rows_per_block][T];

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

    float m = -INFINITY, l = 0.0f, acc[E];
    #pragma unroll
    for (int e = 0; e < E; ++e) { acc[e] = 0.0f; }
    const int c0 = chunk * chunk_cells, c1 = min(c0 + chunk_cells, p.cells);
    const int cell = lane / lanes_per_cell, part = lane % lanes_per_cell;
    for (int t0 = c0; t0 < c1; t0 += T) {
        const int n = min(T, c1 - t0);
        __syncthreads();  // the previous tile's V is no longer read
        load_tile<D>(p.k, p.k_head, p.k_cell, head_kv, t0, n, tile);
        __syncthreads();
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
            float score = -INFINITY;
            if (cell < n) {
                const float bias = bias_at(p, token, t0 + cell);
                if (bias != -INFINITY) { score = fmaf(dot, p.scale, bias); }
            }
            const float top = fmaxf(m, warp_max(score));
            const float weight = score == -INFINITY ? 0.0f : expf(score - top);
            const float keep = m == -INFINITY ? 0.0f : expf(m - top);
            l = l * keep + warp_sum(part == 0 ? weight : 0.0f);
            m = top;
            #pragma unroll
            for (int e = 0; e < E; ++e) { acc[e] *= keep; }
            if (part == 0) { weights[r][cell] = weight; }
        }
        __syncthreads();  // every warp is done with K
        // ponytail: K then V through one tile, no overlap of loads and math; cp.async double
        // buffering is the upgrade (loading both tiles at once lowered occupancy: slower).
        load_tile<D>(p.v, p.v_head, p.v_cell, head_kv, t0, n, tile);
        __syncthreads();
        if (active) {
            for (int c = 0; c < n; ++c) {
                const float weight = weights[r][c];
                if (weight == 0.0f) { continue; }
                const __half * value = reinterpret_cast<const __half *>(tile[c]);
                #pragma unroll
                for (int e = 0; e < E; ++e) { acc[e] = fmaf(weight, __half2float(value[e * 32 + lane]), acc[e]); }
            }
        }
    }
    if (active) {
        float * out = partials + (static_cast<size_t>(chunk) * p.heads_kv * p.rows + head_kv * p.rows + row) * (D + 2);
        if (lane == 0) { out[0] = m; out[1] = l; }
        #pragma unroll
        for (int e = 0; e < E; ++e) { out[2 + e * 32 + lane] = acc[e]; }
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
// GGML's CPU conversion (to nearest, ties to even), to host row indices[r] and to mirror row
// min(indices[r], cells) (the mirror's last row is scratch).
__global__ void tier_append(const char * rows, size_t row_stride, const int64_t * indices, int width,
                            __half * host, __half * mirror, long long cells) {
    const int r = blockIdx.x, e = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= width) { return; }
    const long long target = indices[r];
    const __half value = __float2half_rn(reinterpret_cast<const float *>(rows + r * row_stride)[e]);
    host[target * width + e] = value;
    mirror[min(target, cells) * width + e] = value;
}

Problem problem_of(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    return {static_cast<const char *>(q->data), q->nb[1], q->nb[2],
            static_cast<const char *>(k->data), k->nb[1], k->nb[2],
            static_cast<const char *>(v->data), v->nb[1], v->nb[2],
            static_cast<const char *>(mask->data), mask->nb[1], mask->type == GGML_TYPE_F16,
            static_cast<long long>(argus_cuda_external_extra(dst)),
            static_cast<int>(k->ne[2]), static_cast<int>(q->ne[2] * (q->ne[1] / k->ne[1])),
            static_cast<int>(q->ne[1] / k->ne[1]), static_cast<int>(k->ne[1]), argus_cuda_external_scale(dst)};
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
            static_cast<const int64_t *>(indices->data), width, static_cast<__half *>(host->data),
            static_cast<__half *>(mirror->data), static_cast<long long>(mirror->ne[1] - 1));
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
        reduce_partials<<<static_cast<unsigned>(rows_total), 256, 0, stream>>>(
            static_cast<const float *>(partials->data), static_cast<float *>(dst->data),
            static_cast<int>(partials->ne[2]), rows_total, dv, group, n_head, rows_per_head);
        check(cudaGetLastError());
        return;
    }
    const Problem p = problem_of(dst);
    auto * partials = static_cast<float *>(dst->data);
    switch (dst->src[1]->ne[0]) {
        case 64: launch_partials<64>(p, partials, stream); break;
        case 128: launch_partials<128>(p, partials, stream); break;
        case 256: launch_partials<256>(p, partials, stream); break;
        case 512: launch_partials<512>(p, partials, stream); break;
        default: throw std::runtime_error("ARGUS split attention: unsupported head dimension");
    }
    check(cudaGetLastError());
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
    if (mirror->type != GGML_TYPE_F16 || host->type != GGML_TYPE_F16 || rows->type != GGML_TYPE_F32 ||
        indices->type != GGML_TYPE_I64 || rows->ne[0] != mirror->ne[0] || host->ne[0] != mirror->ne[0] ||
        rows->nb[0] != sizeof(float) || !ggml_is_contiguous(indices) || indices->ne[0] != rows->ne[1] ||
        rows->ne[2] != 1 || rows->ne[3] != 1 || host->nb[1] != host->ne[0] * sizeof(ggml_fp16_t)) {
        return nullptr;
    }
    ggml_tensor * args[] = {rows, indices};
    return argus_cuda_external_inplace(ctx, mirror, args, 2, argus_kind_tier_append, 0.0f,
                                       static_cast<int64_t>(reinterpret_cast<intptr_t>(host)));
}
