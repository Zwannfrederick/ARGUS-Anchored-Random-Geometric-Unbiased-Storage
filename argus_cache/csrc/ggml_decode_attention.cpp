// K1: exact decode attention that reads each KV head once for all of its query heads.
// A task is a chunk of cells across every KV head, so each thread streams contiguous KV.
// Per block of cells and KV head, the scores and value sums of every query row sharing
// that head are computed with register blocking. F16, Q8_0 and Q4_0 rows are decoded
// inside the FMA loops (each element once per call); other types are dequantized once
// per block. Blocks merge by online softmax, chunks by log-sum-exp in a fixed order.
#include "ggml_decode_attention.h"
#include "ggml_decode_e1.h"
#include "ggml_decode_simd.h"
#include "ggml_layer_prefill.h"
#include "argus_config.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace argus_k1;

constexpr int64_t block = 32;   // cells per block
constexpr int max_rows = 8;     // query rows per register block
constexpr float minus_inf = -std::numeric_limits<float>::infinity();

struct AtomicStats {
    std::atomic<uint64_t> total_blocks{0};
    std::atomic<uint64_t> skipped_k_blocks{0};
    std::atomic<uint64_t> skipped_v_blocks{0};
    std::atomic<uint64_t> total_v_cells{0};
    std::atomic<uint64_t> skipped_v_cells{0};
    std::atomic<uint64_t> kernel_ns{0};
};

AtomicStats stat_global;
AtomicStats stat_layers[ARGUS_MAX_TRACKED_LAYERS];
ArgusK1Stats (*ext_get_layer_fn)(int il) = nullptr;
void (*ext_reset_fn)() = nullptr;

std::mutex sheet_lock;
std::deque<KeyBoxSheet> box_sheets;

// The L2 streamer stops at 4 KiB pages, a handful of cells here, so every row of the block
// after this one is requested while this one is computed.
template <typename T>
void prefetch_row(const T * row, int64_t d) {
    const char * bytes = reinterpret_cast<const char *>(row);
    for (size_t offset = 0; offset < row_bytes<T>(d); offset += 64) { __builtin_prefetch(bytes + offset); }
}

#ifdef ARGUS_K1_SIMD
// scores[r * block + c] = dot(q[r], keys[c * stride]) for R rows and n cells; d % W == 0.
template <int R, typename T>
void dot_rows(const float * q, const T * keys, size_t stride, int64_t n, int64_t d, float * scores) {
    constexpr int K32 = 32 / Simd::W;
    for (int64_t c = 0; c < n; ++c) {
        const T * key = keys + c * stride;
        prefetch_row(key + block * stride, d);
        Simd::V acc[R];
        for (int r = 0; r < R; ++r) { acc[r] = Simd::zero(); }
        int64_t i = 0;
        for (; i + 32 <= d; i += 32) {
            Simd::V kv[K32];
            Simd::load32(key, i, kv);
            for (int k = 0; k < K32; ++k) {
                const int64_t off = i + k * Simd::W;
                for (int r = 0; r < R; ++r) { acc[r] = Simd::fma(Simd::load(q + r * d, off), kv[k], acc[r]); }
            }
        }
        for (; i < d; i += Simd::W) {
            const Simd::V kv = Simd::load(key, i);
            for (int r = 0; r < R; ++r) { acc[r] = Simd::fma(Simd::load(q + r * d, i), kv, acc[r]); }
        }
        __m256 halves[max_rows];
        for (int r = 0; r < max_rows; ++r) { halves[r] = r < R ? Simd::half(acc[r]) : _mm256_setzero_ps(); }
        alignas(32) float out[max_rows];
        _mm256_store_ps(out, sums8(halves));
        for (int r = 0; r < R; ++r) { scores[r * block + c] = out[r]; }
    }
}

// out[r] += sum_c weights[r * block + c] * values[c * stride] for R rows and active cells; d % W == 0.
template <int R, typename T>
void mad_rows(const float * weights, const T * values, size_t stride, const int * active_c, int n_active, int64_t d, float * out) {
    constexpr int K32 = 32 / Simd::W;
    for (int idx = 0; idx < n_active; ++idx) { prefetch_row(values + (active_c[idx] + block) * stride, d); }
    int64_t i = 0;
    for (; i + 32 <= d; i += 32) {
        Simd::V acc[R][K32];
        for (int r = 0; r < R; ++r) {
            for (int k = 0; k < K32; ++k) { acc[r][k] = Simd::load(out + r * d, i + k * Simd::W); }
        }
        for (int idx = 0; idx < n_active; ++idx) {
            const int c = active_c[idx];
            Simd::V vv[K32];
            Simd::load32(values + c * stride, i, vv);
            for (int r = 0; r < R; ++r) {
                const Simd::V w = Simd::set1(weights[r * block + c]);
                for (int k = 0; k < K32; ++k) { acc[r][k] = Simd::fma(w, vv[k], acc[r][k]); }
            }
        }
        for (int r = 0; r < R; ++r) {
            for (int k = 0; k < K32; ++k) { Simd::store(out + r * d + i + k * Simd::W, acc[r][k]); }
        }
    }
    for (; i < d; i += Simd::W) {
        Simd::V acc[R];
        for (int r = 0; r < R; ++r) { acc[r] = Simd::load(out + r * d, i); }
        for (int idx = 0; idx < n_active; ++idx) {
            const int c = active_c[idx];
            const Simd::V vv = Simd::load(values + c * stride, i);
            for (int r = 0; r < R; ++r) { acc[r] = Simd::fma(Simd::set1(weights[r * block + c]), vv, acc[r]); }
        }
        for (int r = 0; r < R; ++r) { Simd::store(out + r * d + i, acc[r]); }
    }
}

template <typename T>
using DotFn = void (*)(const float *, const T *, size_t, int64_t, int64_t, float *);
template <typename T>
using MadFn = void (*)(const float *, const T *, size_t, const int *, int, int64_t, float *);
template <typename T>
constexpr DotFn<T> dot_table[max_rows] = {dot_rows<1, T>, dot_rows<2, T>, dot_rows<3, T>, dot_rows<4, T>,
                                          dot_rows<5, T>, dot_rows<6, T>, dot_rows<7, T>, dot_rows<8, T>};
template <typename T>
constexpr MadFn<T> mad_table[max_rows] = {mad_rows<1, T>, mad_rows<2, T>, mad_rows<3, T>, mad_rows<4, T>,
                                          mad_rows<5, T>, mad_rows<6, T>, mad_rows<7, T>, mad_rows<8, T>};
#endif

template <typename T>
void dot_block(int rows, const float * q, const T * keys, size_t stride, int64_t n, int64_t d, float * scores) {
#ifdef ARGUS_K1_SIMD
    if (d % Simd::W == 0) { return dot_table<T>[rows - 1](q, keys, stride, n, d, scores); }
#endif
    for (int r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < n; ++c) {
            float sum = 0.0f;
            for (int64_t i = 0; i < d; ++i) { sum += q[r * d + i] * element(keys + c * stride, i); }
            scores[r * block + c] = sum;
        }
    }
}

template <typename T>
void mad_block(int rows, const float * weights, const T * values, size_t stride, const int * active_c, int n_active,
               int64_t d, float * out) {
#ifdef ARGUS_K1_SIMD
    if (d % Simd::W == 0) { return mad_table<T>[rows - 1](weights, values, stride, active_c, n_active, d, out); }
#endif
    for (int r = 0; r < rows; ++r) {
        for (int idx = 0; idx < n_active; ++idx) {
            const int c = active_c[idx];
            for (int64_t i = 0; i < d; ++i) { out[r * d + i] += weights[r * block + c] * element(values + c * stride, i); }
        }
    }
}

// scores[c] = scores[c] * scale + bias[c] over n cells; returns the max. A bias of -inf hides
// a cell: the product is finite because K is (KV buffers are zeroed before any write).
float bias_block(float * scores, const float * bias, int64_t n, float scale) {
    int64_t c = 0;
    float best = minus_inf;
#ifdef ARGUS_K1_SIMD
    Simd::V top = Simd::set1(minus_inf);
    for (; c + Simd::W <= n; c += Simd::W) {
        const Simd::V x = Simd::fma(Simd::load(scores, c), Simd::set1(scale), Simd::load(bias, c));
        Simd::store(scores + c, x);
        top = Simd::max(top, x);
    }
    best = Simd::max(top);
#endif
    for (; c < n; ++c) {
        scores[c] = scores[c] * scale + bias[c];
        best = std::max(best, scores[c]);
    }
    return best;
}

// weights[c] = exp(scores[c] - max) over n cells; returns their sum.
float exp_block(float * scores, int64_t n, float max) {
    int64_t c = 0;
    float sum = 0.0f;
#ifdef ARGUS_K1_SIMD
    Simd::V total = Simd::zero();
    for (; c + Simd::W <= n; c += Simd::W) {
        const Simd::V e = Simd::exp(Simd::load(scores, c) - Simd::set1(max));
        Simd::store(scores + c, e);
        total = total + e;
    }
    sum = Simd::sum(total);
#endif
    for (; c < n; ++c) {
        scores[c] = std::exp(scores[c] - max);
        sum += scores[c];
    }
    return sum;
}

void scale_row(float * out, float keep, int64_t d) {
    int64_t i = 0;
#ifdef ARGUS_K1_SIMD
    const Simd::V vk = Simd::set1(keep);
    for (; i + Simd::W <= d; i += Simd::W) {
        Simd::store(out + i, Simd::mul(Simd::load(out, i), vk));
    }
#endif
    for (; i < d; ++i) { out[i] *= keep; }
}

void fma_row(float * dst, float keep, const float * src, float add, int64_t d) {
    int64_t i = 0;
#ifdef ARGUS_K1_SIMD
    const Simd::V vk = Simd::set1(keep), va = Simd::set1(add);
    for (; i + Simd::W <= d; i += Simd::W) {
        Simd::store(dst + i, Simd::fma(Simd::load(src, i), va, Simd::mul(Simd::load(dst, i), vk)));
    }
#endif
    for (; i < d; ++i) { dst[i] = dst[i] * keep + src[i] * add; }
}

struct Scratch {
    std::vector<float> q, out, max, sum, wm_max, wm_sum, init_wm_max, init_wm_sum, bias, keys, values, scores;
    std::vector<QueryTopR> q_top_r;
    const char * wm_k_ptr = nullptr;
    int64_t wm_n_kv = -1;
    uint64_t wm_q_tag = 0;
    float wm_skip_eps = -1.0f;
    int64_t last_valid_cell = 0;
    int64_t last_valid_blk = 0;
    uint64_t total_blocks = 0;
    uint64_t skipped_k_blocks = 0;
    uint64_t skipped_v_blocks = 0;
    uint64_t total_v_cells = 0;
    uint64_t skipped_v_cells = 0;
};

template <typename TK, typename TV>
bool check_or_update_key_box(const ArgusK1Problem & p, KeyBoxSheet * sheet, Scratch & s, int64_t head_kv,
                             int64_t first, int64_t base, int rows, int64_t G,
                             const TK * keys, size_t key_stride, const TV * values, size_t value_stride, int64_t n) {
    if (p.skip_eps <= 0.0f || n < 2 || !sheet) { return false; }
    const int64_t blk_idx = first / block;
    const int64_t global_blk_idx = (first + p.mask_cell0) / block;
    const int64_t last_blk_idx = s.last_valid_blk;
    if (global_blk_idx <= 0 || blk_idx >= last_blk_idx - 1 || blk_idx >= sheet->max_blocks) { return false; }
    const size_t slot = static_cast<size_t>(head_kv * sheet->max_blocks + blk_idx);
    const uint64_t tag = fast_block_tag(keys, key_stride * sizeof(TK), n);
    float * box_base = sheet->boxes.data() + slot * static_cast<size_t>(2 * p.dk);
    float * mu_v     = sheet->v_centroids.data() + slot * static_cast<size_t>(p.dv);
    const float * mu  = box_base;
    const float * rad = box_base + p.dk;
    if (sheet->tags[slot].load(std::memory_order_acquire) == tag) {
        bool all_negligible = true;
        const float n_tail = static_cast<float>(std::min<int64_t>(4, n));
        const float n_f    = static_cast<float>(n);
        const float block_tail_norm = sheet->tail_norms[slot];
        float row_mean_scores[max_rows];
        float row_mono_mass[max_rows];
        for (int r = 0; r < rows; ++r) {
            row_mean_scores[r] = minus_inf;
            row_mono_mass[r] = 0.0f;
            const int64_t row = base + r;
            if (s.wm_sum[row] <= 0.0f) { all_negligible = false; break; }
            const float * bias_r = s.bias.data() + (r / G) * block;
            float max_bias = minus_inf;
            for (int64_t c = 0; c < n; ++c) { max_bias = std::max(max_bias, bias_r[c]); }
            if (max_bias == minus_inf) { continue; }
            const float fast_ub_cutoff = (s.wm_max[row] + std::log(p.skip_eps * s.wm_sum[row] / n_f) - max_bias) / p.scale;
            const auto [c_dot, ub_dot] = box_dot_bounds(s.q.data() + row * p.dk, s.q_top_r[row], mu, rad,
                                                        block_tail_norm, p.dk, fast_ub_cutoff, p.top_r);
            const float mean_score = c_dot * p.scale + max_bias;
            const float ub_score   = ub_dot * p.scale + max_bias;
            const float exp_mean   = std::exp(mean_score - s.wm_max[row]);
            const float exp_ub     = std::exp(ub_score - s.wm_max[row]);
            const float dipole_mass = n_tail * std::max(0.0f, exp_ub - exp_mean);
            const float mono_mass   = n_f * exp_mean;
            if (ub_score >= s.wm_max[row] || dipole_mass > p.skip_eps * s.wm_sum[row] ||
                mono_mass > 16.0f * p.skip_eps * s.wm_sum[row]) {
                all_negligible = false;
                break;
            }
            row_mean_scores[r] = mean_score;
            row_mono_mass[r]   = mono_mass;
        }
        if (all_negligible) {
            if (p.multipole) {
                for (int r = 0; r < rows; ++r) {
                    const int64_t row = base + r;
                    if (row_mean_scores[r] == minus_inf || row_mono_mass[r] <= 1e-6f * p.skip_eps * s.wm_sum[row]) {
                        continue;
                    }
                    const float merged = std::max(s.max[row], row_mean_scores[r]);
                    const float keep   = s.max[row] == minus_inf ? 0.0f : std::exp(s.max[row] - merged);
                    const float w_mp   = n_f * std::exp(row_mean_scores[r] - merged);
                    fma_row(s.out.data() + row * p.dv, keep, mu_v, w_mp, p.dv);
                    s.sum[row] = s.sum[row] * keep + w_mp;
                    s.max[row] = merged;
                }
            }
            ++s.skipped_k_blocks;
            ++s.skipped_v_blocks;
            s.skipped_v_cells += static_cast<uint64_t>(n);
            return true;
        }
        return false;
    }
    sheet->tail_norms[slot] = compute_key_box(keys, key_stride, n, p.dk, box_base, box_base + p.dk);
    if (p.multipole) {
        compute_value_centroid(values, value_stride, n, p.dv, mu_v);
    }
    sheet->tags[slot].store(tag, std::memory_order_release);
    return false;
}

// One KV head over n cells of a block: update the head's rows by online softmax.
template <typename TK, typename TV>
void attend_head(const ArgusK1Problem & p, KeyBoxSheet * sheet, Scratch & s, int64_t head_kv, int64_t first,
                 const TK * keys, size_t key_stride, const TV * values, size_t value_stride, int64_t n) {
    const int64_t G = p.group(), R = p.rows(), base = head_kv * R;
    const int64_t blk_idx = first / block;
    for (int64_t r0 = 0; r0 < R; r0 += max_rows) {
        const int rows = static_cast<int>(std::min<int64_t>(max_rows, R - r0));
        ++s.total_blocks;
        s.total_v_cells += static_cast<uint64_t>(n);
        if (p.skip_eps > 0.0f) {
            if (check_or_update_key_box(p, sheet, s, head_kv, first, base + r0, rows, G, keys, key_stride, values, value_stride, n)) {
                continue;
            }
        }
        const float * mu_v = (sheet && p.multipole) ? sheet->v_centroid(head_kv, blk_idx) : nullptr;
        float * scores = s.scores.data();
        dot_block(rows, s.q.data() + (base + r0) * p.dk, keys, key_stride, n, p.dk, scores);
        float row_best[max_rows];
        bool skip_v = p.skip_eps > 0.0f;
        for (int r = 0; r < rows; ++r) {
            const int64_t row = base + r0 + r;
            float * row_scores = scores + r * block;
            const float best = bias_block(row_scores, s.bias.data() + (r0 + r) / G * block, n, p.scale);
            row_best[r] = best;
            if (best == minus_inf) { continue; }
            if (skip_v) {
                if (s.wm_sum[row] <= 0.0f || best >= s.wm_max[row] ||
                    static_cast<float>(n) * std::exp(best - s.wm_max[row]) > p.skip_eps * s.wm_sum[row]) {
                    skip_v = false;
                }
            }
        }
        if (skip_v) {
            if (mu_v) {
                for (int r = 0; r < rows; ++r) {
                    const int64_t row = base + r0 + r;
                    const float best = row_best[r];
                    if (best == minus_inf) { continue; }
                    const float merged = std::max(s.max[row], best);
                    const float keep = s.max[row] == minus_inf ? 0.0f : std::exp(s.max[row] - merged);
                    const float block_sum = exp_block(scores + r * block, n, merged);
                    fma_row(s.out.data() + row * p.dv, keep, mu_v, block_sum, p.dv);
                    s.sum[row] = s.sum[row] * keep + block_sum;
                    s.max[row] = merged;
                }
            }
            ++s.skipped_v_blocks;
            s.skipped_v_cells += static_cast<uint64_t>(n);
            continue;
        }
        float cell_floor[max_rows];
        float row_block_sum[max_rows];
        for (int r = 0; r < rows; ++r) {
            const int64_t row = base + r0 + r;
            float * row_scores = scores + r * block;
            const float best = row_best[r];
            if (best == minus_inf) {
                std::fill_n(row_scores, n, 0.0f);
                cell_floor[r] = 0.0f;
                row_block_sum[r] = 0.0f;
                continue;
            }
            const float merged = std::max(s.max[row], best);
            const float keep = std::exp(s.max[row] - merged);
            if (keep != 1.0f) {
                scale_row(s.out.data() + row * p.dv, keep, p.dv);
            }
            const float block_sum = exp_block(row_scores, n, merged);
            row_block_sum[r] = block_sum;
            s.sum[row] = s.sum[row] * keep + block_sum;
            s.max[row] = merged;
            const int64_t global_blk_idx = (first + p.mask_cell0) / block;
            if (global_blk_idx > 0 && blk_idx < s.last_valid_blk - 1) {
                const float wm_merged = std::max(s.wm_max[row], merged);
                s.wm_sum[row] = s.wm_sum[row] * std::exp(s.wm_max[row] - wm_merged) + block_sum * std::exp(merged - wm_merged);
                s.wm_max[row] = wm_merged;
            }
            const float denom = std::max(s.sum[row], s.wm_sum[row] * std::exp(s.wm_max[row] - s.max[row]));
            cell_floor[r] = (p.skip_eps / static_cast<float>(block)) * denom;
        }
        int active_c[block];
        int n_active = 0;
        if (p.skip_eps <= 0.0f) {
            for (int c = 0; c < static_cast<int>(n); ++c) { active_c[n_active++] = c; }
        } else {
            for (int c = 0; c < static_cast<int>(n); ++c) {
                bool hot = false;
                for (int r = 0; r < rows; ++r) {
                    if (scores[r * block + c] > cell_floor[r]) { hot = true; break; }
                }
                if (hot) { active_c[n_active++] = c; }
            }
            if (n_active == 0) {
                if (mu_v) {
                    for (int r = 0; r < rows; ++r) {
                        if (row_block_sum[r] > 0.0f) {
                            add_scaled_row(s.out.data() + (base + r0 + r) * p.dv, mu_v, row_block_sum[r], p.dv);
                        }
                    }
                }
                ++s.skipped_v_blocks;
                s.skipped_v_cells += static_cast<uint64_t>(n);
                continue;
            }
            if (mu_v && n_active < static_cast<int>(n)) {
                for (int r = 0; r < rows; ++r) {
                    float active_sum = 0.0f;
                    const float * row_scores = scores + r * block;
                    for (int idx = 0; idx < n_active; ++idx) { active_sum += row_scores[active_c[idx]]; }
                    const float w_skipped = std::max(0.0f, row_block_sum[r] - active_sum);
                    if (w_skipped > 0.0f) {
                        add_scaled_row(s.out.data() + (base + r0 + r) * p.dv, mu_v, w_skipped, p.dv);
                    }
                }
            }
            s.skipped_v_cells += static_cast<uint64_t>(n - n_active);
        }
        mad_block(rows, scores, values, value_stride, active_c, n_active, p.dv, s.out.data() + (base + r0) * p.dv);
    }
}

// Calls f with a null pointer of the row type K1 decodes in registers, or returns false.
template <typename F>
bool with_row_type(ggml_type type, F && f) {
    switch (type) {
        case GGML_TYPE_F32:  f(static_cast<const float *>(nullptr)); return true;
        case GGML_TYPE_F16:  f(static_cast<const ggml_fp16_t *>(nullptr)); return true;
        case GGML_TYPE_Q8_0: f(static_cast<const BlockQ8_0 *>(nullptr)); return true;
        case GGML_TYPE_Q4_0: f(static_cast<const BlockQ4_0 *>(nullptr)); return true;
        default: return false;
    }
}

// Bias of every token for cells [b0, b1), shifted to start at the first cell some token sees.
// Returns that visible range; empty when no token sees any cell of the block.
std::pair<int64_t, int64_t> block_bias(const ArgusK1Problem & p, int64_t b0, int64_t b1, float * bias_rows) {
    int64_t first = b1, last = b0;
    for (int64_t token = 0; token < p.n_tokens; ++token) {
        float * bias = bias_rows + token * block;
        const char * row = p.mask ? p.mask + token * p.mask_token_stride : nullptr;
        for (int64_t cell = b0; cell < b1; ++cell) {
            bias[cell - b0] = !row ? 0.0f : p.mask_type == GGML_TYPE_F16
                ? ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(row)[cell])
                : reinterpret_cast<const float *>(row)[cell];
            if (bias[cell - b0] != minus_inf) {
                first = std::min(first, cell);
                last = std::max(last, cell + 1);
            }
        }
    }
    if (first > b0 && first < last) {
        for (int64_t token = 0; token < p.n_tokens; ++token) {
            float * bias = bias_rows + token * block;
            std::memmove(bias, bias + (first - b0), (last - first) * sizeof(float));
        }
    }
    return {first, last};
}

// Seed read-only watermarks (wm_max, wm_sum) from a reference boundary block [b0, b1) without
// modifying the chunk's partial output (out, max, sum) or block counters.
void seed_watermark_block(const ArgusK1Problem & p, Scratch & s, int64_t b0, int64_t b1, bool in_place) {
    const auto [first, last] = block_bias(p, b0, b1, s.bias.data());
    if (first >= last) { return; }
    const int64_t n = last - first, G = p.group(), R = p.rows();
    for (int64_t head_kv = 0; head_kv < p.n_head_kv; ++head_kv) {
        const char * k = p.k + head_kv * p.k_head_stride + first * p.k_cell_stride;
        const int64_t base = head_kv * R;
        for (int64_t r0 = 0; r0 < R; r0 += max_rows) {
            const int rows = static_cast<int>(std::min<int64_t>(max_rows, R - r0));
            float * scores = s.scores.data();
            if (in_place) {
                with_row_type(p.k_type, [&](auto key_type) {
                    using TK = std::remove_const_t<std::remove_pointer_t<decltype(key_type)>>;
                    dot_block(rows, s.q.data() + (base + r0) * p.dk, reinterpret_cast<const TK *>(k),
                              p.k_cell_stride / sizeof(TK), n, p.dk, scores);
                });
            } else {
                s.keys.resize(block * p.dk);
                for (int64_t c = 0; c < n; ++c) {
                    const char * key = k + c * p.k_cell_stride;
                    if (p.k_type == GGML_TYPE_F32) { std::memcpy(s.keys.data() + c * p.dk, key, p.dk * sizeof(float)); }
                    else { ggml_get_type_traits(p.k_type)->to_float(key, s.keys.data() + c * p.dk, p.dk); }
                }
                dot_block(rows, s.q.data() + (base + r0) * p.dk, s.keys.data(), p.dk, n, p.dk, scores);
            }
            for (int r = 0; r < rows; ++r) {
                const int64_t row = base + r0 + r;
                float * row_scores = scores + r * block;
                const float best = bias_block(row_scores, s.bias.data() + (r0 + r) / G * block, n, p.scale);
                if (best == minus_inf) { continue; }
                const float wm_merged = std::max(s.wm_max[row], best);
                const float block_sum = exp_block(row_scores, n, wm_merged);
                s.wm_sum[row] = s.wm_sum[row] * std::exp(s.wm_max[row] - wm_merged) + block_sum;
                s.wm_max[row] = wm_merged;
            }
        }
    }
}

int64_t find_last_valid_cell(const ArgusK1Problem & p) {
    if (!p.mask || p.n_kv <= 0) { return p.n_kv - 1; }
    for (int64_t cell = p.n_kv - 1; cell >= 0; --cell) {
        for (int64_t token = 0; token < p.n_tokens; ++token) {
            const char * row = p.mask + token * p.mask_token_stride;
            const float b = p.mask_type == GGML_TYPE_F16
                ? ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(row)[cell])
                : reinterpret_cast<const float *>(row)[cell];
            if (b != minus_inf) { return cell; }
        }
    }
    return 0;
}

void attend_chunk(const ArgusK1Problem & p, int64_t c0, int64_t c1, float * partials) {
    thread_local Scratch s;
    const int64_t G = p.group(), rows = p.n_head_kv * p.rows();
    s.q.resize(rows * p.dk);
    s.out.assign(rows * p.dv, 0.0f);
    s.max.assign(rows, minus_inf);
    s.sum.assign(rows, 0.0f);
    s.bias.resize(p.n_tokens * block);
    s.scores.resize(max_rows * block);
    s.total_blocks = 0;
    s.skipped_k_blocks = 0;
    s.skipped_v_blocks = 0;
    s.total_v_cells = 0;
    s.skipped_v_cells = 0;
    uint64_t q_tag = 14695981039346656037ULL;
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t head_kv = row / p.rows(), r = row % p.rows();
        const int64_t token = r / G, head = head_kv * G + r % G;
        float * q_row = s.q.data() + row * p.dk;
        std::memcpy(q_row, p.q + head * p.q_head_stride + token * p.q_token_stride, p.dk * sizeof(float));
        uint32_t bits = 0;
        std::memcpy(&bits, q_row, sizeof(uint32_t));
        q_tag = (q_tag ^ bits) * 1099511628211ULL;
    }
    const bool in_place = with_row_type(p.k_type, [](auto) {}) && with_row_type(p.v_type, [](auto) {});
    const bool wm_cached = (s.wm_k_ptr == p.k && s.wm_n_kv == p.n_kv && s.wm_q_tag == q_tag &&
                            s.wm_skip_eps == p.skip_eps && s.init_wm_max.size() == static_cast<size_t>(rows));
    if (!wm_cached) {
        s.q_top_r.resize(rows);
        if (p.skip_eps > 0.0f) {
            for (int64_t row = 0; row < rows; ++row) {
                s.q_top_r[row] = prepare_query_top_r(s.q.data() + row * p.dk, p.dk);
            }
        }
        s.last_valid_cell = find_last_valid_cell(p);
        s.last_valid_blk = s.last_valid_cell / block;
        s.wm_max.assign(rows, minus_inf);
        s.wm_sum.assign(rows, 0.0f);
        if (p.skip_eps > 0.0f && (s.last_valid_cell >= 2 * block || p.mask_cell0 >= block)) {
            const int64_t sink0 = p.mask_cell0 >= block ? -p.mask_cell0 : 0;
            seed_watermark_block(p, s, sink0, sink0 + std::min<int64_t>(block, s.last_valid_cell + p.mask_cell0 + 1), in_place);
            const int64_t last_global_b0 = (s.last_valid_cell / block) * block;
            if (last_global_b0 >= 2 * block || (p.mask_cell0 >= block && last_global_b0 >= block)) {
                seed_watermark_block(p, s, last_global_b0 - block, last_global_b0, in_place);
            }
            if (last_global_b0 >= block || p.mask_cell0 >= block) {
                seed_watermark_block(p, s, last_global_b0, s.last_valid_cell + 1, in_place);
            }
        }
        s.init_wm_max = s.wm_max;
        s.init_wm_sum = s.wm_sum;
        s.wm_k_ptr = p.k;
        s.wm_n_kv = p.n_kv;
        s.wm_q_tag = q_tag;
        s.wm_skip_eps = p.skip_eps;
    } else {
        s.wm_max = s.init_wm_max;
        s.wm_sum = s.init_wm_sum;
    }
    const int64_t c1_valid = std::min(c1, s.last_valid_cell + 1);
    KeyBoxSheet * sheet = p.skip_eps > 0.0f ? get_box_sheet(p, sheet_lock, box_sheets) : nullptr;
    const auto for_block = [&](int64_t first, int64_t n) {
        for (int64_t head_kv = 0; head_kv < p.n_head_kv; ++head_kv) {
            const char * k = p.k + head_kv * p.k_head_stride + first * p.k_cell_stride;
            const char * v = p.v + head_kv * p.v_head_stride + first * p.v_cell_stride;
            if (in_place) {
                with_row_type(p.k_type, [&](auto key_type) {
                    with_row_type(p.v_type, [&](auto value_type) {
                        using TK = std::remove_const_t<std::remove_pointer_t<decltype(key_type)>>;
                        using TV = std::remove_const_t<std::remove_pointer_t<decltype(value_type)>>;
                        attend_head(p, sheet, s, head_kv, first, reinterpret_cast<const TK *>(k), p.k_cell_stride / sizeof(TK),
                                    reinterpret_cast<const TV *>(v), p.v_cell_stride / sizeof(TV), n);
                    });
                });
                continue;
            }
            s.keys.resize(block * p.dk);
            s.values.resize(block * p.dv);
            for (int64_t c = 0; c < n; ++c) {
                const char * key = k + c * p.k_cell_stride, * value = v + c * p.v_cell_stride;
                if (p.k_type == GGML_TYPE_F32) { std::memcpy(s.keys.data() + c * p.dk, key, p.dk * sizeof(float)); }
                else { ggml_get_type_traits(p.k_type)->to_float(key, s.keys.data() + c * p.dk, p.dk); }
                if (p.v_type == GGML_TYPE_F32) { std::memcpy(s.values.data() + c * p.dv, value, p.dv * sizeof(float)); }
                else { ggml_get_type_traits(p.v_type)->to_float(value, s.values.data() + c * p.dv, p.dv); }
            }
            attend_head(p, nullptr, s, head_kv, first, s.keys.data(), p.dk, s.values.data(), p.dv, n);
        }
    };
    for (int64_t b0 = c0; b0 < c1_valid; b0 += block) {
        const auto [first, last] = block_bias(p, b0, std::min(b0 + block, c1_valid), s.bias.data());
        if (first < last) { for_block(first, last - first); }
    }
    argus_k1_record_layer_stats(p.layer_il, s.total_blocks, s.skipped_k_blocks, s.skipped_v_blocks,
                                s.total_v_cells, s.skipped_v_cells);
    const int64_t stride = argus_k1_partial_floats(p);
    for (int64_t row = 0; row < rows; ++row) {
        float * partial = partials + row * stride;
        partial[0] = s.max[row];
        partial[1] = s.sum[row];
        std::memcpy(partial + 2, s.out.data() + row * p.dv, p.dv * sizeof(float));
    }
}

bool k1_enabled() {
    return argus_get_config().k1_enabled;
}

int64_t max_tokens() {
    return argus_get_config().k1_max_tokens;
}

// What a node needs beyond its tensors. GGML custom ops keep their own state in op_params,
// so this travels as interned userdata.
struct Call {
    float scale;
    int64_t mask_cell0;  // cell c of k/v reads mask column mask_cell0 + c
    bool operator==(const Call & other) const { return scale == other.scale && mask_cell0 == other.mask_cell0; }
};

const Call * intern(Call call) {
    static std::mutex lock;
    static std::deque<Call> calls;
    std::lock_guard<std::mutex> guard(lock);
    for (const auto & known : calls) {
        if (known == call) { return &known; }
    }
    calls.push_back(call);
    return &calls.back();
}

ArgusK1Problem problem_of(const Call & call, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                          const ggml_tensor * mask) {
    ArgusK1Problem p{};
    p.dk = k->ne[0]; p.dv = v->ne[0]; p.n_head = q->ne[1]; p.n_head_kv = k->ne[1]; p.n_tokens = q->ne[2]; p.n_kv = k->ne[2];
    p.k_type = k->type; p.v_type = v->type;
    p.q = static_cast<const char *>(q->data); p.q_head_stride = q->nb[1]; p.q_token_stride = q->nb[2];
    p.k = static_cast<const char *>(k->data); p.k_head_stride = k->nb[1]; p.k_cell_stride = k->nb[2];
    p.v = static_cast<const char *>(v->data); p.v_head_stride = v->nb[1]; p.v_cell_stride = v->nb[2];
    p.mask = static_cast<const char *>(mask->data) + call.mask_cell0 * ggml_type_size(mask->type);
    p.mask_type = mask->type; p.mask_token_stride = mask->nb[1];
    const ArgusConfig cfg = argus_get_config();
    p.scale = call.scale;
    p.skip_eps = cfg.skip_eps;
    p.multipole = cfg.multipole_enabled;
    p.top_r = cfg.top_r_enabled;
    if (call.mask_cell0 > 0 && k->view_src && k->view_offs >= static_cast<size_t>(call.mask_cell0) * k->nb[2]) {
        p.mask_cell0 = call.mask_cell0;
    }
    p.layer_il = argus_parse_layer_il(k);
    return p;
}

ArgusK1Problem problem_of(const ggml_tensor * dst, void * userdata) {
    return problem_of(*static_cast<const Call *>(userdata), dst->src[0], dst->src[1], dst->src[2], dst->src[3]);
}

void partials_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto t0 = ith == 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const ArgusK1Problem p = problem_of(dst, userdata);
    static std::once_flag announced;  // one line per process: runs can prove the path was taken
    std::call_once(announced, [&] {
        std::fprintf(stderr, "ARGUS K1: decode attention on host KV (%s/%s, head dim %lld, %lld/%lld heads)\n",
                     ggml_type_name(p.k_type), ggml_type_name(p.v_type), (long long) p.dk,
                     (long long) p.n_head, (long long) p.n_head_kv);
    });
    const int64_t tasks = argus_k1_tasks(p), floats = p.n_head_kv * p.rows() * argus_k1_partial_floats(p);
    for (int64_t task = ith; task < tasks; task += nth) {
        argus_k1_task(p, task, static_cast<float *>(dst->data) + task * floats);
    }
    if (ith == 0) {
        const uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        stat_global.kernel_ns.fetch_add(ns, std::memory_order_relaxed);
        if (p.layer_il >= 0 && p.layer_il < ARGUS_MAX_TRACKED_LAYERS) {
            stat_layers[p.layer_il].kernel_ns.fetch_add(ns, std::memory_order_relaxed);
        }
    }
}

// src[4] holds K1's partials; src[5], when present, partials of the cells before them
// (the GPU side of a split call), merged first.
void reduce_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto t0 = ith == 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const ArgusK1Problem p = problem_of(dst, userdata);
    ArgusK1Partials parts[2];
    int count = 0;
    for (const ggml_tensor * source : {dst->src[5], dst->src[4]}) {
        if (source) { parts[count++] = {static_cast<const float *>(source->data), source->ne[2]}; }
    }
    const int64_t rows = p.n_head_kv * p.rows();
    argus_k1_reduce(p, parts, count, rows * ith / nth, rows * (ith + 1) / nth,
                    static_cast<char *>(dst->data), dst->nb[1], dst->nb[2]);
    if (ith == 0) {
        const uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        stat_global.kernel_ns.fetch_add(ns, std::memory_order_relaxed);
        if (p.layer_il >= 0 && p.layer_il < ARGUS_MAX_TRACKED_LAYERS) {
            stat_layers[p.layer_il].kernel_ns.fetch_add(ns, std::memory_order_relaxed);
        }
    }
}

bool host_resident(const ggml_tensor * tensor) {
    const ggml_tensor * storage = tensor->view_src ? tensor->view_src : tensor;
    return storage->buffer && ggml_backend_buffer_is_host(storage->buffer);
}
} // namespace

int argus_parse_layer_il(const ggml_tensor * k) {
    if (!k) { return 0; }
    const ggml_tensor * src = k->view_src ? k->view_src : k;
    for (const char * name : {src->name, k->name}) {
        if (!name || !*name) { continue; }
        const char * last_l = nullptr;
        for (const char * p = name; (p = std::strstr(p, "_l")) != nullptr; p += 2) {
            if (p[2] >= '0' && p[2] <= '9') { last_l = p + 2; }
        }
        if (last_l) {
            const int il = std::atoi(last_l);
            if (il >= 0 && il < ARGUS_MAX_TRACKED_LAYERS) { return il; }
        }
    }
    return 0;
}

void argus_k1_record_layer_stats(int il, uint64_t total_blocks, uint64_t skipped_k_blocks,
                                 uint64_t skipped_v_blocks, uint64_t total_v_cells, uint64_t skipped_v_cells) {
    if (total_blocks == 0 && total_v_cells == 0) { return; }
    stat_global.total_blocks.fetch_add(total_blocks, std::memory_order_relaxed);
    stat_global.skipped_k_blocks.fetch_add(skipped_k_blocks, std::memory_order_relaxed);
    stat_global.skipped_v_blocks.fetch_add(skipped_v_blocks, std::memory_order_relaxed);
    stat_global.total_v_cells.fetch_add(total_v_cells, std::memory_order_relaxed);
    stat_global.skipped_v_cells.fetch_add(skipped_v_cells, std::memory_order_relaxed);
    if (il >= 0 && il < ARGUS_MAX_TRACKED_LAYERS) {
        stat_layers[il].total_blocks.fetch_add(total_blocks, std::memory_order_relaxed);
        stat_layers[il].skipped_k_blocks.fetch_add(skipped_k_blocks, std::memory_order_relaxed);
        stat_layers[il].skipped_v_blocks.fetch_add(skipped_v_blocks, std::memory_order_relaxed);
        stat_layers[il].total_v_cells.fetch_add(total_v_cells, std::memory_order_relaxed);
        stat_layers[il].skipped_v_cells.fetch_add(skipped_v_cells, std::memory_order_relaxed);
    }
}

int64_t argus_k1_chunk_cells(int64_t n_kv) {
    // About 64 chunks, in 256-cell steps (llama.cpp pads views to 256), at least 256 cells.
    return std::max<int64_t>(256, (n_kv / 64 + 255) / 256 * 256);
}

int64_t argus_k1_tasks(const ArgusK1Problem & p) {
    const int64_t chunk = argus_k1_chunk_cells(p.n_kv);
    return (p.n_kv + chunk - 1) / chunk;
}

void argus_k1_task(const ArgusK1Problem & p, int64_t task, float * partials) {
    const int64_t chunk = argus_k1_chunk_cells(p.n_kv), c0 = task * chunk;
    attend_chunk(p, c0, std::min(c0 + chunk, p.n_kv), partials);
}

void argus_k1_reduce(const ArgusK1Problem & p, const ArgusK1Partials * parts, int n_parts, int64_t head_row0,
                     int64_t head_row1, char * out, size_t out_head_stride, size_t out_token_stride) {
    const int64_t G = p.group(), R = p.rows(), rows = p.n_head_kv * R;
    const int64_t stride = argus_k1_partial_floats(p);
    for (int64_t head_row = head_row0; head_row < head_row1; ++head_row) {
        const int64_t head_kv = head_row / R, row = head_row % R;
        const int64_t token = row / G, head = head_kv * G + row % G;
        auto * result = reinterpret_cast<float *>(out + head * out_head_stride + token * out_token_stride);
        std::fill_n(result, p.dv, 0.0f);
        float max = minus_inf, sum = 0.0f;
        for (int part = 0; part < n_parts; ++part) {
            for (int64_t j = 0; j < parts[part].chunks; ++j) {
                const float * partial = parts[part].data + (j * rows + head_row) * stride;
                if (partial[1] == 0.0f) { continue; }
                const float merged = std::max(max, partial[0]);
                const float keep = std::exp(max - merged), add = std::exp(partial[0] - merged);
                fma_row(result, keep, partial + 2, add, p.dv);
                sum = sum * keep + partial[1] * add;
                max = merged;
            }
        }
        const float inverse = sum == 0.0f ? 0.0f : 1.0f / sum;
        scale_row(result, inverse, p.dv);
    }
}

namespace {
bool k1_supports(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask,
                 int64_t mask_cell0) {
    return q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float) && q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1 &&
        q->ne[0] == k->ne[0] && k->ne[1] == v->ne[1] && k->ne[2] == v->ne[2] && k->ne[2] > 0 && q->ne[1] % k->ne[1] == 0 &&
        v->nb[1] <= v->nb[2] && k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type) &&
        k->ne[0] % ggml_blck_size(k->type) == 0 && v->ne[0] % ggml_blck_size(v->type) == 0 &&
        (k->type == GGML_TYPE_F32 || ggml_get_type_traits(k->type)->to_float) &&
        (v->type == GGML_TYPE_F32 || ggml_get_type_traits(v->type)->to_float) &&
        (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32) && mask_cell0 >= 0 &&
        mask->ne[0] >= mask_cell0 + k->ne[2] && mask->ne[1] >= q->ne[2] && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        host_resident(k) && host_resident(v);
}

ggml_tensor * k1_graph(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
                       int64_t mask_cell0, ggml_tensor * prior, float scale) {
    auto * userdata = const_cast<Call *>(intern({scale, mask_cell0}));
    const int64_t rows = q->ne[2] * q->ne[1];
    const int64_t chunk = argus_k1_chunk_cells(k->ne[2]), tasks = (k->ne[2] + chunk - 1) / chunk;
    ggml_tensor * inputs[] = {q, k, v, mask};
    ggml_tensor * partials = ggml_custom_4d(ctx, GGML_TYPE_F32, v->ne[0] + 2, rows, tasks, 1, inputs, 4,
                                            partials_op, GGML_N_TASKS_MAX, userdata);
    ggml_tensor * reduce_inputs[] = {q, k, v, mask, partials, prior};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, v->ne[0], q->ne[1], q->ne[2], 1, reduce_inputs, prior ? 6 : 5,
                          reduce_op, GGML_N_TASKS_MAX, userdata);
}
} // namespace

bool argus_k1_enabled() { return k1_enabled(); }

bool argus_ggml_is_k1(const ggml_tensor * tensor) {
    if (tensor->op != GGML_OP_CUSTOM) { return false; }
    ggml_custom_op_t fun;  // the first field of GGML's ggml_custom_op_params
    std::memcpy(&fun, tensor->op_params, sizeof fun);
    return fun == partials_op || fun == reduce_op;
}

int64_t argus_k1_max_tokens() { return max_tokens(); }

void argus_register_external_stats(ArgusK1Stats (*get_layer_fn)(int il), void (*reset_fn)()) {
    ext_get_layer_fn = get_layer_fn;
    ext_reset_fn = reset_fn;
}

ArgusK1Stats argus_k1_stats() {
    ArgusK1Stats total{
        stat_global.total_blocks.load(std::memory_order_relaxed),
        stat_global.skipped_k_blocks.load(std::memory_order_relaxed),
        stat_global.skipped_v_blocks.load(std::memory_order_relaxed),
        stat_global.total_v_cells.load(std::memory_order_relaxed),
        stat_global.skipped_v_cells.load(std::memory_order_relaxed),
        stat_global.kernel_ns.load(std::memory_order_relaxed),
    };
    if (ext_get_layer_fn) {
        for (int il = 0; il < ARGUS_MAX_TRACKED_LAYERS; ++il) {
            const ArgusK1Stats ext = ext_get_layer_fn(il);
            total.total_blocks     += ext.total_blocks;
            total.skipped_k_blocks += ext.skipped_k_blocks;
            total.skipped_v_blocks += ext.skipped_v_blocks;
            total.total_v_cells    += ext.total_v_cells;
            total.skipped_v_cells  += ext.skipped_v_cells;
            total.kernel_ns        += ext.kernel_ns;
        }
    }
    return total;
}

ArgusK1Stats argus_k1_layer_stats(int il) {
    if (il < 0 || il >= ARGUS_MAX_TRACKED_LAYERS) { return {}; }
    ArgusK1Stats res{
        stat_layers[il].total_blocks.load(std::memory_order_relaxed),
        stat_layers[il].skipped_k_blocks.load(std::memory_order_relaxed),
        stat_layers[il].skipped_v_blocks.load(std::memory_order_relaxed),
        stat_layers[il].total_v_cells.load(std::memory_order_relaxed),
        stat_layers[il].skipped_v_cells.load(std::memory_order_relaxed),
        stat_layers[il].kernel_ns.load(std::memory_order_relaxed),
    };
    if (ext_get_layer_fn) {
        const ArgusK1Stats ext = ext_get_layer_fn(il);
        res.total_blocks     += ext.total_blocks;
        res.skipped_k_blocks += ext.skipped_k_blocks;
        res.skipped_v_blocks += ext.skipped_v_blocks;
        res.total_v_cells    += ext.total_v_cells;
        res.skipped_v_cells  += ext.skipped_v_cells;
        res.kernel_ns        += ext.kernel_ns;
    }
    return res;
}

void argus_k1_reset_stats() {
    stat_global.total_blocks.store(0, std::memory_order_relaxed);
    stat_global.skipped_k_blocks.store(0, std::memory_order_relaxed);
    stat_global.skipped_v_blocks.store(0, std::memory_order_relaxed);
    stat_global.total_v_cells.store(0, std::memory_order_relaxed);
    stat_global.skipped_v_cells.store(0, std::memory_order_relaxed);
    stat_global.kernel_ns.store(0, std::memory_order_relaxed);
    for (int i = 0; i < ARGUS_MAX_TRACKED_LAYERS; ++i) {
        stat_layers[i].total_blocks.store(0, std::memory_order_relaxed);
        stat_layers[i].skipped_k_blocks.store(0, std::memory_order_relaxed);
        stat_layers[i].skipped_v_blocks.store(0, std::memory_order_relaxed);
        stat_layers[i].total_v_cells.store(0, std::memory_order_relaxed);
        stat_layers[i].skipped_v_cells.store(0, std::memory_order_relaxed);
        stat_layers[i].kernel_ns.store(0, std::memory_order_relaxed);
    }
    if (ext_reset_fn) { ext_reset_fn(); }
}

void argus_k1_print_summary() {
    const ArgusK1Stats s = argus_k1_stats();
    const ArgusMoeStats m = argus_moe_stats ? argus_moe_stats() : ArgusMoeStats{};
    if (s.total_blocks > 0 || s.total_v_cells > 0) {
        const double k_pct = s.total_blocks ? 100.0 * double(s.skipped_k_blocks) / double(s.total_blocks) : 0.0;
        const double v_pct = s.total_v_cells ? 100.0 * double(s.skipped_v_cells) / double(s.total_v_cells) : 0.0;
        std::fprintf(stderr,
                     "[ARGUS Telemetry] K-blocks skipped: %llu / %llu (%.2f%%) | V-cells skipped: %llu / %llu (%.2f%%)\n",
                     (unsigned long long) s.skipped_k_blocks, (unsigned long long) s.total_blocks, k_pct,
                     (unsigned long long) s.skipped_v_cells, (unsigned long long) s.total_v_cells, v_pct);
    }
    if (m.total_routed > 0) {
        const double hit_pct = 100.0 * double(m.hot_hits) / double(m.total_routed);
        std::fprintf(stderr,
                     "[ARGUS MoE Cache] Hot-expert hits: %llu / %llu (%.2f%%) | Promotions: %llu | Demotions: %llu\n",
                     (unsigned long long) m.hot_hits, (unsigned long long) m.total_routed, hit_pct,
                     (unsigned long long) m.promotions, (unsigned long long) m.demotions);
    }
}

ggml_tensor * argus_ggml_decode_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                          ggml_tensor * mask, bool unsupported_features, float scale) {
    if (!k1_enabled() || unsupported_features || !mask || q->ne[2] > max_tokens() || !k1_supports(q, k, v, mask, 0)) {
        return nullptr;
    }
    return k1_graph(ctx, q, k, v, mask, 0, nullptr, scale);
}

ggml_tensor * argus_ggml_decode_attention_after(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                                ggml_tensor * mask, int64_t mask_cell0, ggml_tensor * prior, float scale) {
    if (!k1_supports(q, k, v, mask, mask_cell0) ||
        (prior && (prior->ne[0] != v->ne[0] + 2 || prior->ne[1] != q->ne[1] * q->ne[2] || prior->type != GGML_TYPE_F32))) {
        return nullptr;
    }
    return k1_graph(ctx, q, k, v, mask, mask_cell0, prior, scale);
}
