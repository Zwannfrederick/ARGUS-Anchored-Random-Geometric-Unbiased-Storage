// K1: exact decode attention that reads each KV head once for all of its query heads.
// A task is a chunk of cells across every KV head, so each thread streams contiguous KV.
// Per block of cells and KV head, the scores and value sums of every query row sharing
// that head are computed with register blocking. F16, Q8_0 and Q4_0 rows are decoded
// inside the FMA loops (each element once per call); other types are dequantized once
// per block. Blocks merge by online softmax, chunks by log-sum-exp in a fixed order.
#include "ggml_decode_attention.h"
#include "ggml_decode_simd.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
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
    for (int64_t c = 0; c < n; ++c) {
        const T * key = keys + c * stride;
        prefetch_row(key + block * stride, d);
        Simd::V acc[R];
        for (int r = 0; r < R; ++r) { acc[r] = Simd::zero(); }
        for (int64_t i = 0; i < d; i += Simd::W) {
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

// out[r] += sum_c weights[r * block + c] * values[c * stride] for R rows and n cells; d % W == 0.
template <int R, typename T>
void mad_rows(const float * weights, const T * values, size_t stride, int64_t n, int64_t d, float * out) {
    for (int64_t c = 0; c < n; ++c) { prefetch_row(values + (c + block) * stride, d); }
    for (int64_t i = 0; i < d; i += Simd::W) {
        Simd::V acc[R];
        for (int r = 0; r < R; ++r) { acc[r] = Simd::load(out + r * d, i); }
        for (int64_t c = 0; c < n; ++c) {
            const Simd::V vv = Simd::load(values + c * stride, i);
            for (int r = 0; r < R; ++r) { acc[r] = Simd::fma(Simd::set1(weights[r * block + c]), vv, acc[r]); }
        }
        for (int r = 0; r < R; ++r) { Simd::store(out + r * d + i, acc[r]); }
    }
}

template <typename T>
using KernelFn = void (*)(const float *, const T *, size_t, int64_t, int64_t, float *);
template <typename T>
constexpr KernelFn<T> dot_table[max_rows] = {dot_rows<1, T>, dot_rows<2, T>, dot_rows<3, T>, dot_rows<4, T>,
                                             dot_rows<5, T>, dot_rows<6, T>, dot_rows<7, T>, dot_rows<8, T>};
template <typename T>
constexpr KernelFn<T> mad_table[max_rows] = {mad_rows<1, T>, mad_rows<2, T>, mad_rows<3, T>, mad_rows<4, T>,
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
void mad_block(int rows, const float * weights, const T * values, size_t stride, int64_t n, int64_t d, float * out) {
#ifdef ARGUS_K1_SIMD
    if (d % Simd::W == 0) { return mad_table<T>[rows - 1](weights, values, stride, n, d, out); }
#endif
    for (int r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < n; ++c) {
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

struct Scratch {
    std::vector<float> q, out, max, sum, bias, keys, values, scores;
};

// One KV head over n cells of a block: update the head's rows by online softmax.
template <typename TK, typename TV>
void attend_head(const ArgusK1Problem & p, Scratch & s, int64_t head_kv, const TK * keys, size_t key_stride,
                 const TV * values, size_t value_stride, int64_t n) {
    const int64_t G = p.group(), R = p.rows(), base = head_kv * R;
    for (int64_t r0 = 0; r0 < R; r0 += max_rows) {
        const int rows = static_cast<int>(std::min<int64_t>(max_rows, R - r0));
        float * scores = s.scores.data();
        dot_block(rows, s.q.data() + (base + r0) * p.dk, keys, key_stride, n, p.dk, scores);
        for (int r = 0; r < rows; ++r) {
            const int64_t row = base + r0 + r;
            float * row_scores = scores + r * block;
            const float best = bias_block(row_scores, s.bias.data() + (r0 + r) / G * block, n, p.scale);
            if (best == minus_inf) {
                std::fill_n(row_scores, n, 0.0f);
                continue;
            }
            const float merged = std::max(s.max[row], best);
            const float keep = std::exp(s.max[row] - merged);
            if (keep != 1.0f) {
                float * out = s.out.data() + row * p.dv;
                for (int64_t i = 0; i < p.dv; ++i) { out[i] *= keep; }
            }
            s.sum[row] = s.sum[row] * keep + exp_block(row_scores, n, merged);
            s.max[row] = merged;
        }
        mad_block(rows, scores, values, value_stride, n, p.dv, s.out.data() + (base + r0) * p.dv);
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

void attend_chunk(const ArgusK1Problem & p, int64_t c0, int64_t c1, float * partials) {
    thread_local Scratch s;
    const int64_t G = p.group(), rows = p.n_head_kv * p.rows();
    s.q.resize(rows * p.dk);
    s.out.assign(rows * p.dv, 0.0f);
    s.max.assign(rows, minus_inf);
    s.sum.assign(rows, 0.0f);
    s.bias.resize(p.n_tokens * block);
    s.scores.resize(max_rows * block);
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t head_kv = row / p.rows(), r = row % p.rows();
        const int64_t token = r / G, head = head_kv * G + r % G;
        std::memcpy(s.q.data() + row * p.dk, p.q + head * p.q_head_stride + token * p.q_token_stride, p.dk * sizeof(float));
    }
    // Rows K1 decodes in registers are read in place; any other type is dequantized per block.
    const bool in_place = with_row_type(p.k_type, [](auto) {}) && with_row_type(p.v_type, [](auto) {});
    const auto for_block = [&](int64_t first, int64_t n) {
        for (int64_t head_kv = 0; head_kv < p.n_head_kv; ++head_kv) {
            const char * k = p.k + head_kv * p.k_head_stride + first * p.k_cell_stride;
            const char * v = p.v + head_kv * p.v_head_stride + first * p.v_cell_stride;
            if (in_place) {
                with_row_type(p.k_type, [&](auto key_type) {
                    with_row_type(p.v_type, [&](auto value_type) {
                        using TK = std::remove_const_t<std::remove_pointer_t<decltype(key_type)>>;
                        using TV = std::remove_const_t<std::remove_pointer_t<decltype(value_type)>>;
                        attend_head(p, s, head_kv, reinterpret_cast<const TK *>(k), p.k_cell_stride / sizeof(TK),
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
            attend_head(p, s, head_kv, s.keys.data(), p.dk, s.values.data(), p.dv, n);
        }
    };
    for (int64_t b0 = c0; b0 < c1; b0 += block) {
        const auto [first, last] = block_bias(p, b0, std::min(b0 + block, c1), s.bias.data());
        if (first < last) { for_block(first, last - first); }
    }
    const int64_t stride = argus_k1_partial_floats(p);
    for (int64_t row = 0; row < rows; ++row) {
        float * partial = partials + row * stride;
        partial[0] = s.max[row];
        partial[1] = s.sum[row];
        std::memcpy(partial + 2, s.out.data() + row * p.dv, p.dv * sizeof(float));
    }
}

bool k1_enabled() {
    const char * text = std::getenv("ARGUS_KV_CPU_ATTENTION");
    if (!text || !*text || std::string(text) == "stock") { return false; }
    if (std::string(text) == "k1") { return true; }
    throw std::runtime_error("ARGUS_KV_CPU_ATTENTION must be stock or k1");
}

int64_t max_tokens() {
    const char * text = std::getenv("ARGUS_K1_MAX_TOKENS");
    if (!text) { return 16; }
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end || value <= 0) { throw std::runtime_error("ARGUS_K1_MAX_TOKENS must be a positive integer"); }
    return value;
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
    p.scale = call.scale;
    return p;
}

ArgusK1Problem problem_of(const ggml_tensor * dst, void * userdata) {
    return problem_of(*static_cast<const Call *>(userdata), dst->src[0], dst->src[1], dst->src[2], dst->src[3]);
}

void partials_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
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
}

// src[4] holds K1's partials; src[5], when present, partials of the cells before them
// (the GPU side of a split call), merged first.
void reduce_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ArgusK1Problem p = problem_of(dst, userdata);
    ArgusK1Partials parts[2];
    int count = 0;
    for (const ggml_tensor * source : {dst->src[5], dst->src[4]}) {
        if (source) { parts[count++] = {static_cast<const float *>(source->data), source->ne[2]}; }
    }
    const int64_t rows = p.n_head_kv * p.rows();
    argus_k1_reduce(p, parts, count, rows * ith / nth, rows * (ith + 1) / nth,
                    static_cast<char *>(dst->data), dst->nb[1], dst->nb[2]);
}

bool host_resident(const ggml_tensor * tensor) {
    const ggml_tensor * storage = tensor->view_src ? tensor->view_src : tensor;
    return storage->buffer && ggml_backend_buffer_is_host(storage->buffer);
}
} // namespace

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
                for (int64_t i = 0; i < p.dv; ++i) { result[i] = result[i] * keep + partial[2 + i] * add; }
                sum = sum * keep + partial[1] * add;
                max = merged;
            }
        }
        const float inverse = sum == 0.0f ? 0.0f : 1.0f / sum;
        for (int64_t i = 0; i < p.dv; ++i) { result[i] *= inverse; }
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
