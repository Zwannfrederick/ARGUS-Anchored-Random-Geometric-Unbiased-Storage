// Kernel check: ARGUS paged attention vs a double-precision reference on random
// GQA tensors laid out exactly like llama.cpp's KV views.
// Usage: test_ggml_paged_attention STORAGE_DIR
#include "ggml_host_buffer.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
struct Case { ggml_type type; int64_t dk, n_head, n_head_kv, kv_size, n_kv, n_tokens, used; };

double run(const Case & c, const char * directory) {
    ggml_init_params init = {256 * 1024 * 1024, nullptr, false};
    ggml_context * ctx = ggml_init(init);
    ggml_init_params storage_init = {ggml_tensor_overhead() * 4, nullptr, true};
    ggml_context * storage_ctx = ggml_init(storage_init);
    auto * k_store = ggml_new_tensor_2d(storage_ctx, c.type, c.dk * c.n_head_kv, c.kv_size);
    auto * v_store = ggml_new_tensor_2d(storage_ctx, c.type, c.dk * c.n_head_kv, c.kv_size);
    setenv("ARGUS_KV_DIR", directory, 1);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(storage_ctx, argus_ggml_host_buffer_type());
    if (!buffer) { throw std::runtime_error("ARGUS buffer allocation"); }

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> k_ref(c.kv_size * c.n_head_kv * c.dk), v_ref(k_ref.size());
    for (auto & x : k_ref) { x = normal(rng) * 3.0f; }
    for (auto & x : v_ref) { x = normal(rng) * 20.0f; }
    // Store through the real type conversion, then decode back so the reference sees stored values.
    const auto * cpu = ggml_get_type_traits_cpu(c.type);
    const auto * base = ggml_get_type_traits(c.type);
    for (int64_t cell = 0; cell < c.kv_size; ++cell) {
        for (auto [store, ref] : {std::pair{k_store, &k_ref}, std::pair{v_store, &v_ref}}) {
            char * row = static_cast<char *>(store->data) + cell * store->nb[1];
            float * values = ref->data() + cell * c.n_head_kv * c.dk;
            if (c.type == GGML_TYPE_F32) { std::copy_n(values, c.n_head_kv * c.dk, reinterpret_cast<float *>(row)); }
            else { cpu->from_float(values, row, c.n_head_kv * c.dk); base->to_float(row, values, c.n_head_kv * c.dk); }
        }
    }

    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.dk, c.n_head, c.n_tokens);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) { static_cast<float *>(q->data)[i] = normal(rng) * 3.0f; }
    // GGML's dot product sees the query in K's vec_dot type; the reference must too.
    std::vector<float> q_ref(static_cast<float *>(q->data), static_cast<float *>(q->data) + ggml_nelements(q));
    if (cpu->vec_dot_type != GGML_TYPE_F32) {
        std::vector<char> encoded(ggml_row_size(cpu->vec_dot_type, c.dk));
        for (size_t row = 0; row < q_ref.size(); row += c.dk) {
            ggml_get_type_traits_cpu(cpu->vec_dot_type)->from_float(q_ref.data() + row, encoded.data(), c.dk);
            ggml_get_type_traits(cpu->vec_dot_type)->to_float(encoded.data(), q_ref.data() + row, c.dk);
        }
    }
    auto * k = ggml_view_4d(ctx, k_store, c.dk, c.n_head_kv, c.n_kv, 1, ggml_row_size(c.type, c.dk),
                            ggml_row_size(c.type, c.dk * c.n_head_kv), ggml_row_size(c.type, c.dk * c.n_head_kv * c.kv_size), 0);
    auto * v = ggml_view_4d(ctx, v_store, c.dk, c.n_head_kv, c.n_kv, 1, ggml_row_size(c.type, c.dk),
                            ggml_row_size(c.type, c.dk * c.n_head_kv), ggml_row_size(c.type, c.dk * c.n_head_kv * c.kv_size), 0);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_tokens, 1, 1);
    const float scale = 1.0f / std::sqrt(float(c.dk));
    std::vector<float> bias(c.n_kv * c.n_tokens);
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t cell = 0; cell < c.n_kv; ++cell) {
            const bool visible = cell <= c.used - c.n_tokens + t;
            bias[t * c.n_kv + cell] = visible ? 0.0f : -INFINITY;
            static_cast<ggml_fp16_t *>(mask->data)[t * c.n_kv + cell] = ggml_fp32_to_fp16(bias[t * c.n_kv + cell]);
        }
    }

    for (const char * invalid : {"", "0", "-1", "16junk", "999999999999999999999999"}) {
        setenv("ARGUS_KV_BLOCK_CELLS", invalid, 1);
        bool refused = false;
        try { argus_ggml_paged_attention(ctx, q, k, v, mask, false, scale); }
        catch (const std::runtime_error &) { refused = true; }
        if (!refused) { throw std::runtime_error("invalid block size accepted"); }
    }
    setenv("ARGUS_KV_BLOCK_CELLS", "16", 1);
    auto * out = argus_ggml_paged_attention(ctx, q, k, v, mask, false, scale);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    double max_error = 0.0;
    // Reuse the same graph with both idle workers and changing worker counts.
    for (int threads : {1, 4, 16, 4}) {
        if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("kernel execution failed");
        }
        for (int64_t t = 0; t < c.n_tokens; ++t) {
            for (int64_t h = 0; h < c.n_head; ++h) {
                const int64_t hk = h / (c.n_head / c.n_head_kv);
                const float * query = q_ref.data() + (t * c.n_head + h) * c.dk;
                std::vector<double> logits(c.n_kv);
                double best = -INFINITY;
                for (int64_t cell = 0; cell < c.n_kv; ++cell) {
                    double dot = 0.0;
                    const float * key = k_ref.data() + (cell * c.n_head_kv + hk) * c.dk;
                    for (int64_t i = 0; i < c.dk; ++i) { dot += double(query[i]) * key[i]; }
                    logits[cell] = dot * scale + bias[t * c.n_kv + cell];
                    best = std::max(best, logits[cell]);
                }
                double total = 0.0;
                for (auto x : logits) { total += std::exp(x - best); }
                for (int64_t i = 0; i < c.dk; ++i) {
                    double expected = 0.0;
                    for (int64_t cell = 0; cell < c.n_kv; ++cell) {
                        expected += std::exp(logits[cell] - best) / total * v_ref[(cell * c.n_head_kv + hk) * c.dk + i];
                    }
                    const float got = *reinterpret_cast<const float *>(static_cast<char *>(out->data) + t * out->nb[2] + h * out->nb[1] + i * out->nb[0]);
                    if (!std::isfinite(got) || !std::isfinite(expected)) {
                        throw std::runtime_error("non-finite attention result");
                    }
                    max_error = std::max(max_error, std::abs(got - expected));
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(storage_ctx);
    ggml_free(ctx);
    return max_error;
}
} // namespace

int main(int argc, char ** argv) try {
    if (argc != 2) { throw std::runtime_error("usage: STORAGE_DIR"); }
    setenv("ARGUS_KV_MAX_BYTES", "1073741824", 1);
    setenv("ARGUS_KV_RESIDENT_BYTES", "4096", 1);
    setenv("ARGUS_KV_BLOCK_CELLS", "16", 1);
    const Case cases[] = {
        {GGML_TYPE_F32, 48, 6, 6, 128, 96, 5, 90},
        {GGML_TYPE_F16, 48, 6, 6, 128, 96, 5, 90},
        {GGML_TYPE_F16, 256, 8, 2, 128, 96, 7, 70},
        {GGML_TYPE_BF16, 256, 8, 2, 128, 96, 7, 70},
        {GGML_TYPE_Q8_0, 256, 8, 2, 128, 96, 1, 96},
        {GGML_TYPE_Q8_0, 512, 8, 2, 128, 64, 12, 40},
        {GGML_TYPE_Q4_0, 256, 8, 2, 128, 96, 1, 96},
    };
    for (const auto & c : cases) {
        const double error = run(c, argv[1]);
        std::printf("type=%s dk=%lld heads=%lld/%lld n_kv=%lld tokens=%lld max_error=%g\n", ggml_type_name(c.type),
                    (long long) c.dk, (long long) c.n_head, (long long) c.n_head_kv, (long long) c.n_kv,
                    (long long) c.n_tokens, error);
        if (error > 1e-3) { throw std::runtime_error("kernel mismatch"); }
    }
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAILED: %s\n", error.what());
    return 1;
}
