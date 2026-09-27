// K1 vs stock GGML flash attention, single-token decode on the CPU, for the full-attention
// geometry of the v0.8 models. Both read the same K/V tensors in one graph each.
// Reported bandwidth is K+V bytes of the view / median call time.
// Usage: k1_bw <threads> [repeats]
#include "ggml_decode_attention.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
struct Geometry { const char * name; int64_t d, n_head, n_head_kv; };

void fill(ggml_tensor * t, std::mt19937 & rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> values(ggml_nelements(t));
    for (auto & x : values) { x = normal(rng); }
    std::vector<char> bytes(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) { std::memcpy(bytes.data(), values.data(), bytes.size()); }
    else if (t->type == GGML_TYPE_F16) { ggml_fp32_to_fp16_row(values.data(), reinterpret_cast<ggml_fp16_t *>(bytes.data()), values.size()); }
    else { ggml_quantize_chunk(t->type, values.data(), bytes.data(), 0, ggml_nrows(t), t->ne[0], nullptr); }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

std::vector<double> time_graph(ggml_backend_t backend, ggml_cgraph * graph, int calls) {
    std::vector<double> ms;
    for (int i = 0; i < calls + 2; ++i) {
        const auto start = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, graph);
        const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (i >= 2) { ms.push_back(t); }
    }
    std::sort(ms.begin(), ms.end());
    return ms;
}
} // namespace

int main(int argc, char ** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    const int calls = argc > 2 ? std::atoi(argv[2]) : 10;
    setenv("ARGUS_KV_CPU_ATTENTION", "k1", 1);
    const Geometry geometries[] = {
        {"qwen3.6-35b-a3b", 256, 16, 2}, {"gemma-4-26b-a4b-global", 512, 16, 2}, {"ui-mate-9b", 256, 16, 4}};
    std::mt19937 rng(1234);
    std::printf("[\n");
    bool first = true;
    for (const auto & g : geometries) {
        for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
            for (int64_t n_kv : {16384, 65536, 131072}) {
                ggml_backend_t backend = ggml_backend_cpu_init();
                ggml_backend_cpu_set_n_threads(backend, threads);
                ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
                auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, g.d, g.n_head, 1);
                auto * k = ggml_new_tensor_3d(ctx, type, g.d, g.n_head_kv, n_kv);
                auto * v = ggml_new_tensor_3d(ctx, type, g.d, g.n_head_kv, n_kv);
                auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1);
                ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                fill(q, rng);
                fill(k, rng);
                fill(v, rng);
                std::vector<ggml_fp16_t> zeros(n_kv, ggml_fp32_to_fp16(0.0f));
                ggml_backend_tensor_set(mask, zeros.data(), 0, ggml_nbytes(mask));
                const float scale = 1.0f / std::sqrt(float(g.d));

                ggml_context * graph_ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 2, nullptr, true});
                ggml_tensor * stock_out = ggml_flash_attn_ext(graph_ctx, ggml_permute(graph_ctx, q, 0, 2, 1, 3),
                    ggml_permute(graph_ctx, k, 0, 2, 1, 3), ggml_permute(graph_ctx, v, 0, 2, 1, 3), mask, scale, 0.0f, 0.0f);
                ggml_prec_set_acc(stock_out, GGML_PREC_F32);
                ggml_cgraph * stock = ggml_new_graph(graph_ctx);
                ggml_build_forward_expand(stock, stock_out);
                ggml_cgraph * k1 = ggml_new_graph(graph_ctx);
                ggml_build_forward_expand(k1, argus_ggml_decode_attention(graph_ctx, q, k, v, mask, false, scale));
                ggml_gallocr_t stock_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                ggml_gallocr_t k1_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                ggml_gallocr_alloc_graph(stock_alloc, stock);
                ggml_gallocr_alloc_graph(k1_alloc, k1);

                const double kv_bytes = double(ggml_nbytes(k) + ggml_nbytes(v));
                for (const char * kernel : {"stock", "k1"}) {
                    const auto ms = time_graph(backend, kernel[0] == 's' ? stock : k1, calls);
                    const double median = ms[ms.size() / 2];
                    std::printf("%s  {\"model\": \"%s\", \"kernel\": \"%s\", \"threads\": %d, \"type\": \"%s\", \"n_kv\": %lld, "
                                "\"kv_bytes\": %.0f, \"median_ms\": %.3f, \"min_ms\": %.3f, \"max_ms\": %.3f, \"gbps\": %.2f}",
                                first ? "" : ",\n", g.name, kernel, threads, ggml_type_name(type), (long long) n_kv, kv_bytes,
                                median, ms.front(), ms.back(), kv_bytes / (median * 1e6));
                    first = false;
                    std::fflush(stdout);
                }
                ggml_gallocr_free(stock_alloc);
                ggml_gallocr_free(k1_alloc);
                ggml_free(graph_ctx);
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
                ggml_backend_free(backend);
            }
        }
    }
    std::printf("\n]\n");
    return 0;
}
