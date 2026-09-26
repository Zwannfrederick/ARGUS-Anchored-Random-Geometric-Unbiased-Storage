// Single-token decode attention bandwidth of stock GGML flash attention, on the CPU and on
// CUDA, for one full-attention layer of Qwen3.6-35B-A3B (16 query heads, 2 KV heads,
// head dim 256). Reported bandwidth is K+V bytes read per call / median call time.
// Usage: attn_bw <cpu|cuda> [threads]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void fill(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t row = t->ne[0], rows = ggml_nrows(t);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> values(row * rows);
    for (auto & x : values) { x = normal(rng); }
    std::vector<char> bytes(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) { std::memcpy(bytes.data(), values.data(), bytes.size()); }
    else if (t->type == GGML_TYPE_F16) { ggml_fp32_to_fp16_row(values.data(), reinterpret_cast<ggml_fp16_t *>(bytes.data()), values.size()); }
    else { ggml_quantize_chunk(t->type, values.data(), bytes.data(), 0, rows, row, nullptr); }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && std::string(argv[1]) == "cuda";
    const int threads = argc > 2 ? std::atoi(argv[2]) : 4;
    const int d = 256, heads = 16, kv_heads = 2;
    std::mt19937 rng(1234);
    std::printf("[\n");
    bool first = true;
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        for (int64_t n_kv : {16384, 65536, 131072}) {
            ggml_backend_t backend = cuda ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
            if (!cuda) { ggml_backend_cpu_set_n_threads(backend, threads); }
            ggml_init_params params{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, heads, 1);
            ggml_tensor * k = ggml_new_tensor_4d(ctx, type, d, n_kv, kv_heads, 1);
            ggml_tensor * v = ggml_new_tensor_4d(ctx, type, d, n_kv, kv_heads, 1);
            ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, 1, 1, 1);
            ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / 16.0f, 0.0f, 0.0f);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            fill(q, rng);
            fill(k, rng);
            fill(v, rng);
            std::vector<ggml_fp16_t> zeros(n_kv, ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(mask, zeros.data(), 0, ggml_nbytes(mask));
            if (!ggml_backend_supports_op(backend, out)) {
                std::printf("%s  {\"backend\": \"%s\", \"type\": \"%s\", \"n_kv\": %lld, \"supported\": false}", first ? "" : ",\n",
                            cuda ? "cuda" : "cpu", ggml_type_name(type), (long long) n_kv);
            } else {
                std::vector<double> ms;
                for (int i = 0; i < 12; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    ggml_backend_graph_compute(backend, graph); // synchronizes before returning
                    const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                    if (i >= 2) { ms.push_back(t); }
                }
                std::sort(ms.begin(), ms.end());
                const double median = ms[ms.size() / 2], kv_bytes = double(ggml_nbytes(k) + ggml_nbytes(v));
                std::printf("%s  {\"backend\": \"%s\", \"threads\": %d, \"type\": \"%s\", \"n_kv\": %lld, \"kv_bytes\": %.0f, \"median_ms\": %.3f, \"min_ms\": %.3f, \"gbps\": %.2f}",
                            first ? "" : ",\n", cuda ? "cuda" : "cpu", cuda ? 0 : threads, ggml_type_name(type), (long long) n_kv,
                            kv_bytes, median, ms.front(), kv_bytes / (median * 1e6));
            }
            first = false;
            std::fflush(stdout);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
        }
    }
    std::printf("\n]\n");
    return 0;
}
