// Benchmark A: Stock GGML FlashAttention vs libargus K1 Exact vs libargus K1 + E1 Certified Skipping
// across F16, Q8_0 (2x compression), and Q4_0 (3.55x compression) at 16K, 64K, and 128K contexts.
#include "ggml_decode_attention.h"
#include "argus_config.h"

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
void fill_peaked(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, std::mt19937 & rng) {
    const int64_t d = k->ne[0], n_head = q->ne[1], n_head_kv = k->ne[1], n_kv = k->ne[2];
    std::normal_distribution<float> noise(0.0f, 0.25f);
    std::vector<float> qv(d * n_head, 1.2f);
    std::vector<float> kf(d * n_head_kv * n_kv), vf(d * n_head_kv * n_kv);
    // Realistic long-context attention distribution: strong sink [0..64) and local window [n_kv-256..n_kv),
    // plus a needle cluster in the middle; rest is background context.
    for (int64_t cell = 0; cell < n_kv; ++cell) {
        const bool hot = (cell < 64) || (cell >= n_kv - 256) || (cell >= n_kv / 2 && cell < n_kv / 2 + 32);
        const float bias = hot ? 2.0f : -1.2f;
        for (int64_t i = 0; i < d * n_head_kv; ++i) {
            kf[cell * d * n_head_kv + i] = bias + noise(rng);
            vf[cell * d * n_head_kv + i] = noise(rng) * 2.0f;
        }
    }
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    auto set_kv = [&](ggml_tensor * t, const std::vector<float> & src) {
        std::vector<char> bytes(ggml_nbytes(t));
        if (t->type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(src.data(), reinterpret_cast<ggml_fp16_t *>(bytes.data()), src.size());
        } else {
            ggml_quantize_chunk(t->type, src.data(), bytes.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    };
    set_kv(k, kf);
    set_kv(v, vf);
}

double median_ms(ggml_backend_t backend, ggml_cgraph * graph, int warmup, int calls) {
    std::vector<double> ms;
    for (int i = 0; i < warmup + calls; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, graph);
        const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i >= warmup) { ms.push_back(dt); }
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

double rel_error(const std::vector<float> & a, const std::vector<float> & b) {
    double max_diff = 0.0, max_ref = 1e-12;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(double(a[i]) - double(b[i])));
        max_ref = std::max(max_ref, std::fabs(double(b[i])));
    }
    return max_diff / max_ref;
}
} // namespace

int main() {
    const int threads = 4, calls = 7;
    setenv("ARGUS_KV_CPU_ATTENTION", "k1", 1);
    const int64_t d = 256, n_head = 16, n_head_kv = 2; // Qwen3.6-35B-A3B full-attention geometry
    std::mt19937 rng(2026);

    std::printf("| Context | KV Codec | KV Size (MiB) | Stock GGML (ms) | K1 Exact (ms) | K1+E1 Skip (ms) | Speedup vs Stock | Blocks Skipped (K / V) | Max Rel Err |\n");
    std::printf("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n");

    for (int64_t n_kv : {16384, 65536, 131072}) {
        for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
            argus_reset_config();
            ggml_backend_t backend = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(backend, threads);
            ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
            auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head, 1);
            auto * k = ggml_new_tensor_3d(ctx, type, d, n_head_kv, n_kv);
            auto * v = ggml_new_tensor_3d(ctx, type, d, n_head_kv, n_kv);
            auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            fill_peaked(q, k, v, rng);
            std::vector<ggml_fp16_t> zeros(n_kv, ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(mask, zeros.data(), 0, ggml_nbytes(mask));
            const float scale = 1.0f / std::sqrt(float(d));

            ggml_context * gctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead() * 3, nullptr, true});
            ggml_tensor * stock_out = ggml_flash_attn_ext(gctx, ggml_permute(gctx, q, 0, 2, 1, 3),
                ggml_permute(gctx, k, 0, 2, 1, 3), ggml_permute(gctx, v, 0, 2, 1, 3), mask, scale, 0.0f, 0.0f);
            ggml_prec_set_acc(stock_out, GGML_PREC_F32);
            ggml_cgraph * stock_graph = ggml_new_graph(gctx);
            ggml_build_forward_expand(stock_graph, stock_out);

            ggml_tensor * k1_out = argus_ggml_decode_attention(gctx, q, k, v, mask, false, scale);
            ggml_cgraph * k1_graph = ggml_new_graph(gctx);
            ggml_build_forward_expand(k1_graph, k1_out);

            ggml_gallocr_t stock_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            ggml_gallocr_t k1_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            ggml_gallocr_alloc_graph(stock_alloc, stock_graph);
            ggml_gallocr_alloc_graph(k1_alloc, k1_graph);

            const double t_stock = median_ms(backend, stock_graph, 1, calls);
            const double t_k1 = median_ms(backend, k1_graph, 1, calls);
            std::vector<float> exact_vec(ggml_nelements(k1_out));
            ggml_backend_tensor_get(k1_out, exact_vec.data(), 0, ggml_nbytes(k1_out));

            // Enable E1 Certified Page Skipping (eps = 1e-4)
            ArgusConfig cfg = argus_get_config();
            cfg.skip_eps = 1e-4f;
            argus_set_config(cfg);
            argus_k1_reset_stats();
            // Warmup 1 pass to populate bounding-box cache, then reset counters (keeping box cache warm)
            ggml_backend_graph_compute(backend, k1_graph);
            const ArgusK1Stats warm_stats = argus_k1_stats();
            const double t_skip = median_ms(backend, k1_graph, 0, calls);
            const ArgusK1Stats after_stats = argus_k1_stats();

            std::vector<float> skip_vec(ggml_nelements(k1_out));
            ggml_backend_tensor_get(k1_out, skip_vec.data(), 0, ggml_nbytes(k1_out));

            const uint64_t total_b = (after_stats.total_blocks - warm_stats.total_blocks) / calls;
            const uint64_t skip_k = (after_stats.skipped_k_blocks - warm_stats.skipped_k_blocks) / calls;
            const uint64_t skip_v = (after_stats.skipped_v_blocks - warm_stats.skipped_v_blocks) / calls;
            const double kv_mib = double(ggml_nbytes(k) + ggml_nbytes(v)) / (1024.0 * 1024.0);

            std::printf("| %lldK | `%s` | %.1f | %.2f | %.2f (%.1fx) | **%.2f** | **%.1fx** | %llu/%llu (%.0f%%) | %.2e |\n",
                        (long long)(n_kv / 1024), ggml_type_name(type), kv_mib,
                        t_stock, t_k1, t_stock / t_k1, t_skip, t_stock / t_skip,
                        (unsigned long long)skip_k, (unsigned long long)total_b,
                        total_b ? (100.0 * double(skip_v) / double(total_b)) : 0.0,
                        rel_error(skip_vec, exact_vec));
            std::fflush(stdout);

            argus_k1_reset_stats();
            argus_reset_config();
            ggml_gallocr_free(stock_alloc);
            ggml_gallocr_free(k1_alloc);
            ggml_free(gctx);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
        }
    }
    return 0;
}
