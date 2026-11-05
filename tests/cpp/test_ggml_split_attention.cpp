// S1 check: split decode attention (GPU partials over the first G cells, K1 over the rest,
// merged on the CPU) and GPU-only decode attention, run through GGML's scheduler with a
// CUDA and a CPU backend, against a double-precision reference.
// Usage: test_ggml_split_attention
#include "ggml_decode_attention.h"
#include "ggml_split_attention.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Case { int64_t d, n_head, n_head_kv, n_kv, n_tokens, used, gpu_cells; };

void require(bool condition, const std::string & what) {
    if (!condition) { throw std::runtime_error(what); }
}

float mask_value(const Case & c, int64_t token, int64_t cell) {
    return cell < c.used - c.n_tokens + 1 + token ? 0.0f : -std::numeric_limits<float>::infinity();
}

double run(const Case & c) {
    ggml_backend_t gpu = ggml_backend_cuda_init(0), cpu = ggml_backend_cpu_init();
    require(gpu && cpu, "backends");
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_context * host_ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    ggml_context * dev_ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    auto * q = ggml_new_tensor_3d(host_ctx, GGML_TYPE_F32, c.d, c.n_head, c.n_tokens);
    auto * k = ggml_new_tensor_3d(host_ctx, GGML_TYPE_F16, c.d, c.n_head_kv, c.n_kv);
    auto * v = ggml_new_tensor_3d(host_ctx, GGML_TYPE_F16, c.d, c.n_head_kv, c.n_kv);
    auto * mask = ggml_new_tensor_2d(host_ctx, GGML_TYPE_F16, c.n_kv, c.n_tokens);
    auto * k_dev = ggml_new_tensor_3d(dev_ctx, GGML_TYPE_F16, c.d, c.n_head_kv, c.gpu_cells);
    auto * v_dev = ggml_new_tensor_3d(dev_ctx, GGML_TYPE_F16, c.d, c.n_head_kv, c.gpu_cells);
    ggml_backend_buffer_t host_buf = ggml_backend_alloc_ctx_tensors(host_ctx, cpu);
    ggml_backend_buffer_t dev_buf = ggml_backend_alloc_ctx_tensors(dev_ctx, gpu);

    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> qv(ggml_nelements(q)), kf(ggml_nelements(k)), vf(ggml_nelements(v));
    for (auto & x : qv) { x = 2.0f * normal(rng); }
    for (auto & x : kf) { x = normal(rng); }
    for (auto & x : vf) { x = normal(rng); }
    std::vector<ggml_fp16_t> kh(kf.size()), vh(vf.size()), mh(c.n_kv * c.n_tokens);
    ggml_fp32_to_fp16_row(kf.data(), kh.data(), kf.size());
    ggml_fp32_to_fp16_row(vf.data(), vh.data(), vf.size());
    for (size_t i = 0; i < kf.size(); ++i) { kf[i] = ggml_fp16_to_fp32(kh[i]); vf[i] = ggml_fp16_to_fp32(vh[i]); }
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t cell = 0; cell < c.n_kv; ++cell) { mh[t * c.n_kv + cell] = ggml_fp32_to_fp16(mask_value(c, t, cell)); }
    }
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kh.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, vh.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(mask, mh.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(k_dev, kh.data(), 0, ggml_nbytes(k_dev));  // the mirror: first gpu_cells cells
    ggml_backend_tensor_set(v_dev, vh.data(), 0, ggml_nbytes(v_dev));

    const float scale = 1.0f / std::sqrt(float(c.d));
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * out;
    if (c.gpu_cells == c.n_kv) {
        out = argus_ggml_cuda_decode_attention(ctx, q, k_dev, v_dev, mask, scale);
    } else {
        ggml_tensor * prior = argus_ggml_cuda_decode_partials(ctx, q, k_dev, v_dev, mask, 0, scale);
        require(prior != nullptr, "GPU partials declined");
        const int64_t rest = c.n_kv - c.gpu_cells;
        ggml_tensor * k_host = ggml_view_3d(ctx, k, c.d, c.n_head_kv, rest, k->nb[1], k->nb[2], c.gpu_cells * k->nb[2]);
        ggml_tensor * v_host = ggml_view_3d(ctx, v, c.d, c.n_head_kv, rest, v->nb[1], v->nb[2], c.gpu_cells * v->nb[2]);
        out = argus_ggml_decode_attention_after(ctx, q, k_host, v_host, mask, c.gpu_cells, prior, scale);
    }
    require(out != nullptr, "split attention declined");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backends[] = {gpu, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
    if (argus_ggml_is_k1(out)) {
        ggml_backend_sched_set_tensor_backend(sched, out->src[4], cpu);
        ggml_backend_sched_set_tensor_backend(sched, out, cpu);
    }
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "graph compute");
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));

    const int64_t group = c.n_head / c.n_head_kv;
    double worst = 0.0, value_scale = 0.0;
    for (float x : vf) { value_scale = std::max(value_scale, double(std::fabs(x))); }
    std::vector<double> scores(c.n_kv), mixed(c.d);
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t h = 0; h < c.n_head; ++h) {
            const int64_t hk = h / group;
            double best = -INFINITY;
            for (int64_t cell = 0; cell < c.n_kv; ++cell) {
                double dot = 0.0;
                for (int64_t i = 0; i < c.d; ++i) { dot += double(qv[(t * c.n_head + h) * c.d + i]) * kf[(cell * c.n_head_kv + hk) * c.d + i]; }
                scores[cell] = std::isinf(mask_value(c, t, cell)) ? -INFINITY : dot * scale;
                best = std::max(best, scores[cell]);
            }
            std::fill(mixed.begin(), mixed.end(), 0.0);
            double total = 0.0;
            for (int64_t cell = 0; best != -INFINITY && cell < c.n_kv; ++cell) {
                const double w = std::exp(scores[cell] - best);
                total += w;
                for (int64_t i = 0; i < c.d; ++i) { mixed[i] += w * vf[(cell * c.n_head_kv + hk) * c.d + i]; }
            }
            for (int64_t i = 0; i < c.d; ++i) {
                const double expected = total > 0 ? mixed[i] / total : 0.0;
                const float got = result[(t * c.n_head + h) * c.d + i];
                require(std::isfinite(got), "non-finite output");
                worst = std::max(worst, std::fabs(got - expected));
            }
        }
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(host_buf);
    ggml_backend_buffer_free(dev_buf);
    ggml_free(host_ctx);
    ggml_free(dev_ctx);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return worst / value_scale;
}

double run_mixed_precision(ggml_type host_type) {
    const Case c{256, 16, 2, 1024, 1, 1000, 512};
    ggml_backend_t gpu = ggml_backend_cuda_init(0), cpu = ggml_backend_cpu_init();
    require(gpu && cpu, "backends");
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_context * host_ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    ggml_context * dev_ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    auto * q = ggml_new_tensor_3d(host_ctx, GGML_TYPE_F32, c.d, c.n_head, c.n_tokens);
    auto * k = ggml_new_tensor_3d(host_ctx, host_type, c.d, c.n_head_kv, c.n_kv);
    auto * v = ggml_new_tensor_3d(host_ctx, host_type, c.d, c.n_head_kv, c.n_kv);
    auto * mask = ggml_new_tensor_2d(host_ctx, GGML_TYPE_F16, c.n_kv, c.n_tokens);
    auto * k_dev = ggml_new_tensor_3d(dev_ctx, GGML_TYPE_F16, c.d, c.n_head_kv, c.gpu_cells);
    auto * v_dev = ggml_new_tensor_3d(dev_ctx, GGML_TYPE_F16, c.d, c.n_head_kv, c.gpu_cells);
    ggml_backend_buffer_t host_buf = ggml_backend_alloc_ctx_tensors(host_ctx, cpu);
    ggml_backend_buffer_t dev_buf = ggml_backend_alloc_ctx_tensors(dev_ctx, gpu);

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> qv(ggml_nelements(q)), kf(ggml_nelements(k)), vf(ggml_nelements(v));
    for (auto & x : qv) { x = 2.0f * normal(rng); }
    for (auto & x : kf) { x = normal(rng); }
    for (auto & x : vf) { x = normal(rng); }

    std::vector<char> kb(ggml_nbytes(k)), vb(ggml_nbytes(v));
    ggml_quantize_chunk(host_type, kf.data(), kb.data(), 0, ggml_nrows(k), k->ne[0], nullptr);
    ggml_quantize_chunk(host_type, vf.data(), vb.data(), 0, ggml_nrows(v), v->ne[0], nullptr);
    ggml_get_type_traits(host_type)->to_float(kb.data(), kf.data(), kf.size());
    ggml_get_type_traits(host_type)->to_float(vb.data(), vf.data(), vf.size());

    const size_t gpu_elems = static_cast<size_t>(c.d * c.n_head_kv * c.gpu_cells);
    std::vector<ggml_fp16_t> kh_dev(gpu_elems), vh_dev(gpu_elems), mh(c.n_kv * c.n_tokens);
    ggml_fp32_to_fp16_row(kf.data(), kh_dev.data(), gpu_elems);
    ggml_fp32_to_fp16_row(vf.data(), vh_dev.data(), gpu_elems);
    for (size_t i = 0; i < gpu_elems; ++i) {
        kf[i] = ggml_fp16_to_fp32(kh_dev[i]);
        vf[i] = ggml_fp16_to_fp32(vh_dev[i]);
    }
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t cell = 0; cell < c.n_kv; ++cell) { mh[t * c.n_kv + cell] = ggml_fp32_to_fp16(mask_value(c, t, cell)); }
    }
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kb.data(), 0, kb.size());
    ggml_backend_tensor_set(v, vb.data(), 0, vb.size());
    ggml_backend_tensor_set(mask, mh.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(k_dev, kh_dev.data(), 0, ggml_nbytes(k_dev));
    ggml_backend_tensor_set(v_dev, vh_dev.data(), 0, ggml_nbytes(v_dev));

    const float scale = 1.0f / std::sqrt(float(c.d));
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * prior = argus_ggml_cuda_decode_partials(ctx, q, k_dev, v_dev, mask, 0, scale);
    require(prior != nullptr, "GPU partials declined in mixed-precision check");
    const int64_t rest = c.n_kv - c.gpu_cells;
    ggml_tensor * k_host = ggml_view_3d(ctx, k, c.d, c.n_head_kv, rest, k->nb[1], k->nb[2], c.gpu_cells * k->nb[2]);
    ggml_tensor * v_host = ggml_view_3d(ctx, v, c.d, c.n_head_kv, rest, v->nb[1], v->nb[2], c.gpu_cells * v->nb[2]);
    ggml_tensor * out = argus_ggml_decode_attention_after(ctx, q, k_host, v_host, mask, c.gpu_cells, prior, scale);
    require(out != nullptr, "mixed-precision split attention declined");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backends[] = {gpu, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
    ggml_backend_sched_set_tensor_backend(sched, out->src[4], cpu);
    ggml_backend_sched_set_tensor_backend(sched, out, cpu);
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "mixed-precision graph compute");
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));

    const int64_t group = c.n_head / c.n_head_kv;
    double worst = 0.0, value_scale = 0.0;
    for (float x : vf) { value_scale = std::max(value_scale, double(std::fabs(x))); }
    std::vector<double> scores(c.n_kv), mixed(c.d);
    for (int64_t h = 0; h < c.n_head; ++h) {
        const int64_t hk = h / group;
        double best = -INFINITY;
        for (int64_t cell = 0; cell < c.n_kv; ++cell) {
            double dot = 0.0;
            for (int64_t i = 0; i < c.d; ++i) { dot += double(qv[h * c.d + i]) * kf[(cell * c.n_head_kv + hk) * c.d + i]; }
            scores[cell] = std::isinf(mask_value(c, 0, cell)) ? -INFINITY : dot * scale;
            best = std::max(best, scores[cell]);
        }
        std::fill(mixed.begin(), mixed.end(), 0.0);
        double total = 0.0;
        for (int64_t cell = 0; cell < c.n_kv; ++cell) {
            const double w = std::exp(scores[cell] - best);
            total += w;
            for (int64_t i = 0; i < c.d; ++i) { mixed[i] += w * vf[(cell * c.n_head_kv + hk) * c.d + i]; }
        }
        for (int64_t i = 0; i < c.d; ++i) {
            const double expected = mixed[i] / total;
            worst = std::max(worst, std::fabs(double(result[h * c.d + i]) - expected));
        }
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(host_buf);
    ggml_backend_buffer_free(dev_buf);
    ggml_free(host_ctx);
    ggml_free(dev_ctx);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return worst / value_scale;
}
} // namespace

int main() try {
    const Case cases[] = {
        {256, 16, 2, 20480, 1, 20001, 8192},   // Qwen3.6 layer: GPU part and host part
        {512, 16, 2, 6144, 1, 6000, 2048},     // Gemma 4 global layer
        {256, 16, 4, 3072, 3, 3000, 1024},     // UI-Mate, three draft tokens
        {128, 32, 8, 1024, 2, 1000, 1000},     // host part is only masked padding
        {256, 16, 2, 4096, 1, 4000, 4096},     // the whole context on the GPU
        {256, 16, 2, 4096, 4, 4000, 4096},     // Qwen3.6 4-token MTP / speculative batch on GPU mirror
        {64, 14, 2, 777, 1, 777, 777},         // Qwen2.5-0.5B, whole context on the GPU
        {256, 16, 2, 300, 1, 257, 256},        // one block of GPU cells, a partial host block
    };
    for (const auto & c : cases) {
        const double error = run(c);
        std::printf("d=%lld heads=%lld/%lld n_kv=%lld tokens=%lld gpu_cells=%lld error=%.3g\n", (long long) c.d,
                    (long long) c.n_head, (long long) c.n_head_kv, (long long) c.n_kv, (long long) c.n_tokens,
                    (long long) c.gpu_cells, error);
        require(error < 2e-6, "split attention is not within fp32 rounding of the float64 reference");
    }
    for (ggml_type ht : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        const double c1_err = run_mixed_precision(ht);
        std::printf("c1_mixed_tier host_type=%s c1_err=%.3g\n", ggml_type_name(ht), c1_err);
        require(c1_err < 2e-6, "C1 mixed-precision split attention is not within fp32 rounding of reference");
    }
    std::printf("ok\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
