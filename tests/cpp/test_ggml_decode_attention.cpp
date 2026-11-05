// K1 check: grouped decode attention vs a double-precision reference, stock GGML flash
// attention, and itself at other thread counts (the result must not change by one bit).
// Usage: test_ggml_decode_attention
#include "ggml_decode_attention.h"
#include "argus_config.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Case { ggml_type type; int64_t d, n_head, n_head_kv, n_kv, n_tokens, used; };

struct Result { std::vector<float> k1, stock, reference; float value_scale; };

void fill(ggml_tensor * t, std::mt19937 & rng, float sigma) {
    std::normal_distribution<float> normal(0.0f, sigma);
    std::vector<float> values(ggml_nelements(t));
    for (auto & x : values) { x = normal(rng); }
    std::vector<char> bytes(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) { std::memcpy(bytes.data(), values.data(), bytes.size()); }
    else if (t->type == GGML_TYPE_F16) { ggml_fp32_to_fp16_row(values.data(), reinterpret_cast<ggml_fp16_t *>(bytes.data()), values.size()); }
    else { ggml_quantize_chunk(t->type, values.data(), bytes.data(), 0, ggml_nrows(t), t->ne[0], nullptr); }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

std::vector<float> dequantized(const ggml_tensor * t) {
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    std::vector<float> values(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) { std::memcpy(values.data(), bytes.data(), bytes.size()); }
    else { ggml_get_type_traits(t->type)->to_float(bytes.data(), values.data(), values.size()); }
    return values;
}

std::vector<float> read(const ggml_tensor * t) {
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
    return values;
}

// Token t of n_tokens sees cells [0, used - n_tokens + 1 + t); cells past `used` are padding.
float mask_value(const Case & c, int64_t token, int64_t cell) {
    return cell < c.used - c.n_tokens + 1 + token ? 0.0f : -std::numeric_limits<float>::infinity();
}

Result run(const Case & c, int threads, float scale) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_init_params params{ggml_tensor_overhead() * 32 + ggml_graph_overhead() * 2, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.d, c.n_head, c.n_tokens);
    auto * k = ggml_new_tensor_3d(ctx, c.type, c.d, c.n_head_kv, c.n_kv);
    auto * v = ggml_new_tensor_3d(ctx, c.type, c.d, c.n_head_kv, c.n_kv);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_kv, c.n_tokens);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(1234);
    fill(q, rng, 2.0f);
    fill(k, rng, 1.0f);
    fill(v, rng, 1.0f);
    std::vector<ggml_fp16_t> mask_data(c.n_kv * c.n_tokens);
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t cell = 0; cell < c.n_kv; ++cell) { mask_data[t * c.n_kv + cell] = ggml_fp32_to_fp16(mask_value(c, t, cell)); }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    ggml_context * graph_ctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead() * 2, nullptr, true});
    ggml_tensor * k1 = argus_ggml_decode_attention(graph_ctx, q, k, v, mask, false, scale);
    if (!k1) { throw std::runtime_error("K1 declined a supported problem"); }
    ggml_tensor * stock = ggml_flash_attn_ext(graph_ctx, ggml_permute(graph_ctx, q, 0, 2, 1, 3), ggml_permute(graph_ctx, k, 0, 2, 1, 3),
                                              ggml_permute(graph_ctx, v, 0, 2, 1, 3), mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(stock, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx);
    ggml_build_forward_expand(graph, k1);
    ggml_build_forward_expand(graph, stock);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) { throw std::runtime_error("graph allocation"); }
    ggml_backend_graph_compute(backend, graph);

    Result result{read(k1), read(stock), {}, 0.0f};
    const auto qv = read(q), kv = dequantized(k), vv = dequantized(v);
    for (float x : vv) { result.value_scale = std::max(result.value_scale, std::fabs(x)); }
    const int64_t group = c.n_head / c.n_head_kv;
    result.reference.assign(c.d * c.n_head * c.n_tokens, 0.0f);
    std::vector<double> scores(c.n_kv), mixed(c.d);
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t h = 0; h < c.n_head; ++h) {
            const int64_t hk = h / group;
            double best = -INFINITY;
            for (int64_t cell = 0; cell < c.n_kv; ++cell) {
                const float bias = mask_value(c, t, cell);
                double dot = 0.0;
                for (int64_t i = 0; i < c.d; ++i) { dot += double(qv[(t * c.n_head + h) * c.d + i]) * kv[(cell * c.n_head_kv + hk) * c.d + i]; }
                scores[cell] = std::isinf(bias) ? -INFINITY : dot * scale + bias;
                best = std::max(best, scores[cell]);
            }
            if (best == -INFINITY) { continue; }
            double total = 0.0;
            std::fill(mixed.begin(), mixed.end(), 0.0);
            for (int64_t cell = 0; cell < c.n_kv; ++cell) {
                const double w = std::exp(scores[cell] - best);
                if (w == 0.0) { continue; }
                total += w;
                for (int64_t i = 0; i < c.d; ++i) { mixed[i] += w * vv[(cell * c.n_head_kv + hk) * c.d + i]; }
            }
            for (int64_t i = 0; i < c.d; ++i) { result.reference[(t * c.n_head + h) * c.d + i] = float(mixed[i] / total); }
        }
    }
    ggml_gallocr_free(allocator);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}

double max_error(const std::vector<float> & a, const std::vector<float> & b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) { return INFINITY; }
        worst = std::max(worst, std::fabs(double(a[i]) - b[i]));
    }
    return worst;
}

void require(bool condition, const std::string & what) {
    if (!condition) { throw std::runtime_error(what); }
}

void test_certified_skip_and_config() {
    // 1. P3 128K VRAM budget planner: 642 MiB free VRAM with 498 MiB MoE expert cache fits via adaptive 64 MiB floor.
    const size_t planned = argus_plan_vram_budget(642ULL << 20, 498ULL << 20, ArgusConfig{});
    require(planned == (498ULL << 20), "128K P3 VRAM budget plan failed");

    // 2. Unified autonomous config (ARGUS_AUTO=1) and programmatic override.
    setenv("ARGUS_AUTO", "1", 1);
    argus_reset_config();
    const ArgusConfig auto_cfg = argus_get_config();
    require(auto_cfg.auto_mode && auto_cfg.k1_enabled && auto_cfg.vram_auto &&
            auto_cfg.prefill_on_gpu && auto_cfg.layer_prefill &&
            auto_cfg.madv_random && auto_cfg.hetero_sched && auto_cfg.hetero_overlap &&
            !auto_cfg.multipole_enabled && !auto_cfg.top_r_enabled &&
            !auto_cfg.moe_cache_enabled && auto_cfg.skip_eps == 0.0f,
            "ARGUS_AUTO=1 did not enable calibrated autonomous levers");
    const ArgusHardwareProfile hw = argus_probe_hardware(1750ULL << 20, 3824ULL << 20);
    require(hw.cpu_logical_cores >= 1 && hw.ram_total_bytes > 0, "Hardware probe returned invalid host metrics");
    const ArgusOpPlacementCost c1 = argus_evaluate_moe_cost(hw, 1, 489000000ULL, 256, 8);
    const ArgusOpPlacementCost c32 = argus_evaluate_moe_cost(hw, 32, 489000000ULL, 256, 8);
    require(!c1.prefer_gpu && c1.overlap_with_cpu && c32.prefer_gpu,
            "Heterogeneous MoE cost model did not switch from CPU token-major (bs=1) to GPU P3 (bs=32)");
    require(argus_hetero_select_ubatch(2048, 32768, 1750ULL << 20, 3824ULL << 20, false) == 512,
            "Adaptive ubatch did not cap 32K P3 ubatch at 512");
    unsetenv("ARGUS_AUTO");
    argus_reset_config();

    // 3. E1 Bounded-Mass Approximate Skipping: Stage 2 (Pass 1 V skip) + Stage 1 (Pass 2 K+V bounding-box skip).
    const int64_t d = 128, n_head = 8, n_head_kv = 2, n_kv = 1024;
    const float scale = 1.0f / std::sqrt(float(d));
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 16, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head, 1);
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, n_head_kv, n_kv);
    auto * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, n_head_kv, n_kv);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(99);
    std::normal_distribution<float> noise(0.0f, 0.15f);
    std::vector<float> qv(d * n_head, 1.5f), kf(d * n_head_kv * n_kv), vf(d * n_head_kv * n_kv);
    for (int64_t cell = 0; cell < n_kv; ++cell) {
        const float boost = (cell < 4 || cell >= n_kv - 4) ? 2.5f : -1.5f;
        for (int64_t i = 0; i < d * n_head_kv; ++i) {
            kf[cell * d * n_head_kv + i] = boost + noise(rng);
            vf[cell * d * n_head_kv + i] = noise(rng) * 4.0f;
        }
    }
    std::vector<ggml_fp16_t> kh(kf.size()), vh(vf.size()), mh(n_kv, ggml_fp32_to_fp16(0.0f));
    ggml_fp32_to_fp16_row(kf.data(), kh.data(), kf.size());
    ggml_fp32_to_fp16_row(vf.data(), vh.data(), vf.size());
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kh.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, vh.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(mask, mh.data(), 0, ggml_nbytes(mask));

    auto eval_once = [&]() {
        ggml_context * gctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
        ggml_tensor * out = argus_ggml_decode_attention(gctx, q, k, v, mask, false, scale);
        ggml_cgraph * graph = ggml_new_graph(gctx);
        ggml_build_forward_expand(graph, out);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(alloc, graph);
        ggml_backend_graph_compute(backend, graph);
        std::vector<float> res = read(out);
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        return res;
    };

    const std::vector<float> exact = eval_once();
    ArgusConfig skip_cfg = argus_get_config();
    skip_cfg.skip_eps = 1e-3f;
    argus_set_config(skip_cfg);
    argus_k1_reset_stats();
    const std::vector<float> pass1 = eval_once();
    const ArgusK1Stats s1 = argus_k1_stats();
    require(s1.skipped_v_blocks > 0, "E1 Stage 2 did not skip cold V blocks on Pass 1");
    const std::vector<float> pass2 = eval_once();
    const ArgusK1Stats s2 = argus_k1_stats();
    require(s2.skipped_k_blocks > 0, "E1 Stage 1 did not skip cold K+V blocks via cached bounding boxes on Pass 2");
    require(max_error(pass1, exact) < 1e-3 && max_error(pass2, exact) < 1e-3, "E1 skip exceeded error bound");
    std::printf("certified_skip pass1_v_skipped=%llu pass2_k_skipped=%llu total_blocks=%llu skip_err=%.3g\n",
                (unsigned long long) s1.skipped_v_blocks, (unsigned long long) s2.skipped_k_blocks,
                (unsigned long long) s2.total_blocks, max_error(pass2, exact));

    argus_k1_reset_stats();
    argus_reset_config();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
} // namespace

int main() try {
    setenv("ARGUS_KV_CPU_ATTENTION", "k1", 1);
    const Case cases[] = {
        {GGML_TYPE_F16, 256, 16, 2, 20224, 1, 20001},   // Qwen3.6 full-attention layer, padded view
        {GGML_TYPE_Q8_0, 256, 16, 2, 5120, 1, 5000},
        {GGML_TYPE_Q4_0, 256, 16, 2, 5120, 1, 5000},
        {GGML_TYPE_F16, 512, 16, 2, 4352, 1, 4200},     // Gemma 4 global layer
        {GGML_TYPE_F16, 256, 16, 4, 3072, 3, 3000},     // UI-Mate, three draft tokens
        {GGML_TYPE_F32, 64, 8, 8, 700, 2, 650},         // no grouping
        {GGML_TYPE_F16, 40, 4, 1, 300, 1, 257},         // head dim off the SIMD width
        {GGML_TYPE_F16, 256, 16, 2, 33, 1, 33},         // one cell past a block
        {GGML_TYPE_F16, 256, 16, 2, 1, 1, 1},
        {GGML_TYPE_F16, 128, 32, 8, 12, 12, 11},        // first token sees nothing
    };
    for (const auto & c : cases) {
        const float scale = 1.0f / std::sqrt(float(c.d));
        const Result one = run(c, 1, scale);
        const double k1_error = max_error(one.k1, one.reference) / one.value_scale;
        const double stock_error = max_error(one.stock, one.reference) / one.value_scale;
        std::printf("type=%s d=%lld heads=%lld/%lld n_kv=%lld tokens=%lld k1_error=%.3g stock_error=%.3g\n",
                    ggml_type_name(c.type), (long long) c.d, (long long) c.n_head, (long long) c.n_head_kv,
                    (long long) c.n_kv, (long long) c.n_tokens, k1_error, stock_error);
        // F32 accumulation over at most ~20K cells: a few float ulps of the value scale.
        require(k1_error < 2e-6, "K1 is not within fp32 rounding of the float64 reference");
        require(stock_error < 5e-3, "stock and reference disagree: the reference itself is suspect");
        for (int threads : {2, 3, 4, 8}) {
            const Result other = run(c, threads, scale);
            require(std::memcmp(other.k1.data(), one.k1.data(), one.k1.size() * sizeof(float)) == 0,
                    "K1 result depends on the thread count");
        }
    }
    test_certified_skip_and_config();
    // Declines: disabled, too many tokens for decode, unsupported features.
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, 32);
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 2, 64);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 32);
    require(!argus_ggml_decode_attention(ctx, q, k, k, mask, false, 1.0f), "K1 accepted a prefill batch");
    require(!argus_ggml_decode_attention(ctx, q, k, k, mask, true, 1.0f), "K1 accepted unsupported features");
    setenv("ARGUS_KV_CPU_ATTENTION", "stock", 1);
    auto * q1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, 1);
    require(!argus_ggml_decode_attention(ctx, q1, k, k, mask, false, 1.0f), "K1 ran while disabled");
    ggml_free(ctx);
    std::printf("ok\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
