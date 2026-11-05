#include "argus_config.h"
#include "ggml_decode_attention.h"
#include "ggml_decode_e1.h"
#include "ggml_layer_prefill.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void test_hot_expert_cache() {
    setenv("ARGUS_AUTO", "1", 1);
    setenv("ARGUS_MOE_CACHE", "1", 1);
    setenv("ARGUS_HOT_EXPERTS", "32", 1);
    argus_reset_config();
    argus_moe_reset_stats();

    const int n_experts = 256;
    const int d_in = 128;
    const int d_ff = 64;
    const size_t ctx_bytes = 64 * 1024 * 1024;
    ggml_init_params params = { ctx_bytes, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * up_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_in, d_ff, n_experts);
    ggml_tensor * gate_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_in, d_ff, n_experts);
    ggml_tensor * down_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_ff, d_in, n_experts);
    ggml_set_name(up_exps, "blk.0.ffn_up_exps.weight");
    ggml_set_name(gate_exps, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(down_exps, "blk.0.ffn_down_exps.weight");

    ArgusNamedTensors tensors = {
        {"blk.0.ffn_up_exps.weight", up_exps},
        {"blk.0.ffn_gate_exps.weight", gate_exps},
        {"blk.0.ffn_down_exps.weight", down_exps},
    };
    argus_moe_init_host_experts(tensors, 1);

    const int n_used = 8;
    for (int step = 0; step < 64; ++step) {
        ggml_init_params step_params = { 4 * 1024 * 1024, nullptr, false };
        ggml_context * sctx = ggml_init(step_params);
        ggml_tensor * cur = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, d_ff, n_used, 1);
        ggml_tensor * sel = ggml_new_tensor_2d(sctx, GGML_TYPE_I32, n_used, 1);
        float * cur_d = static_cast<float *>(cur->data);
        for (int i = 0; i < d_ff * n_used; ++i) { cur_d[i] = float(i) * 0.01f; }
        int32_t * sel_d = static_cast<int32_t *>(sel->data);
        for (int k = 0; k < n_used; ++k) {
            if (step < 48) {
                sel_d[k] = (k < 7) ? ((step * 3 + k) % 24) : (32 + ((step * 17 + k) % 200));
            } else {
                sel_d[k] = (k < 7) ? (100 + ((step * 5 + k) % 24)) : ((step * 13 + k) % 24);
            }
        }
        ggml_tensor * obs = argus_moe_observe_routing(sctx, cur, sel, up_exps, gate_exps, down_exps, nullptr, 0);
        ggml_cgraph * gf = ggml_new_graph(sctx);
        ggml_build_forward_expand(gf, obs);
        ggml_graph_compute_with_ctx(sctx, gf, 1);
        ggml_free(sctx);
    }

    const ArgusMoeStats st = argus_moe_stats();
    std::printf("[Stage-1 MoE Test] advised_tensors=%llu advised_bytes=%llu total_routed=%llu hot_hits=%llu (%.2f%%) promotions=%llu demotions=%llu\n",
                (unsigned long long) st.advised_tensors,
                (unsigned long long) st.advised_bytes,
                (unsigned long long) st.total_routed,
                (unsigned long long) st.hot_hits,
                st.total_routed ? 100.0 * double(st.hot_hits) / double(st.total_routed) : 0.0,
                (unsigned long long) st.promotions,
                (unsigned long long) st.demotions);
    if (st.advised_tensors != 3 || st.total_routed != 64 * 8 || st.promotions == 0 || st.hot_hits == 0) {
        std::fprintf(stderr, "Stage-1 MoE test failed assertions!\n");
        std::exit(1);
    }
    unsetenv("ARGUS_MOE_CACHE");
    unsetenv("ARGUS_HOT_EXPERTS");
    unsetenv("ARGUS_AUTO");
    argus_moe_reset_stats();
    argus_reset_config();
    ggml_free(ctx);
}

static void test_multipole_and_top_r() {
    const int dk = 256;
    const int dv = 256;
    const int n_blocks = 64;
    const int T = static_cast<int>(argus_k1::e1_block);
    const int n_cells = n_blocks * T;

    std::vector<float> q(dk, 0.0f);
    for (int d = 0; d < dk; ++d) {
        const int seg = d / 8;
        if (seg == 2 || seg == 9 || seg == 17 || seg == 28) {
            q[d] = 0.85f * std::sin(float(d) * 0.37f);
        } else {
            q[d] = 0.04f * std::cos(float(d) * 0.19f);
        }
    }

    std::vector<ggml_fp16_t> k_f16(size_t(n_cells) * dk);
    std::vector<ggml_fp16_t> v_f16(size_t(n_cells) * dv);

    for (int b = 0; b < n_blocks; ++b) {
        for (int c = 0; c < T; ++c) {
            const int cell = b * T + c;
            for (int d = 0; d < dk; ++d) {
                float kv = 0.0f;
                if (b == 0 || b == n_blocks - 1) {
                    kv = (c < 4) ? 1.8f * q[d] : 0.4f * q[d];
                } else {
                    const float center = -0.15f * q[d];
                    const float jitter = 0.008f * std::sin(float(cell * 13 + d));
                    kv = center + jitter;
                }
                k_f16[size_t(cell) * dk + d] = ggml_fp32_to_fp16(kv);
            }
            for (int d = 0; d < dv; ++d) {
                const float v_center = (b == 0 || b == n_blocks - 1)
                    ? 0.2f * std::cos(float(d) * 0.11f)
                    : 1.5f + 0.5f * std::sin(float(d) * 0.07f);
                const float v_jitter = 0.02f * std::cos(float(cell * 7 + d));
                v_f16[size_t(cell) * dv + d] = ggml_fp32_to_fp16(v_center + v_jitter);
            }
        }
    }

    const argus_k1::QueryTopR q_top_r = argus_k1::prepare_query_top_r(q.data(), dk);
    float mu_k[512], rad_k[512], mu_v[512];
    const float r_blk = argus_k1::compute_key_box(&k_f16[size_t(T) * dk], dk, T, dk, mu_k, rad_k);
    argus_k1::compute_value_centroid(&v_f16[size_t(T) * dv], dv, T, dv, mu_v);

    const auto [c_dot, ub_dot] = argus_k1::box_dot_bounds(q.data(), q_top_r, mu_k, rad_k, r_blk, dk, 1e30f, true);
    const auto [c_dot_full, ub_dot_full] = argus_k1::box_dot_bounds(q.data(), q_top_r, mu_k, rad_k, r_blk, dk, -1e30f, false);
    std::printf("[Stage-2/3 Box Test] q_tail_norm=%.4f R_block=%.4f Stage1a_ub=%.4f Stage1b_ub=%.4f c_dot=%.4f dipole_gap=%.6f mu_v[0]=%.4f\n",
                q_top_r.q_tail_norm, r_blk, ub_dot, ub_dot_full, c_dot_full, ub_dot_full - c_dot_full, mu_v[0]);
    if (ub_dot < ub_dot_full || ub_dot_full < c_dot_full || r_blk <= 0.0f) {
        std::fprintf(stderr, "Box bound invariant violated!\n");
        std::exit(1);
    }

    // Verify Top-r + Multipole across causal mask, RoPE-rotated Q/K, Q8_0/Q4_0 quantization, and multi-token MTP (n_tokens = 4).
    for (ggml_type kv_t : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        for (int64_t n_tok : {1, 4}) {
            ggml_backend_t cpu = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(cpu, 2);
            ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 16, nullptr, true});
            const int64_t n_h = 8, n_hkv = 2, n_kv_test = 512;
            ggml_tensor * qt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dk, n_h, n_tok);
            ggml_tensor * kt = ggml_new_tensor_3d(ctx, kv_t, dk, n_hkv, n_kv_test);
            ggml_tensor * vt = ggml_new_tensor_3d(ctx, kv_t, dv, n_hkv, n_kv_test);
            ggml_tensor * mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_test, n_tok);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);

            std::vector<float> q_vec(dk * n_h * n_tok), k_vec(dk * n_hkv * n_kv_test), v_vec(dv * n_hkv * n_kv_test);
            for (int64_t t = 0; t < n_tok; ++t) {
                for (int64_t h = 0; h < n_h; ++h) {
                    const float theta = 0.03f * float(t + 1);
                    for (int d = 0; d < dk; d += 2) {
                        const float x0 = q[d], x1 = q[d + 1];
                        q_vec[(t * n_h + h) * dk + d]     = x0 * std::cos(theta) - x1 * std::sin(theta);
                        q_vec[(t * n_h + h) * dk + d + 1] = x0 * std::sin(theta) + x1 * std::cos(theta);
                    }
                }
            }
            for (int64_t c = 0; c < n_kv_test; ++c) {
                const bool hot = (c < 8 || c >= 496);
                const float theta = 0.01f * float(c % 64);
                for (int64_t hk = 0; hk < n_hkv; ++hk) {
                    for (int d = 0; d < dk; d += 2) {
                        const float base_x = hot ? 1.8f * q[d] : -0.25f * q[d] + 0.01f * std::sin(float(c + d));
                        const float base_y = hot ? 1.8f * q[d + 1] : -0.25f * q[d + 1] + 0.01f * std::cos(float(c + d));
                        k_vec[(c * n_hkv + hk) * dk + d]     = base_x * std::cos(theta) - base_y * std::sin(theta);
                        k_vec[(c * n_hkv + hk) * dk + d + 1] = base_x * std::sin(theta) + base_y * std::cos(theta);
                        v_vec[(c * n_hkv + hk) * dv + d]     = 0.5f * std::sin(float(d + hk)) + 0.02f * std::cos(float(c));
                        v_vec[(c * n_hkv + hk) * dv + d + 1] = 0.5f * std::cos(float(d + hk)) + 0.02f * std::sin(float(c));
                    }
                }
            }
            auto set_tensor = [](ggml_tensor * t, const std::vector<float> & src) {
                std::vector<char> raw(ggml_nbytes(t));
                if (t->type == GGML_TYPE_F16) {
                    ggml_fp32_to_fp16_row(src.data(), reinterpret_cast<ggml_fp16_t *>(raw.data()), src.size());
                } else {
                    ggml_quantize_chunk(t->type, src.data(), raw.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
                }
                ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
            };
            ggml_backend_tensor_set(qt, q_vec.data(), 0, ggml_nbytes(qt));
            set_tensor(kt, k_vec);
            set_tensor(vt, v_vec);
            std::vector<ggml_fp16_t> m_vec(n_kv_test * n_tok);
            for (int64_t t = 0; t < n_tok; ++t) {
                for (int64_t c = 0; c < n_kv_test; ++c) {
                    const float mv = (c <= 500 + t) ? 0.0f : -INFINITY;
                    m_vec[t * n_kv_test + c] = ggml_fp32_to_fp16(mv);
                }
            }
            ggml_backend_tensor_set(mt, m_vec.data(), 0, ggml_nbytes(mt));

            auto run_k1 = [&](float eps, bool mp, bool tr) {
                ArgusConfig cfg{};
                cfg.k1_enabled = true;
                cfg.skip_eps = eps;
                cfg.multipole_enabled = mp;
                cfg.top_r_enabled = tr;
                argus_set_config(cfg);
                ggml_context * gctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
                ggml_tensor * out = argus_ggml_decode_attention(gctx, qt, kt, vt, mt, false, 1.0f / std::sqrt(float(dk)));
                ggml_cgraph * gf = ggml_new_graph(gctx);
                ggml_build_forward_expand(gf, out);
                ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
                ggml_gallocr_alloc_graph(ga, gf);
                ggml_backend_graph_compute(cpu, gf);
                std::vector<float> res(ggml_nelements(out));
                ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
                ggml_gallocr_free(ga);
                ggml_free(gctx);
                return res;
            };
            const auto ref_out = run_k1(0.0f, false, false);
            (void) run_k1(1e-3f, true, true); // Pass 1 builds boxes
            const auto sota_out = run_k1(1e-3f, true, true); // Pass 2 uses Top-r + Multipole
            double max_diff = 0.0;
            for (size_t i = 0; i < ref_out.size(); ++i) {
                max_diff = std::max(max_diff, std::fabs(double(ref_out[i]) - double(sota_out[i])));
            }
            std::printf("[MTP/Quant/RoPE Test] type=%s n_tokens=%lld max_diff=%.3e\n",
                        ggml_type_name(kv_t), (long long) n_tok, max_diff);
            if (!std::isfinite(max_diff) || max_diff > 5e-3) {
                std::fprintf(stderr, "MTP/Quant/RoPE test exceeded error bound: %.3e\n", max_diff);
                std::exit(1);
            }
            argus_reset_config();
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            ggml_backend_free(cpu);
        }
    }
}

int main() {
    test_hot_expert_cache();
    test_multipole_and_top_r();
    argus_k1_print_summary();
    std::printf("ALL SOTA-3 TESTS PASSED\n");
    return 0;
}

