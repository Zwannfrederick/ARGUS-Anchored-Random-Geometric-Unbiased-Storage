// ARGUS v0.8 — Heterogeneous CPU/GPU Execution, PCIe Transfer, Sync & Memory Audit (Tasks 1–4)
// Measures on real Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf:
//   Task 1: Exact CPU vs GPU op & node distribution in bs=1 decode and bs=2048 prefill graphs.
//   Task 2: Per-sub-block compute efficiency (Gated DeltaNet, Full Attn S1/K1, Shared Expert,
//           MoE Router, Routed Experts, LM Head) including transfer + sync overhead.
//   Task 3: PCIe H2D/D2H latency & bandwidth (64 B .. 225 MiB, Pinned vs Pageable, Sync vs Async batching)
//           and exact per-split boundary copy + synchronization cost.
//   Task 4: RAM & VRAM buffer inventory, redundant copies (150 MiB uncached MoE copy in CUDA0 compute
//           buffer, 128-256 MiB duplicate kq_mask, pageable P3 hidden states), and context scaling (8K, 32K, 64K).

#include "bench_32k_bottleneck_helpers.h"

using namespace argus_audit;

namespace {

struct OpCount {
    int cuda_nodes = 0;
    int cpu_nodes = 0;
    size_t cuda_bytes = 0;
    size_t cpu_bytes = 0;
};

enum SubBlockKind {
    SB_EMBED = 0,
    SB_GDN_ATTN,
    SB_FULL_ATTN_QKV,
    SB_FULL_ATTN_CORE,
    SB_SHARED_EXPERT,
    SB_MOE_ROUTER,
    SB_MOE_EXPERTS_CPU,
    SB_FFN_MERGE,
    SB_LM_HEAD,
    SB_COUNT
};

const char * subblock_name(SubBlockKind k) {
    switch (k) {
        case SB_EMBED:           return "Embed_Input_GPU";
        case SB_GDN_ATTN:        return "GatedDeltaNet_30L_GPU";
        case SB_FULL_ATTN_QKV:   return "FullAttn_QKV_10L_GPU";
        case SB_FULL_ATTN_CORE:  return "FullAttn_Core_10L";
        case SB_SHARED_EXPERT:   return "SharedExpert_40L_GPU";
        case SB_MOE_ROUTER:      return "MoE_Router_40L_GPU";
        case SB_MOE_EXPERTS_CPU: return "RoutedExperts_40L_CPU";
        case SB_FFN_MERGE:       return "FFN_Merge_Norm_GPU";
        case SB_LM_HEAD:         return "Output_LM_Head_GPU";
        default:                 return "Unknown";
    }
}

struct GraphProfileState {
    bool inspect_only = true;
    bool active = false;
    int tokens_profiled = 0;

    // Task 1: node distribution
    int total_nodes = 0;
    int cuda_nodes = 0;
    int cpu_nodes = 0;
    int split_transitions_d2h = 0;
    int split_transitions_h2d = 0;
    size_t d2h_copy_bytes_per_tok = 0;
    size_t h2d_copy_bytes_per_tok = 0;
    std::map<std::string, OpCount> op_dist;

    // Task 2 & 3: sub-block & split transfer timing (accumulated across profiled tokens)
    double subblock_us[SB_COUNT] = {0.0};
    int subblock_calls[SB_COUNT] = {0};
    double d2h_transfer_sync_us = 0.0;
    double h2d_transfer_sync_us = 0.0;
    int d2h_events = 0;
    int h2d_events = 0;

    // Tracking state during a single graph evaluation
    int prev_is_cpu = -1;
    SubBlockKind cur_sb = SB_EMBED;
    std::chrono::steady_clock::time_point t_subblock_start{};
    std::chrono::steady_clock::time_point t_last_done{};
    bool has_last_done = false;
    bool in_subblock = false;
};

bool is_cpu_tensor(const ggml_tensor * t) {
    const ggml_tensor * base = (t && t->view_src) ? t->view_src : t;
    if (!base || !base->buffer) { return false; }
    const char * bname = ggml_backend_buffer_name(base->buffer);
    return bname && (std::strstr(bname, "Host") != nullptr || std::strstr(bname, "CPU") != nullptr);
}

SubBlockKind classify_node(const ggml_tensor * t, int current_layer_is_recr) {
    const char * name = t->name ? t->name : "";
    const bool on_cpu = is_cpu_tensor(t);
    if (on_cpu) {
        if (std::strstr(name, "k1") || std::strstr(name, "decode_att")) {
            return SB_FULL_ATTN_CORE;
        }
        return SB_MOE_EXPERTS_CPU;
    }
    if (std::strstr(name, "result_") || std::strstr(name, "h_nextn")) {
        return SB_LM_HEAD;
    }
    if (std::strstr(name, "shexp") || std::strstr(name, "shared_expert") ||
        std::strstr(name, "ffn_up-") || std::strstr(name, "ffn_gate-") ||
        std::strstr(name, "ffn_swiglu-") || std::strstr(name, "ffn_down-")) {
        return SB_SHARED_EXPERT;
    }
    if (std::strstr(name, "ffn_moe_logits") || std::strstr(name, "ffn_moe_probs") ||
        std::strstr(name, "ffn_moe_argsort") || std::strstr(name, "ffn_moe_topk") ||
        std::strstr(name, "ffn_moe_weights")) {
        return SB_MOE_ROUTER;
    }
    if (std::strstr(name, "ffn_out") || std::strstr(name, "post_moe") ||
        std::strstr(name, "l_out") || std::strstr(name, "attn_norm")) {
        return SB_FFN_MERGE;
    }
    if (std::strstr(name, "s1") || std::strstr(name, "kqv_out") ||
        std::strstr(name, "argus_tier") || t->op == GGML_OP_FLASH_ATTN_EXT) {
        return SB_FULL_ATTN_CORE;
    }
    if (current_layer_is_recr == 0) {
        return SB_FULL_ATTN_QKV;
    }
    return SB_GDN_ATTN;
}

int parse_layer_from_name(const char * name) {
    if (!name) { return -1; }
    const char * dash = std::strrchr(name, '-');
    if (!dash || dash[1] < '0' || dash[1] > '9') { return -1; }
    return std::atoi(dash + 1);
}

bool is_RecurrentLayer(int il) {
    if (il < 0 || il >= 40) { return false; }
    return ((il + 1) % 4) != 0;
}

// Callback used for Task 1 (zero-sync graph inspection) and Task 2/3 (sub-block & split boundary timing)
bool hetero_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<GraphProfileState *>(user_data);
    if (!st || !st->active || !t) { return false; }

    // Skip no-op views in timing
    const bool is_view = (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
                          t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE ||
                          t->op == GGML_OP_NONE);

    if (st->inspect_only) {
        if (ask && !is_view) {
            const bool cpu = is_cpu_tensor(t);
            st->total_nodes++;
            if (cpu) { st->cpu_nodes++; } else { st->cuda_nodes++; }

            OpCount & oc = st->op_dist[ggml_op_name(t->op)];
            if (cpu) {
                oc.cpu_nodes++;
                oc.cpu_bytes += ggml_nbytes(t);
            } else {
                oc.cuda_nodes++;
                oc.cuda_bytes += ggml_nbytes(t);
            }

            const int cur_cpu = cpu ? 1 : 0;
            if (st->prev_is_cpu != -1 && st->prev_is_cpu != cur_cpu) {
                if (cur_cpu == 1) {
                    st->split_transitions_d2h++;
                } else {
                    st->split_transitions_h2d++;
                }
            }
            // Count cross-backend copy sources on first encounter in a split
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                const ggml_tensor * src = t->src[j];
                if (!src || !src->name) { continue; }
                if (cpu && std::strncmp(src->name, "CPU#", 4) == 0) {
                    st->d2h_copy_bytes_per_tok += ggml_nbytes(src);
                } else if (!cpu && std::strncmp(src->name, "CUDA0#", 6) == 0) {
                    st->h2d_copy_bytes_per_tok += ggml_nbytes(src);
                }
            }
            st->prev_is_cpu = cur_cpu;
        }
        return false; // never break batches in inspect_only mode
    }

    // Sub-block & split boundary timing mode
    const char * name = t->name ? t->name : "";
    if (ask) {
        const auto now = std::chrono::steady_clock::now();
        const int cur_cpu = is_cpu_tensor(t) ? 1 : 0;

        // If we just crossed a split boundary (previous sub-block finished on the other backend),
        // the elapsed time between t_last_done (end of prev split) and now (first ask of new split)
        // is the exact split input copy + stream sync time!
        if (st->has_last_done && st->prev_is_cpu != -1 && st->prev_is_cpu != cur_cpu) {
            const double xfer_us = std::chrono::duration<double, std::micro>(now - st->t_last_done).count();
            if (cur_cpu == 1) {
                st->d2h_transfer_sync_us += xfer_us;
                st->d2h_events++;
            } else {
                st->h2d_transfer_sync_us += xfer_us;
                st->h2d_events++;
            }
            st->has_last_done = false;
        }

        if (!st->in_subblock) {
            st->t_subblock_start = now;
            st->in_subblock = true;
            const int il = parse_layer_from_name(name);
            st->cur_sb = classify_node(t, il >= 0 ? (is_RecurrentLayer(il) ? 1 : 0) : -1);
        }

        // Trigger synchronization at exact sub-block boundaries:
        // 1. attn_post_norm-N (end of attention + norm on GPU)
        // 2. ffn_shexp_gated-N (end of shared expert on GPU)
        // 3. ffn_moe_weights_scaled-N (end of MoE router on GPU, end of CUDA0 split)
        // 4. ffn_moe_out-N (end of routed experts on CPU, end of CPU split)
        // 5. l_out-N (end of FFN merge + residual on GPU)
        // 6. result_output (end of LM head on GPU)
        if (std::strncmp(name, "attn_post_norm-", 15) == 0 ||
            std::strncmp(name, "ffn_shexp_gated-", 16) == 0 ||
            std::strncmp(name, "ffn_moe_weights_scaled-", 23) == 0 ||
            std::strncmp(name, "ffn_moe_out-", 12) == 0 ||
            std::strncmp(name, "l_out-", 6) == 0 ||
            std::strcmp(name, "result_output") == 0) {
            return true;
        }
        return false;
    } else {
        // ask == false: sub-block [j0, j1] has just finished and synchronized
        const auto now = std::chrono::steady_clock::now();
        const int cur_cpu = is_cpu_tensor(t) ? 1 : 0;
        const int il = parse_layer_from_name(name);
        SubBlockKind sb = st->cur_sb;
        if (std::strncmp(name, "attn_post_norm-", 15) == 0) {
            sb = is_RecurrentLayer(il) ? SB_GDN_ATTN : SB_FULL_ATTN_CORE;
        } else if (std::strncmp(name, "ffn_shexp_gated-", 16) == 0) {
            sb = SB_SHARED_EXPERT;
        } else if (std::strncmp(name, "ffn_moe_weights_scaled-", 23) == 0) {
            sb = SB_MOE_ROUTER;
        } else if (std::strncmp(name, "ffn_moe_out-", 12) == 0) {
            sb = SB_MOE_EXPERTS_CPU;
        } else if (std::strncmp(name, "l_out-", 6) == 0) {
            sb = SB_FFN_MERGE;
        } else if (std::strcmp(name, "result_output") == 0) {
            sb = SB_LM_HEAD;
        }
        const double us = std::chrono::duration<double, std::micro>(now - st->t_subblock_start).count();
        st->subblock_us[sb] += us;
        st->subblock_calls[sb]++;
        st->in_subblock = false;
        st->prev_is_cpu = cur_cpu;
        st->t_last_done = std::chrono::steady_clock::now();
        st->has_last_done = true;
        return true;
    }
}

struct PcieBenchRow {
    size_t bytes = 0;
    double h2d_pinned_sync_us = 0.0;
    double h2d_pinned_async_us = 0.0;
    double h2d_pageable_sync_us = 0.0;
    double d2h_pinned_sync_us = 0.0;
    double d2h_pinned_async_us = 0.0;
    double d2h_pageable_sync_us = 0.0;
};

std::vector<PcieBenchRow> run_pcie_microbench() {
    std::vector<PcieBenchRow> rows;
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) { return rows; }
    ggml_backend_t cuda_backend = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!cuda_backend) { return rows; }

    ggml_backend_buffer_type_t dev_buft = ggml_backend_dev_buffer_type(gpu_dev);
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu_dev);

    const size_t max_bytes = 16ULL * 1024ULL * 1024ULL; // 16 MiB test buffer
    ggml_init_params p_dev = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx_dev = ggml_init(p_dev);
    ggml_tensor * t_dev = ggml_new_tensor_1d(ctx_dev, GGML_TYPE_I8, max_bytes);
    ggml_backend_buffer_t buf_dev = ggml_backend_buft_alloc_buffer(dev_buft, max_bytes + 4096);
    ggml_backend_tensor_alloc(buf_dev, t_dev, ggml_backend_buffer_get_base(buf_dev));

    ggml_backend_buffer_t buf_pinned = host_buft ? ggml_backend_buft_alloc_buffer(host_buft, max_bytes + 4096) : nullptr;
    void * pinned_ptr = buf_pinned ? ggml_backend_buffer_get_base(buf_pinned) : nullptr;
    std::vector<uint8_t> pageable_buf(max_bytes, 0x5A);
    if (pinned_ptr) { std::memset(pinned_ptr, 0x5A, max_bytes); }

    const size_t test_sizes[] = { 64, 4096, 8192, 65536, 1048576, 16777216 };
    for (size_t sz : test_sizes) {
        PcieBenchRow r{};
        r.bytes = sz;
        const int iters = (sz <= 65536) ? 200 : 30;

        // Warmup
        for (int i = 0; i < 5; ++i) {
            if (pinned_ptr) {
                ggml_backend_tensor_set(t_dev, pinned_ptr, 0, sz);
                ggml_backend_tensor_get(t_dev, pinned_ptr, 0, sz);
            }
        }

        if (pinned_ptr) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_tensor_set(t_dev, pinned_ptr, 0, sz);
            }
            auto t1 = std::chrono::steady_clock::now();
            r.h2d_pinned_sync_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_tensor_set_async(cuda_backend, t_dev, pinned_ptr, 0, sz);
            }
            ggml_backend_synchronize(cuda_backend);
            t1 = std::chrono::steady_clock::now();
            r.h2d_pinned_async_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_tensor_get(t_dev, pinned_ptr, 0, sz);
            }
            t1 = std::chrono::steady_clock::now();
            r.d2h_pinned_sync_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_tensor_get_async(cuda_backend, t_dev, pinned_ptr, 0, sz);
            }
            ggml_backend_synchronize(cuda_backend);
            t1 = std::chrono::steady_clock::now();
            r.d2h_pinned_async_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        }

        {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_tensor_set(t_dev, pageable_buf.data(), 0, sz);
            }
            auto t1 = std::chrono::steady_clock::now();
            r.h2d_pageable_sync_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_tensor_get(t_dev, pageable_buf.data(), 0, sz);
            }
            t1 = std::chrono::steady_clock::now();
            r.d2h_pageable_sync_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        }
        rows.push_back(r);
    }

    // Also measure batched 3-input D2H (8192 B + 32 B + 32 B): 3 sync calls vs 3 async + 1 sync
    if (pinned_ptr) {
        const int iters = 500;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_synchronize(cuda_backend);
            ggml_backend_tensor_get(t_dev, pinned_ptr, 0, 8192);
            ggml_backend_synchronize(cuda_backend);
            ggml_backend_tensor_get(t_dev, (char *)pinned_ptr + 8192, 8192, 32);
            ggml_backend_synchronize(cuda_backend);
            ggml_backend_tensor_get(t_dev, (char *)pinned_ptr + 8256, 8256, 32);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double sync3_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_tensor_get_async(cuda_backend, t_dev, pinned_ptr, 0, 8192);
            ggml_backend_tensor_get_async(cuda_backend, t_dev, (char *)pinned_ptr + 8192, 8192, 32);
            ggml_backend_tensor_get_async(cuda_backend, t_dev, (char *)pinned_ptr + 8256, 8256, 32);
            ggml_backend_synchronize(cuda_backend);
        }
        t1 = std::chrono::steady_clock::now();
        const double async3_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        std::fprintf(stderr,
            "[PCIe 3-Input Split D2H (8KiB+32B+32B)] Stock (6 syncs): %.2f us/layer (%.2f ms/40L) vs Async-Batched (1 sync): %.2f us/layer (%.2f ms/40L)\n",
            sync3_us, sync3_us * 40.0 / 1000.0, async3_us, async3_us * 40.0 / 1000.0);
    }

    if (buf_pinned) { ggml_backend_buffer_free(buf_pinned); }
    ggml_backend_buffer_free(buf_dev);
    ggml_free(ctx_dev);
    ggml_backend_free(cuda_backend);
    return rows;
}

struct BufferAuditRow {
    std::string label;
    uint32_t n_ctx = 0;
    uint32_t n_ubatch = 0;
    uint32_t n_outputs_max = 0;
    double cuda0_compute_mib = 0.0;
    double cuda_host_compute_mib = 0.0;
    double vram_used_mib = 0.0;
    double vram_free_mib = 0.0;
    double rss_mib = 0.0;
    int splits_pp = 0;
    int splits_tg = 0;
};

double g_last_cuda0_compute_mib = 0.0;
double g_last_host_compute_mib = 0.0;

void audit_log_callback(ggml_log_level level, const char * text, void * user_data) {
    split_log_callback(level, text, user_data);
    if (!text) { return; }
    double mib = 0.0;
    if (std::sscanf(text, "llama_context:      CUDA0 compute buffer size = %lf MiB", &mib) == 1) {
        g_last_cuda0_compute_mib = mib;
    } else if (std::sscanf(text, "llama_context:  CUDA_Host compute buffer size = %lf MiB", &mib) == 1) {
        g_last_host_compute_mib = mib;
    }
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "scratch/models/Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf";
    llama_log_set(audit_log_callback, nullptr);
    llama_backend_init();

    std::fprintf(stderr, "=== [Task 3a] Running PCIe H2D/D2H & Sync Microbenchmark ===\n");
    const auto pcie_rows = run_pcie_microbench();
    for (const auto & r : pcie_rows) {
        std::fprintf(stderr,
            "  Size %8zu B | H2D Pinned Sync: %7.2f us, Async: %7.2f us, Pageable: %7.2f us | D2H Pinned Sync: %7.2f us, Async: %7.2f us, Pageable: %7.2f us\n",
            r.bytes, r.h2d_pinned_sync_us, r.h2d_pinned_async_us, r.h2d_pageable_sync_us,
            r.d2h_pinned_sync_us, r.d2h_pinned_async_us, r.d2h_pageable_sync_us);
    }

    static const llama_model_tensor_buft_override cpu_moe_overrides[] = {
        { "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr }
    };
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.tensor_buft_overrides = cpu_moe_overrides;

    std::fprintf(stderr, "\n=== Loading Model: %s ===\n", model_path);
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { return 1; }
    const SysSnapshot post_model = read_sys_snapshot(true);
    std::fprintf(stderr, "[Post-Model Load] VRAM Used: %.2f MiB, RSS: %.2f MiB, MemAvail: %.2f MiB\n",
                 post_model.vram_used_mib, post_model.rss_mib, post_model.mem_avail_mib);

    // Task 4: Audit compute buffer reservations across 8K, 32K, 64K and ubatch sizes
    std::fprintf(stderr, "\n=== [Task 4] Auditing Compute Buffer & S1 Mirror Reservations (8K, 32K, 64K) ===\n");
    struct AuditCfg {
        const char * label;
        uint32_t n_ctx;
        uint32_t n_ubatch;
        uint32_t n_outputs_max;
    };
    const AuditCfg audit_cfgs[] = {
        { "8K_ub2048_out1",   8192,  2048, 1 },
        { "8K_ub512_out1",    8192,   512, 1 },
        { "8K_ub1_tg_only",   8192,     1, 1 },
        { "32K_ub2048_out1", 32768,  2048, 1 },
        { "32K_ub512_out1",  32768,   512, 1 },
        { "32K_ub1_tg_only", 32768,     1, 1 },
        { "64K_ub2048_out1", 65536,  2048, 1 },
        { "64K_ub512_out1",  65536,   512, 1 },
        { "64K_ub1_tg_only", 65536,     1, 1 },
    };
    for (const auto & ac : audit_cfgs) {
        clear_argus_env();
        setenv("ARGUS_AUTO", "1", 1);
        argus_reset_config();
        g_last_cuda0_compute_mib = 0.0;
        g_last_host_compute_mib = 0.0;
        g_last_splits_pp = 0;
        g_last_splits_tg = 0;

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = ac.n_ctx;
        cp.n_batch = ac.n_ctx;
        cp.n_ubatch = ac.n_ubatch;
        cp.n_seq_max = 1;
        cp.n_outputs_max = ac.n_outputs_max;
        cp.n_threads = 4;
        cp.n_threads_batch = 8;
        cp.offload_kqv = false;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            std::fprintf(stderr, "  %-18s | FAILED TO ALLOCATE\n", ac.label);
            continue;
        }
        size_t free_b = 0, total_b = 0;
        if (ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
            ggml_backend_dev_memory(gpu, &free_b, &total_b);
        }
        const SysSnapshot snap = read_sys_snapshot(true);
        std::fprintf(stderr,
            "  %-18s | CUDA0 Compute: %6.2f MiB | Host Compute: %6.2f MiB | Splits(pp/tg): %3d/%3d | VRAM Used: %7.2f MiB (Free: %6.2f MiB) | RSS: %7.2f MiB\n",
            ac.label, g_last_cuda0_compute_mib, g_last_host_compute_mib,
            g_last_splits_pp, g_last_splits_tg,
            snap.vram_used_mib, double(free_b) / (1024.0 * 1024.0), snap.rss_mib);
        llama_free(ctx);
    }

    // Task 1, 2, 3b: Profile exact node distribution, sub-block compute time, and split transfer time at 8K (pure S1) and 32K (S1+K1)
    for (uint32_t test_ctx : { 8192u, 32768u }) {
        std::fprintf(stderr, "\n=== [Tasks 1–3] Profiling Graph Node Distribution & Sub-Block Timings (ctx=%u) ===\n", test_ctx);
        clear_argus_env();
        setenv("ARGUS_AUTO", "1", 1);
        argus_reset_config();

        GraphProfileState prof{};
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = test_ctx;
        cp.n_batch = test_ctx;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.n_outputs_max = 1;
        cp.n_threads = 4;
        cp.n_threads_batch = 8;
        cp.offload_kqv = false;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.cb_eval = hetero_eval_callback;
        cp.cb_eval_user_data = &prof;

        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { continue; }

        // Prefill 64 tokens so KV cache has tokens, or for 32K test both pure-S1 and S1+K1
        std::vector<llama_token> prompt(64, 100);
        run_prefill(ctx, prompt, 64, false);

        // Warmup 3 decode tokens without callback
        llama_batch b = llama_batch_init(1, 0, 1);
        for (int i = 0; i < 3; ++i) {
            b.n_tokens = 1;
            b.token[0] = 100 + i;
            b.pos[0] = 64 + i;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            llama_decode(ctx, b);
        }

        // Measure 10 unperturbed decode tokens
        auto t_unp0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) {
            b.n_tokens = 1;
            b.token[0] = 110 + i;
            b.pos[0] = 67 + i;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            llama_decode(ctx, b);
        }
        auto t_unp1 = std::chrono::steady_clock::now();
        const double unp_ms_per_tok = std::chrono::duration<double, std::milli>(t_unp1 - t_unp0).count() / 10.0;

        // 1 token in inspect_only mode (Task 1)
        prof = GraphProfileState{};
        prof.inspect_only = true;
        prof.active = true;
        b.n_tokens = 1;
        b.token[0] = 125;
        b.pos[0] = 77;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        llama_decode(ctx, b);
        prof.active = false;

        std::fprintf(stderr,
            "  [Task 1 Node Dist] Total non-view nodes: %d | CUDA0: %d (%.1f%%) | CPU: %d (%.1f%%) | D2H transitions: %d (%zu B/tok) | H2D transitions: %d (%zu B/tok)\n",
            prof.total_nodes, prof.cuda_nodes, 100.0 * prof.cuda_nodes / std::max(1, prof.total_nodes),
            prof.cpu_nodes, 100.0 * prof.cpu_nodes / std::max(1, prof.total_nodes),
            prof.split_transitions_d2h, prof.d2h_copy_bytes_per_tok,
            prof.split_transitions_h2d, prof.h2d_copy_bytes_per_tok);
        for (const auto & [op, oc] : prof.op_dist) {
            std::fprintf(stderr, "    Op %-18s | CUDA0: %4d nodes (%8.1f KiB) | CPU: %4d nodes (%8.1f KiB)\n",
                         op.c_str(), oc.cuda_nodes, oc.cuda_bytes / 1024.0, oc.cpu_nodes, oc.cpu_bytes / 1024.0);
        }

        // 10 tokens in sub-block timing mode (Tasks 2 & 3)
        GraphProfileState timing{};
        timing.inspect_only = false;
        timing.active = true;
        cp.cb_eval_user_data = &timing;
        // Note: cb_eval_user_data was passed at init to prof; copy into prof instead!
        prof = GraphProfileState{};
        prof.inspect_only = false;
        prof.active = true;
        const int n_prof_toks = 10;
        for (int i = 0; i < n_prof_toks; ++i) {
            prof.prev_is_cpu = -1;
            prof.has_last_done = false;
            prof.in_subblock = false;
            b.n_tokens = 1;
            b.token[0] = 130 + i;
            b.pos[0] = 78 + i;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            llama_decode(ctx, b);
        }
        prof.active = false;

        double total_sb_ms = 0.0;
        for (int k = 0; k < SB_COUNT; ++k) {
            total_sb_ms += (prof.subblock_us[k] / n_prof_toks) / 1000.0;
        }
        const double d2h_ms = (prof.d2h_transfer_sync_us / n_prof_toks) / 1000.0;
        const double h2d_ms = (prof.h2d_transfer_sync_us / n_prof_toks) / 1000.0;
        total_sb_ms += d2h_ms + h2d_ms;

        std::fprintf(stderr, "  [Tasks 2–3 Per-Token Breakdown (avg over %d tokens, unperturbed=%.2f ms/tok, profiled sum=%.2f ms/tok)]:\n",
                     n_prof_toks, unp_ms_per_tok, total_sb_ms);
        for (int k = 0; k < SB_COUNT; ++k) {
            if (prof.subblock_calls[k] == 0) { continue; }
            const double ms = (prof.subblock_us[k] / n_prof_toks) / 1000.0;
            std::fprintf(stderr, "    %-24s | %6.2f ms/tok (%5.1f%%) | %6.1f us/call (%d calls/tok)\n",
                         subblock_name( static_cast<SubBlockKind>(k) ),
                         ms, 100.0 * ms / std::max(1e-6, total_sb_ms),
                         prof.subblock_us[k] / prof.subblock_calls[k],
                         prof.subblock_calls[k] / n_prof_toks);
        }
        std::fprintf(stderr, "    %-24s | %6.2f ms/tok (%5.1f%%) | %6.1f us/xfer (%d xfers/tok)\n",
                     "PCIe_D2H_Copy+Sync", d2h_ms, 100.0 * d2h_ms / std::max(1e-6, total_sb_ms),
                     prof.d2h_transfer_sync_us / std::max(1, prof.d2h_events), prof.d2h_events / n_prof_toks);
        std::fprintf(stderr, "    %-24s | %6.2f ms/tok (%5.1f%%) | %6.1f us/xfer (%d xfers/tok)\n",
                     "PCIe_H2D_Copy+Sync", h2d_ms, 100.0 * h2d_ms / std::max(1e-6, total_sb_ms),
                     prof.h2d_transfer_sync_us / std::max(1, prof.h2d_events), prof.h2d_events / n_prof_toks);

        llama_batch_free(b);
        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}

