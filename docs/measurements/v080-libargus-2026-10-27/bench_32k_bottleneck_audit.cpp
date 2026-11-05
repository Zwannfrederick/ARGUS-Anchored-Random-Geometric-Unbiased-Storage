// ARGUS v0.8 — 32K Runtime Bottleneck Isolation & Systems Telemetry Audit
// Answers the 5 core systems questions on Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf at 32,768 context:
//   Q1. How much of CPU MoE expert weights (18.56 GiB across 40 layers) is physically in RAM (mincore)?
//   Q2. Which layers (blk.0 .. blk.39) generate major/minor page faults and disk read I/O?
//   Q3. Does POSIX_MADV_RANDOM genuinely reduce OS readahead wasted I/O vs POSIX_MADV_NORMAL?
//   Q4. How do K/V/Box memory bytes read and isolated kernel execution time change with Multipole on S1 & K1?
//   Q5. What is the standalone contribution of each graph-split optimization (282 -> 262 -> 182 / 202 -> 102 splits) to 32K decode speed?

#include "bench_32k_bottleneck_helpers.h"

using namespace argus_audit;

int main(int argc, char ** argv) {
    const char * model_path = "scratch/models/Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf";
    bool q3_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--q3-only") == 0) {
            q3_only = true;
        } else {
            model_path = argv[i];
        }
    }

    llama_log_set(split_log_callback, nullptr);
    llama_backend_init();

    static const llama_model_tensor_buft_override cpu_moe_overrides[] = {
        { "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr }
    };

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.tensor_buft_overrides = cpu_moe_overrides;

    std::fprintf(stderr, "=== [32K Bottleneck Audit] Loading model: %s ===\n", model_path);
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "Failed to load model %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Initialize host MoE tensor table with a quick 64-tok context and capture Post-Load residency
    clear_argus_env();
    setenv("ARGUS_AUTO", "1", 1);
    argus_reset_config();
    {
        llama_context_params wparams = llama_context_default_params();
        wparams.n_ctx = 256;
        wparams.n_batch = 64;
        wparams.n_ubatch = 64;
        wparams.n_threads = 4;
        wparams.n_threads_batch = 8;
        wparams.offload_kqv = false;
        llama_context * wctx = llama_init_from_model(model, wparams);
        if (wctx) { llama_free(wctx); }
    }
    const LayerResidencySummary res_after_load = query_all_moe_residency();
    std::fprintf(stderr, "[Q1] Post-mmap MoE Residency: %.2f / %.2f MiB (%.2f%%)\n",
                 res_after_load.resident_mib, res_after_load.total_mib, res_after_load.resident_pct);

    // Build 32K prompt (32,640 tokens) and run 32K prefill once (or 256 tokens if --q3-only)
    const int active_ctx = q3_only ? 512 : N_CTX_32K;
    const int target_prompt = q3_only ? 256 : (N_CTX_32K - 128);
    const auto prompt = build_multi_needle_prompt(vocab, target_prompt);
    const int n_prompt = int(prompt.size());

    std::vector<uint8_t> saved_32k_state;
    llama_token first_tok_32k = 0;
    double prefill_32k_s = 0.0;
    LayerResidencySummary res_after_prefill{};
    LayerResidencySummary res_after_decode{};
    LayerFaultTracker steady_fault_tracker{};

    {
        clear_argus_env();
        setenv("ARGUS_AUTO", "1", 1);
        argus_reset_config();

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = active_ctx;
        cparams.n_batch         = active_ctx;
        cparams.n_ubatch        = q3_only ? 256 : 2048;
        cparams.n_seq_max       = 1;
        cparams.n_outputs_max   = 1;
        cparams.n_threads       = 4;
        cparams.n_threads_batch = 8;
        cparams.offload_kqv     = false;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { throw std::runtime_error("Failed to create prefill context"); }

        std::fprintf(stderr, "\n=== [Step 1] Running Prefill (%d tokens, P3 layer-major) ===\n", n_prompt);
        const auto t0 = std::chrono::steady_clock::now();
        run_prefill(ctx, prompt, q3_only ? 256 : 2048, !q3_only);
        const auto t1 = std::chrono::steady_clock::now();
        prefill_32k_s = std::chrono::duration<double>(t1 - t0).count();

        res_after_prefill = query_all_moe_residency();
        std::fprintf(stderr, "[Q1] Post-32K-Prefill MoE Residency: %.2f / %.2f MiB (%.2f%%), Prefill: %.2f s (%.2f tok/s)\n",
                     res_after_prefill.resident_mib, res_after_prefill.total_mib, res_after_prefill.resident_pct,
                     prefill_32k_s, double(n_prompt) / prefill_32k_s);

        const float * pf_logits = llama_get_logits_ith(ctx, -1);
        first_tok_32k = greedy_argmax(pf_logits, n_vocab);

        const size_t state_sz = llama_state_get_size(ctx);
        saved_32k_state.resize(state_sz);
        const size_t written = llama_state_get_data(ctx, saved_32k_state.data(), saved_32k_state.size());
        saved_32k_state.resize(written);
        std::fprintf(stderr, "Saved 32K state buffer: %.2f MiB\n", double(written) / (1024.0 * 1024.0));

        // Decode 32 tokens in steady state and check residency
        llama_batch b = llama_batch_init(1, 0, 1);
        llama_token tok = first_tok_32k;
        for (int s = 0; s < 32; ++s) {
            b.n_tokens = 1;
            b.token[0] = tok;
            b.pos[0] = n_prompt + s;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            llama_decode(ctx, b);
            tok = greedy_argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        llama_batch_free(b);
        res_after_decode = query_all_moe_residency();
        std::fprintf(stderr, "[Q1] Post-32K-Steady-Decode MoE Residency: %.2f / %.2f MiB (%.2f%%)\n",
                     res_after_decode.resident_mib, res_after_decode.total_mib, res_after_decode.resident_pct);
        llama_free(ctx);
    }

    // =========================================================================
    // QUESTION 2 & QUESTION 3:
    //   2A. Steady-state per-layer faults across blk.0 .. blk.39 (8 decode tokens)
    //   2B & 3A. Cold-layer readahead test (evict blk.0 .. blk.7 = 3,712 MiB):
    //            Compare MADV_NORMAL (MADV_RANDOM=0) vs MADV_RANDOM (MADV_RANDOM=1)
    //            per-layer majflt, minflt, disk_read_mib, resident_mib_after,
    //            unique routed expert MiB, readahead bloat factor, and decode latency!
    // =========================================================================
    struct ColdMadvReport {
        bool madv_random = false;
        double cold_evicted_mib = 0.0;
        double res_before_l0_7_mib = 0.0;
        double res_after_l0_7_mib = 0.0;
        double theoretical_routed_l0_7_mib = 0.0;
        double unrouted_bloat_l0_7_mib = 0.0;
        double bloat_ratio = 0.0;
        uint64_t total_majflt = 0;
        uint64_t total_minflt = 0;
        double total_disk_read_mib = 0.0;
        double io_bytes_per_majflt = 0.0;
        double decode_8tok_s = 0.0;
        double decode_tok_s = 0.0;
        double token_p50_ms = 0.0;
        double token_p95_ms = 0.0;
        LayerFaultStats per_layer[N_LAYERS]{};
        std::vector<double> layer_res_after_mib;
    };

    // 2A: Steady-state per-layer fault scan (all 40 layers warm in RAM)
    {
        clear_argus_env();
        setenv("ARGUS_AUTO", "1", 1);
        argus_reset_config();

        steady_fault_tracker.active = true;
        steady_fault_tracker.reset();

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = N_CTX_32K;
        cparams.n_batch         = 32;
        cparams.n_ubatch        = 32;
        cparams.n_seq_max       = 1;
        cparams.n_outputs_max   = 1;
        cparams.n_threads       = 4;
        cparams.n_threads_batch = 8;
        cparams.offload_kqv     = false;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cparams.cb_eval         = layer_fault_cb;
        cparams.cb_eval_user_data = &steady_fault_tracker;

        llama_context * ctx = llama_init_from_model(model, cparams);
        llama_state_set_data(ctx, saved_32k_state.data(), saved_32k_state.size());
        steady_fault_tracker.reset();

        llama_batch b = llama_batch_init(1, 0, 1);
        llama_token tok = first_tok_32k;
        for (int s = 0; s < 8; ++s) {
            b.n_tokens = 1;
            b.token[0] = tok;
            b.pos[0] = n_prompt + s;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            llama_decode(ctx, b);
            tok = greedy_argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        llama_batch_free(b);
        llama_free(ctx);
    }

    auto run_cold_madv_experiment = [&](bool use_madv_random) -> ColdMadvReport {
        ColdMadvReport rep{};
        rep.madv_random = use_madv_random;

        clear_argus_env();
        setenv("ARGUS_AUTO", "1", 1);
        setenv("ARGUS_MADV_RANDOM", use_madv_random ? "1" : "0", 1);
        argus_reset_config();

        LayerFaultTracker tracker{};
        tracker.active = true;

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = active_ctx;
        cparams.n_batch         = 32;
        cparams.n_ubatch        = 32;
        cparams.n_seq_max       = 1;
        cparams.n_outputs_max   = 1;
        cparams.n_threads       = 4;
        cparams.n_threads_batch = 8;
        cparams.offload_kqv     = false;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cparams.cb_eval         = layer_fault_cb;
        cparams.cb_eval_user_data = &tracker;

        llama_context * ctx = llama_init_from_model(model, cparams);
        llama_state_set_data(ctx, saved_32k_state.data(), saved_32k_state.size());

        // Evict layers 0..7 (8 layers * 464 MiB = 3,712 MiB of host MoE weights)
        rep.cold_evicted_mib = 8.0 * 464.0;
        for (int il = 0; il < 8; ++il) {
            argus_moe_evict_layer_pages(il);
        }
        const LayerResidencySummary before_res = query_all_moe_residency();
        for (int il = 0; il < 8; ++il) {
            rep.res_before_l0_7_mib += before_res.layer_resident_mib[il];
        }

        tracker.reset();
        const SysSnapshot snap0 = read_sys_snapshot(false);
        std::vector<double> tok_lats_ms;
        llama_batch b = llama_batch_init(1, 0, 1);
        llama_token tok = first_tok_32k;
        const auto t_start = std::chrono::steady_clock::now();
        for (int s = 0; s < 8; ++s) {
            const auto ts0 = std::chrono::steady_clock::now();
            b.n_tokens = 1;
            b.token[0] = tok;
            b.pos[0] = n_prompt + s;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            llama_decode(ctx, b);
            const auto ts1 = std::chrono::steady_clock::now();
            tok_lats_ms.push_back(std::chrono::duration<double, std::milli>(ts1 - ts0).count());
            tok = greedy_argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        const auto t_end = std::chrono::steady_clock::now();
        const SysSnapshot snap1 = read_sys_snapshot(false);
        llama_batch_free(b);

        rep.decode_8tok_s = std::chrono::duration<double>(t_end - t_start).count();
        rep.decode_tok_s  = 8.0 / rep.decode_8tok_s;
        std::sort(tok_lats_ms.begin(), tok_lats_ms.end());
        rep.token_p50_ms  = percentile_sorted(tok_lats_ms, 50.0);
        rep.token_p95_ms  = percentile_sorted(tok_lats_ms, 95.0);

        rep.total_majflt = snap1.majflt >= snap0.majflt ? (snap1.majflt - snap0.majflt) : 0;
        rep.total_minflt = snap1.minflt >= snap0.minflt ? (snap1.minflt - snap0.minflt) : 0;
        const uint64_t read_b = snap1.io_read_bytes >= snap0.io_read_bytes ? (snap1.io_read_bytes - snap0.io_read_bytes) : 0;
        rep.total_disk_read_mib = double(read_b) / (1024.0 * 1024.0);
        rep.io_bytes_per_majflt = rep.total_majflt > 0 ? double(read_b) / double(rep.total_majflt) : 0.0;

        const LayerResidencySummary after_res = query_all_moe_residency();
        rep.layer_res_after_mib = after_res.layer_resident_mib;
        for (int il = 0; il < N_LAYERS; ++il) {
            rep.per_layer[il] = tracker.layers[il];
        }
        // Each unique routed expert in a layer occupies 464 MiB / 256 experts = 1.8125 MiB (1,856 KiB = 464 pages of 4 KiB)
        for (int il = 0; il < 8; ++il) {
            rep.res_after_l0_7_mib += after_res.layer_resident_mib[il];
            const double unique_exp_mib = double(tracker.layers[il].unique_experts.size()) * (464.0 / 256.0);
            rep.theoretical_routed_l0_7_mib += unique_exp_mib;
        }
        rep.unrouted_bloat_l0_7_mib = std::max(0.0, rep.res_after_l0_7_mib - rep.theoretical_routed_l0_7_mib);
        rep.bloat_ratio = rep.theoretical_routed_l0_7_mib > 0.0
            ? (rep.res_after_l0_7_mib / rep.theoretical_routed_l0_7_mib) : 1.0;

        std::fprintf(stderr,
                     "[Q3 Cold L0..7 Readahead] MADV_RANDOM=%d | Before=%.2f MiB | After=%.2f MiB | Routed=%.2f MiB | "
                     "Bloat=+%.2f MiB (%.2fx) | majflt=%llu | disk_read=%.2f MiB (%.1f KiB/majflt) | %.2f tok/s (p50=%.1f ms)\n",
                     use_madv_random ? 1 : 0, rep.res_before_l0_7_mib, rep.res_after_l0_7_mib,
                     rep.theoretical_routed_l0_7_mib, rep.unrouted_bloat_l0_7_mib, rep.bloat_ratio,
                     (unsigned long long) rep.total_majflt, rep.total_disk_read_mib,
                     rep.io_bytes_per_majflt / 1024.0, rep.decode_tok_s, rep.token_p50_ms);

        llama_free(ctx);
        return rep;
    };

    std::fprintf(stderr, "\n=== [Step 2] Running Q2 & Q3 Cold-Layer Readahead Isolation (MADV_NORMAL vs MADV_RANDOM) ===\n");
    const ColdMadvReport cold_madv_normal = run_cold_madv_experiment(false);
    const ColdMadvReport cold_madv_random = run_cold_madv_experiment(true);
    if (q3_only) {
        std::ofstream q3out("docs/measurements/v080-libargus-2026-10-27/q3_cold_readahead.json");
        q3out << "{\n  \"madv_normal\": {\n";
        q3out << "    \"res_before_l0_7_mib\": " << cold_madv_normal.res_before_l0_7_mib << ",\n";
        q3out << "    \"res_after_l0_7_mib\": " << cold_madv_normal.res_after_l0_7_mib << ",\n";
        q3out << "    \"theoretical_routed_l0_7_mib\": " << cold_madv_normal.theoretical_routed_l0_7_mib << ",\n";
        q3out << "    \"unrouted_bloat_l0_7_mib\": " << cold_madv_normal.unrouted_bloat_l0_7_mib << ",\n";
        q3out << "    \"bloat_ratio\": " << cold_madv_normal.bloat_ratio << ",\n";
        q3out << "    \"total_majflt\": " << cold_madv_normal.total_majflt << ",\n";
        q3out << "    \"total_minflt\": " << cold_madv_normal.total_minflt << ",\n";
        q3out << "    \"total_disk_read_mib\": " << cold_madv_normal.total_disk_read_mib << ",\n";
        q3out << "    \"io_kib_per_majflt\": " << (cold_madv_normal.io_bytes_per_majflt / 1024.0) << ",\n";
        q3out << "    \"decode_tok_s\": " << cold_madv_normal.decode_tok_s << ",\n";
        q3out << "    \"token_p50_ms\": " << cold_madv_normal.token_p50_ms << ",\n";
        q3out << "    \"token_p95_ms\": " << cold_madv_normal.token_p95_ms << ",\n";
        q3out << "    \"layers_0_7\": [\n";
        for (int il = 0; il < 8; ++il) {
            const auto & st = cold_madv_normal.per_layer[il];
            q3out << "      {\"layer\": " << il
                  << ", \"resident_after_mib\": " << cold_madv_normal.layer_res_after_mib[il]
                  << ", \"unique_routed_experts\": " << st.unique_experts.size()
                  << ", \"routed_mib\": " << (double(st.unique_experts.size()) * 464.0 / 256.0)
                  << ", \"majflt\": " << st.majflt
                  << ", \"minflt\": " << st.minflt
                  << ", \"disk_read_mib\": " << (double(st.io_read_bytes) / (1024.0 * 1024.0))
                  << ", \"elapsed_ms\": " << st.elapsed_ms << "}" << (il < 7 ? "," : "") << "\n";
        }
        q3out << "    ]\n  },\n  \"madv_random\": {\n";
        q3out << "    \"res_before_l0_7_mib\": " << cold_madv_random.res_before_l0_7_mib << ",\n";
        q3out << "    \"res_after_l0_7_mib\": " << cold_madv_random.res_after_l0_7_mib << ",\n";
        q3out << "    \"theoretical_routed_l0_7_mib\": " << cold_madv_random.theoretical_routed_l0_7_mib << ",\n";
        q3out << "    \"unrouted_bloat_l0_7_mib\": " << cold_madv_random.unrouted_bloat_l0_7_mib << ",\n";
        q3out << "    \"bloat_ratio\": " << cold_madv_random.bloat_ratio << ",\n";
        q3out << "    \"total_majflt\": " << cold_madv_random.total_majflt << ",\n";
        q3out << "    \"total_minflt\": " << cold_madv_random.total_minflt << ",\n";
        q3out << "    \"total_disk_read_mib\": " << cold_madv_random.total_disk_read_mib << ",\n";
        q3out << "    \"io_kib_per_majflt\": " << (cold_madv_random.io_bytes_per_majflt / 1024.0) << ",\n";
        q3out << "    \"decode_tok_s\": " << cold_madv_random.decode_tok_s << ",\n";
        q3out << "    \"token_p50_ms\": " << cold_madv_random.token_p50_ms << ",\n";
        q3out << "    \"token_p95_ms\": " << cold_madv_random.token_p95_ms << ",\n";
        q3out << "    \"layers_0_7\": [\n";
        for (int il = 0; il < 8; ++il) {
            const auto & st = cold_madv_random.per_layer[il];
            q3out << "      {\"layer\": " << il
                  << ", \"resident_after_mib\": " << cold_madv_random.layer_res_after_mib[il]
                  << ", \"unique_routed_experts\": " << st.unique_experts.size()
                  << ", \"routed_mib\": " << (double(st.unique_experts.size()) * 464.0 / 256.0)
                  << ", \"majflt\": " << st.majflt
                  << ", \"minflt\": " << st.minflt
                  << ", \"disk_read_mib\": " << (double(st.io_read_bytes) / (1024.0 * 1024.0))
                  << ", \"elapsed_ms\": " << st.elapsed_ms << "}" << (il < 7 ? "," : "") << "\n";
        }
        q3out << "    ]\n  }\n}\n";
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    // Ensure all 40 layers are 100% warm before Step 3 & Step 4
    {
        clear_argus_env();
        setenv("ARGUS_AUTO", "1", 1);
        argus_reset_config();
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        cparams.n_batch = 256;
        cparams.n_ubatch = 256;
        cparams.n_threads = 4;
        cparams.n_threads_batch = 8;
        cparams.offload_kqv = false;
        llama_context * wctx = llama_init_from_model(model, cparams);
        if (wctx) {
            std::vector<llama_token> wtoks(256, 100);
            for (int i = 0; i < 256; ++i) { wtoks[i] = prompt[i]; }
            run_prefill(wctx, wtoks, 256, false);
            llama_free(wctx);
        }
    }

    // =========================================================================
    // QUESTION 4:
    //   Multipole K/V/Box Memory Read Volume & Kernel Execution Time at 32K
    //   Evaluated on BOTH:
    //     - GPU S1 (ARGUS_KV_VRAM_BYTES=680000000, default 32K ARGUS_AUTO path)
    //     - CPU K1 (ARGUS_KV_VRAM_BYTES=0, host-RAM attention path)
    //   Across 3 modes:
    //     1. E1 OFF (skip_eps=0)
    //     2. E1 ON, Multipole OFF (skip_eps=1e-4, ARGUS_MULTIPOLE=0)
    //     3. E1 ON, Multipole ON  (skip_eps=1e-4, ARGUS_MULTIPOLE=1)
    // =========================================================================
    struct AttnProfileLayer {
        uint64_t k_blocks_total = 0;
        uint64_t k_blocks_skipped = 0;
        double k_skip_pct = 0.0;
        uint64_t v_cells_total = 0;
        uint64_t v_cells_skipped = 0;
        uint64_t v_blocks_skipped = 0;
        double v_skip_pct = 0.0;
        double k_mib_per_tok = 0.0;
        double v_mib_per_tok = 0.0;
        double box_mib_per_tok = 0.0;
        double total_mib_per_tok = 0.0;
        double kernel_us_per_tok = 0.0;
    };

    struct AttnProfileResult {
        std::string name;
        bool gpu_s1 = false;
        float skip_eps = 0.0f;
        bool multipole = false;
        double decode_tok_s = 0.0;
        double k_skip_pct = 0.0;
        double v_skip_pct = 0.0;
        double k_mib_per_tok = 0.0;
        double v_mib_per_tok = 0.0;
        double box_mib_per_tok = 0.0;
        double total_kv_mib_per_tok = 0.0;
        double kv_bandwidth_saved_pct = 0.0;
        double total_kernel_ms_per_tok = 0.0;
        std::map<int, AttnProfileLayer> layers;
    };

    auto run_q4_attn_profile = [&](const std::string & name, bool gpu_s1, float skip_eps, bool multipole) -> AttnProfileResult {
        AttnProfileResult res{};
        res.name = name;
        res.gpu_s1 = gpu_s1;
        res.skip_eps = skip_eps;
        res.multipole = multipole;

        clear_argus_env();
        setenv("ARGUS_AUTO", "1", 1);
        setenv("ARGUS_KV_VRAM_BYTES", gpu_s1 ? "680000000" : "0", 1);
        if (skip_eps > 0.0f) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.6g", skip_eps);
            setenv("ARGUS_KV_SKIP_EPS", buf, 1);
        } else {
            setenv("ARGUS_KV_SKIP_EPS", "0", 1);
        }
        setenv("ARGUS_MULTIPOLE", multipole ? "1" : "0", 1);
        setenv("ARGUS_TOP_R", "0", 1);
        setenv("ARGUS_PROFILE_KERNELS", "1", 1);
        argus_reset_config();

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = N_CTX_32K;
        cparams.n_batch         = 32;
        cparams.n_ubatch        = 32;
        cparams.n_seq_max       = 1;
        cparams.n_outputs_max   = 1;
        cparams.n_threads       = 4;
        cparams.n_threads_batch = 8;
        cparams.offload_kqv     = false;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        llama_context * ctx = llama_init_from_model(model, cparams);
        llama_state_set_data(ctx, saved_32k_state.data(), saved_32k_state.size());

        llama_batch b = llama_batch_init(1, 0, 1);
        // 1 warmup pass (16 tokens) so KeyBox sheets are populated
        {
            llama_token tok = first_tok_32k;
            for (int s = 0; s < 16; ++s) {
                b.n_tokens = 1;
                b.token[0] = tok;
                b.pos[0] = n_prompt + s;
                b.n_seq_id[0] = 1;
                b.seq_id[0][0] = 0;
                b.logits[0] = 1;
                llama_decode(ctx, b);
                tok = greedy_argmax(llama_get_logits_ith(ctx, -1), n_vocab);
            }
        }

        // Reset state & stats, then measure 32 decode tokens
        llama_state_set_data(ctx, saved_32k_state.data(), saved_32k_state.size());
        argus_k1_reset_stats();
        constexpr int N_PROF_TOK = 32;
        llama_token tok = first_tok_32k;
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < N_PROF_TOK; ++s) {
            b.n_tokens = 1;
            b.token[0] = tok;
            b.pos[0] = n_prompt + s;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            llama_decode(ctx, b);
            tok = greedy_argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        llama_synchronize(ctx);
        const auto t1 = std::chrono::steady_clock::now();
        llama_batch_free(b);

        res.decode_tok_s = double(N_PROF_TOK) / std::chrono::duration<double>(t1 - t0).count();

        const ArgusK1Stats tot = argus_k1_stats();
        res.k_skip_pct = tot.total_blocks > 0 ? 100.0 * double(tot.skipped_k_blocks) / double(tot.total_blocks) : 0.0;
        res.v_skip_pct = tot.total_v_cells > 0 ? 100.0 * double(tot.skipped_v_cells) / double(tot.total_v_cells) : 0.0;

        // Each cell in Qwen3.6-35B-A3B has d=256, F16 = 512 bytes per KV head.
        // Note: on GPU S1, each query-row group (16 query heads / 8 rows_per_block = 2 CTAs per KV head)
        // streams tiles into shared memory; on CPU K1, all 8 query heads per KV head share 1 register-blocked pass.
        // We compute exact physical tensor bytes read from the KV cache & KeyBox sheets per token:
        //   - Logical KV cache bytes per layer per token = 2 KV heads * n_kv * 512 B (K) + 2 * n_kv * 512 B (V) = 64.0 MiB/layer
        //   - Across 10 layers = 640.0 MiB/token when 0% skipped.
        // Wait: in GPU S1, 16 query heads / 2 KV heads = group 8 = rows_per_block 8!
        // Because group == rows_per_block (8 == 8), grid.z = (8 + 8 - 1) / 8 = 1 CTA per (chunk, head_kv)!
        // So BOTH GPU S1 and CPU K1 read each KV head's cell EXACTLY ONCE per token!
        constexpr double BYTES_PER_CELL = 256.0 * 2.0; // 512 B (d=256, F16)
        constexpr double BYTES_PER_BLOCK = 32.0 * BYTES_PER_CELL; // 16,384 B

        for (int il : FULL_ATTN_LAYERS) {
            const ArgusK1Stats ls = argus_k1_layer_stats(il);
            AttnProfileLayer pl{};
            pl.k_blocks_total   = ls.total_blocks;
            pl.k_blocks_skipped = ls.skipped_k_blocks;
            pl.k_skip_pct       = ls.total_blocks > 0 ? 100.0 * double(ls.skipped_k_blocks) / double(ls.total_blocks) : 0.0;
            pl.v_cells_total    = ls.total_v_cells;
            pl.v_cells_skipped  = ls.skipped_v_cells;
            pl.v_blocks_skipped = ls.skipped_v_blocks;
            pl.v_skip_pct       = ls.total_v_cells > 0 ? 100.0 * double(ls.skipped_v_cells) / double(ls.total_v_cells) : 0.0;

            const double k_blocks_read = double(ls.total_blocks - ls.skipped_k_blocks) / double(N_PROF_TOK);
            const double v_cells_read  = double(ls.total_v_cells - ls.skipped_v_cells) / double(N_PROF_TOK);
            pl.k_mib_per_tok = (k_blocks_read * BYTES_PER_BLOCK) / (1024.0 * 1024.0);
            pl.v_mib_per_tok = (v_cells_read  * BYTES_PER_CELL)  / (1024.0 * 1024.0);

            // KeyBox metadata read per token:
            // - E1 OFF: 0 B
            // - E1 ON, Multipole OFF: every block reads mu_k (512 B) + rad_k (512 B) = 1024 B
            // - E1 ON, Multipole ON:  every block reads mu_k (512 B) + rad_k (512 B), plus blocks with partial/full V skip read mu_v (512 B)
            double box_bytes_tok = 0.0;
            if (skip_eps > 0.0f) {
                const double tot_blks_tok = double(ls.total_blocks) / double(N_PROF_TOK);
                box_bytes_tok = tot_blks_tok * 1024.0;
                if (multipole) {
                    const double v_skip_blks_tok = std::max(
                        double(ls.skipped_v_blocks - ls.skipped_k_blocks) / double(N_PROF_TOK),
                        double(ls.skipped_v_cells) / (32.0 * double(N_PROF_TOK)));
                    box_bytes_tok += v_skip_blks_tok * 512.0;
                }
            }
            pl.box_mib_per_tok   = box_bytes_tok / (1024.0 * 1024.0);
            pl.total_mib_per_tok = pl.k_mib_per_tok + pl.v_mib_per_tok + pl.box_mib_per_tok;
            pl.kernel_us_per_tok = (double(ls.kernel_ns) / 1000.0) / double(N_PROF_TOK);

            res.k_mib_per_tok          += pl.k_mib_per_tok;
            res.v_mib_per_tok          += pl.v_mib_per_tok;
            res.box_mib_per_tok        += pl.box_mib_per_tok;
            res.total_kv_mib_per_tok   += pl.total_mib_per_tok;
            res.total_kernel_ms_per_tok += pl.kernel_us_per_tok / 1000.0;
            res.layers[il] = pl;
        }
        const double baseline_kv_mib = (double(tot.total_v_cells) / double(N_PROF_TOK)) * (2.0 * BYTES_PER_CELL) / (1024.0 * 1024.0);
        res.kv_bandwidth_saved_pct = baseline_kv_mib > 0.0
            ? 100.0 * (baseline_kv_mib - res.total_kv_mib_per_tok) / baseline_kv_mib : 0.0;

        std::fprintf(stderr,
                     "[Q4 %s] K-skip=%.2f%% | V-skip=%.2f%% | Read: K=%.2f + V=%.2f + Box=%.2f = %.2f MiB/tok "
                     "(Saved %.2f%%) | Attn Kernel=%.3f ms/tok | E2E Decode=%.2f tok/s\n",
                     name.c_str(), res.k_skip_pct, res.v_skip_pct,
                     res.k_mib_per_tok, res.v_mib_per_tok, res.box_mib_per_tok, res.total_kv_mib_per_tok,
                     res.kv_bandwidth_saved_pct, res.total_kernel_ms_per_tok, res.decode_tok_s);

        llama_free(ctx);
        return res;
    };

    std::fprintf(stderr, "\n=== [Step 3] Running Q4 Multipole K/V Read Volume & Kernel Time Isolation at 32K ===\n");
    std::vector<AttnProfileResult> q4_results;
    q4_results.push_back(run_q4_attn_profile("GPU S1 — E1 OFF (skip_eps=0)",                  true,  0.0f,  false));
    q4_results.push_back(run_q4_attn_profile("GPU S1 — E1 ON, Multipole OFF (skip_eps=1e-4)", true,  1e-4f, false));
    q4_results.push_back(run_q4_attn_profile("GPU S1 — E1 ON, Multipole ON  (skip_eps=1e-4)", true,  1e-4f, true));
    q4_results.push_back(run_q4_attn_profile("CPU K1 — E1 OFF (skip_eps=0)",                  false, 0.0f,  false));
    q4_results.push_back(run_q4_attn_profile("CPU K1 — E1 ON, Multipole OFF (skip_eps=1e-4)", false, 1e-4f, false));
    q4_results.push_back(run_q4_attn_profile("CPU K1 — E1 ON, Multipole ON  (skip_eps=1e-4)", false, 1e-4f, true));

    // =========================================================================
    // QUESTION 5:
    //   Standalone Decode Contribution of Graph-Split Optimizations at 32K
    //   (3 warmup + 5 measured runs of 16 tokens each, starting from saved_32k_state)
    // =========================================================================
    struct SplitAblationSpec {
        std::string name;
        bool use_argus_s1_k1;
        bool rs_gpu_opt;
        bool moe_split_opt;
    };

    struct SplitAblationResult {
        std::string name;
        bool use_argus_s1_k1 = false;
        bool rs_gpu_opt = false;
        bool moe_split_opt = false;
        int splits_pp = 0;
        int splits_tg = 0;
        double decode_median_tok_s = 0.0;
        double decode_mean_tok_s = 0.0;
        double decode_min_tok_s = 0.0;
        double decode_max_tok_s = 0.0;
        double token_p50_ms = 0.0;
        double token_p95_ms = 0.0;
    };

    const std::vector<SplitAblationSpec> q5_specs = {
        {"1. Stock llama.cpp (-nkvo baseline)",                           false, false, false},
        {"2. ARGUS S1+K1 only (RS on CPU, MoE Split Opt OFF)",            true,  false, false},
        {"3. ARGUS S1+K1 + Recurrent State on GPU (MoE Split Opt OFF)",   true,  true,  false},
        {"4. ARGUS S1+K1 + MoE Split Fusion (RS on CPU)",                 true,  false, true},
        {"5. ARGUS AUTO Full (S1+K1 + RS on GPU + MoE Split Fusion ON)",  true,  true,  true},
    };

    std::fprintf(stderr, "\n=== [Step 4] Running Q5 Graph-Split Optimization Decode Ablation at 32K ===\n");
    std::vector<SplitAblationResult> q5_results;

    for (const auto & sp : q5_specs) {
        clear_argus_env();
        if (sp.use_argus_s1_k1) {
            setenv("ARGUS_AUTO", "1", 1);
        }
        setenv("ARGUS_KV_SKIP_EPS", "0", 1);
        setenv("ARGUS_RS_GPU_OPT",    sp.rs_gpu_opt    ? "1" : "0", 1);
        setenv("ARGUS_MOE_SPLIT_OPT", sp.moe_split_opt ? "1" : "0", 1);
        argus_reset_config();

        g_last_splits_pp = 0;
        g_last_splits_tg = 0;

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = N_CTX_32K;
        cparams.n_batch         = 2048;
        cparams.n_ubatch        = 2048;
        cparams.n_seq_max       = 1;
        cparams.n_outputs_max   = 1;
        cparams.n_threads       = 4;
        cparams.n_threads_batch = 8;
        cparams.offload_kqv     = false;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { throw std::runtime_error("Failed to create Q5 context"); }

        SplitAblationResult r{};
        r.name            = sp.name;
        r.use_argus_s1_k1 = sp.use_argus_s1_k1;
        r.rs_gpu_opt      = sp.rs_gpu_opt;
        r.moe_split_opt   = sp.moe_split_opt;
        r.splits_pp       = g_last_splits_pp;
        r.splits_tg       = g_last_splits_tg;

        constexpr int Q5_WARMUP = 3;
        constexpr int Q5_MEASURED = 5;
        constexpr int Q5_TOKENS = 16;

        std::vector<double> run_tok_s;
        std::vector<double> tok_lats_ms;
        llama_batch b = llama_batch_init(1, 0, 1);

        for (int run = 0; run < Q5_WARMUP + Q5_MEASURED; ++run) {
            llama_state_set_data(ctx, saved_32k_state.data(), saved_32k_state.size());
            llama_token tok = first_tok_32k;
            const auto td0 = std::chrono::steady_clock::now();
            for (int s = 0; s < Q5_TOKENS; ++s) {
                const auto ts0 = std::chrono::steady_clock::now();
                b.n_tokens = 1;
                b.token[0] = tok;
                b.pos[0] = n_prompt + s;
                b.n_seq_id[0] = 1;
                b.seq_id[0][0] = 0;
                b.logits[0] = 1;
                llama_decode(ctx, b);
                const auto ts1 = std::chrono::steady_clock::now();
                if (run >= Q5_WARMUP) {
                    tok_lats_ms.push_back(std::chrono::duration<double, std::milli>(ts1 - ts0).count());
                }
                tok = greedy_argmax(llama_get_logits_ith(ctx, -1), n_vocab);
            }
            const auto td1 = std::chrono::steady_clock::now();
            const double ts = double(Q5_TOKENS) / std::chrono::duration<double>(td1 - td0).count();
            if (run >= Q5_WARMUP) {
                run_tok_s.push_back(ts);
            }
        }
        llama_batch_free(b);
        llama_free(ctx);

        std::sort(run_tok_s.begin(), run_tok_s.end());
        std::sort(tok_lats_ms.begin(), tok_lats_ms.end());
        r.decode_median_tok_s = percentile_sorted(run_tok_s, 50.0);
        r.decode_mean_tok_s   = std::accumulate(run_tok_s.begin(), run_tok_s.end(), 0.0) / double(run_tok_s.size());
        r.decode_min_tok_s    = run_tok_s.front();
        r.decode_max_tok_s    = run_tok_s.back();
        r.token_p50_ms        = percentile_sorted(tok_lats_ms, 50.0);
        r.token_p95_ms        = percentile_sorted(tok_lats_ms, 95.0);

        std::fprintf(stderr,
                     "[Q5] %-60s | splits(tg)=%3d | median=%.2f tok/s (mean=%.2f, p50=%.2f ms, p95=%.2f ms)\n",
                     r.name.c_str(), r.splits_tg, r.decode_median_tok_s, r.decode_mean_tok_s, r.token_p50_ms, r.token_p95_ms);
        q5_results.push_back(r);
    }

    // Write full JSON report to docs/measurements/v080-libargus-2026-10-27/bottleneck_32k_audit.json
    const char * json_path = "docs/measurements/v080-libargus-2026-10-27/bottleneck_32k_audit.json";
    std::ofstream out(json_path);
    out << "{\n";
    out << "  \"model\": \"Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf\",\n";
    out << "  \"context\": " << N_CTX_32K << ",\n";
    out << "  \"prompt_tokens\": " << n_prompt << ",\n";
    out << "  \"prefill_32k_s\": " << prefill_32k_s << ",\n";
    out << "  \"prefill_32k_tok_s\": " << (double(n_prompt) / prefill_32k_s) << ",\n";

    // Q1
    out << "  \"q1_moe_residency\": {\n";
    out << "    \"total_moe_mib\": " << res_after_prefill.total_mib << ",\n";
    out << "    \"total_moe_pages_4k\": " << res_after_prefill.total_pages << ",\n";
    out << "    \"after_load_resident_mib\": " << res_after_load.resident_mib << ",\n";
    out << "    \"after_load_resident_pct\": " << res_after_load.resident_pct << ",\n";
    out << "    \"after_32k_prefill_resident_mib\": " << res_after_prefill.resident_mib << ",\n";
    out << "    \"after_32k_prefill_resident_pct\": " << res_after_prefill.resident_pct << ",\n";
    out << "    \"after_32k_decode_resident_mib\": " << res_after_decode.resident_mib << ",\n";
    out << "    \"after_32k_decode_resident_pct\": " << res_after_decode.resident_pct << "\n";
    out << "  },\n";

    // Q2
    out << "  \"q2_per_layer_faults\": {\n";
    out << "    \"steady_state_8tok\": [\n";
    for (int il = 0; il < N_LAYERS; ++il) {
        const auto & st = steady_fault_tracker.layers[il];
        out << "      {\"layer\": " << il
            << ", \"resident_mib\": " << res_after_decode.layer_resident_mib[il]
            << ", \"resident_pct\": " << res_after_decode.layer_resident_pct[il]
            << ", \"majflt\": " << st.majflt
            << ", \"minflt\": " << st.minflt
            << ", \"disk_read_bytes\": " << st.io_read_bytes
            << ", \"elapsed_ms\": " << st.elapsed_ms << "}"
            << (il + 1 < N_LAYERS ? "," : "") << "\n";
    }
    out << "    ],\n";
    out << "    \"cold_l0_7_madv_normal_8tok\": [\n";
    for (int il = 0; il < N_LAYERS; ++il) {
        const auto & st = cold_madv_normal.per_layer[il];
        out << "      {\"layer\": " << il
            << ", \"resident_after_mib\": " << cold_madv_normal.layer_res_after_mib[il]
            << ", \"unique_routed_experts\": " << st.unique_experts.size()
            << ", \"majflt\": " << st.majflt
            << ", \"minflt\": " << st.minflt
            << ", \"disk_read_mib\": " << (double(st.io_read_bytes) / (1024.0 * 1024.0))
            << ", \"elapsed_ms\": " << st.elapsed_ms << "}"
            << (il + 1 < N_LAYERS ? "," : "") << "\n";
    }
    out << "    ],\n";
    out << "    \"cold_l0_7_madv_random_8tok\": [\n";
    for (int il = 0; il < N_LAYERS; ++il) {
        const auto & st = cold_madv_random.per_layer[il];
        out << "      {\"layer\": " << il
            << ", \"resident_after_mib\": " << cold_madv_random.layer_res_after_mib[il]
            << ", \"unique_routed_experts\": " << st.unique_experts.size()
            << ", \"majflt\": " << st.majflt
            << ", \"minflt\": " << st.minflt
            << ", \"disk_read_mib\": " << (double(st.io_read_bytes) / (1024.0 * 1024.0))
            << ", \"elapsed_ms\": " << st.elapsed_ms << "}"
            << (il + 1 < N_LAYERS ? "," : "") << "\n";
    }
    out << "    ]\n";
    out << "  },\n";

    // Q3
    auto write_madv_json = [&](const ColdMadvReport & r) {
        out << "      \"madv_random\": " << (r.madv_random ? "true" : "false") << ",\n";
        out << "      \"evicted_l0_7_mib\": " << r.cold_evicted_mib << ",\n";
        out << "      \"res_before_l0_7_mib\": " << r.res_before_l0_7_mib << ",\n";
        out << "      \"res_after_l0_7_mib\": " << r.res_after_l0_7_mib << ",\n";
        out << "      \"theoretical_routed_l0_7_mib\": " << r.theoretical_routed_l0_7_mib << ",\n";
        out << "      \"unrouted_bloat_l0_7_mib\": " << r.unrouted_bloat_l0_7_mib << ",\n";
        out << "      \"bloat_ratio\": " << r.bloat_ratio << ",\n";
        out << "      \"total_majflt\": " << r.total_majflt << ",\n";
        out << "      \"total_minflt\": " << r.total_minflt << ",\n";
        out << "      \"total_disk_read_mib\": " << r.total_disk_read_mib << ",\n";
        out << "      \"io_kib_per_majflt\": " << (r.io_bytes_per_majflt / 1024.0) << ",\n";
        out << "      \"decode_tok_s\": " << r.decode_tok_s << ",\n";
        out << "      \"token_p50_ms\": " << r.token_p50_ms << ",\n";
        out << "      \"token_p95_ms\": " << r.token_p95_ms << "\n";
    };
    out << "  \"q3_madv_random_vs_normal\": {\n";
    out << "    \"madv_normal\": {\n";
    write_madv_json(cold_madv_normal);
    out << "    },\n";
    out << "    \"madv_random\": {\n";
    write_madv_json(cold_madv_random);
    out << "    }\n";
    out << "  },\n";

    // Q4
    out << "  \"q4_multipole_kv_io_and_kernel_time\": [\n";
    for (size_t i = 0; i < q4_results.size(); ++i) {
        const auto & r = q4_results[i];
        out << "    {\n";
        out << "      \"name\": \"" << r.name << "\",\n";
        out << "      \"backend\": \"" << (r.gpu_s1 ? "GPU_S1" : "CPU_K1") << "\",\n";
        out << "      \"skip_eps\": " << r.skip_eps << ",\n";
        out << "      \"multipole\": " << (r.multipole ? "true" : "false") << ",\n";
        out << "      \"decode_tok_s\": " << r.decode_tok_s << ",\n";
        out << "      \"k_skip_pct\": " << r.k_skip_pct << ",\n";
        out << "      \"v_skip_pct\": " << r.v_skip_pct << ",\n";
        out << "      \"k_mib_per_tok\": " << r.k_mib_per_tok << ",\n";
        out << "      \"v_mib_per_tok\": " << r.v_mib_per_tok << ",\n";
        out << "      \"box_mib_per_tok\": " << r.box_mib_per_tok << ",\n";
        out << "      \"total_kv_mib_per_tok\": " << r.total_kv_mib_per_tok << ",\n";
        out << "      \"kv_bandwidth_saved_pct\": " << r.kv_bandwidth_saved_pct << ",\n";
        out << "      \"total_kernel_ms_per_tok\": " << r.total_kernel_ms_per_tok << ",\n";
        out << "      \"per_layer\": {\n";
        for (size_t li = 0; li < 10; ++li) {
            const int il = FULL_ATTN_LAYERS[li];
            const auto & pl = r.layers.at(il);
            out << "        \"" << il << "\": {"
                << "\"k_skip_pct\": " << pl.k_skip_pct
                << ", \"v_skip_pct\": " << pl.v_skip_pct
                << ", \"k_mib_tok\": " << pl.k_mib_per_tok
                << ", \"v_mib_tok\": " << pl.v_mib_per_tok
                << ", \"box_mib_tok\": " << pl.box_mib_per_tok
                << ", \"total_mib_tok\": " << pl.total_mib_per_tok
                << ", \"kernel_us_tok\": " << pl.kernel_us_per_tok << "}"
                << (li + 1 < 10 ? "," : "") << "\n";
        }
        out << "      }\n";
        out << "    }" << (i + 1 < q4_results.size() ? "," : "") << "\n";
    }
    out << "  ],\n";

    // Q5
    out << "  \"q5_graph_split_ablation\": [\n";
    for (size_t i = 0; i < q5_results.size(); ++i) {
        const auto & r = q5_results[i];
        out << "    {\n";
        out << "      \"name\": \"" << r.name << "\",\n";
        out << "      \"use_argus_s1_k1\": " << (r.use_argus_s1_k1 ? "true" : "false") << ",\n";
        out << "      \"rs_gpu_opt\": " << (r.rs_gpu_opt ? "true" : "false") << ",\n";
        out << "      \"moe_split_opt\": " << (r.moe_split_opt ? "true" : "false") << ",\n";
        out << "      \"splits_pp\": " << r.splits_pp << ",\n";
        out << "      \"splits_tg\": " << r.splits_tg << ",\n";
        out << "      \"decode_median_tok_s\": " << r.decode_median_tok_s << ",\n";
        out << "      \"decode_mean_tok_s\": " << r.decode_mean_tok_s << ",\n";
        out << "      \"decode_min_tok_s\": " << r.decode_min_tok_s << ",\n";
        out << "      \"decode_max_tok_s\": " << r.decode_max_tok_s << ",\n";
        out << "      \"token_p50_ms\": " << r.token_p50_ms << ",\n";
        out << "      \"token_p95_ms\": " << r.token_p95_ms << "\n";
        out << "    }" << (i + 1 < q5_results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();

    std::fprintf(stderr, "\n=== Saved complete 32K bottleneck audit to %s ===\n", json_path);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
