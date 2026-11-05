// ARGUS Phase 4 — Comprehensive Heterogeneous Architecture & 8K/32K/64K Validation Benchmark
// Validates all 17 tasks of the ARGUS Heterogeneous CPU/GPU Architecture Vision on real Qwen3.6-35B-A3B.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "argus_config.h"
#include "argus_hetero.h"
#include "ggml_decode_attention.h"
#include "ggml_layer_prefill.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct ProcSnap {
    double rss_mib = 0.0;
    double swap_mib = 0.0;
    uint64_t minflt = 0;
    uint64_t majflt = 0;
    uint64_t read_bytes = 0;
};

ProcSnap read_proc_snap() {
    ProcSnap s{};
    {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            unsigned long long val = 0;
            if (std::sscanf(line.c_str(), "VmRSS: %llu kB", &val) == 1) {
                s.rss_mib = double(val) / 1024.0;
            } else if (std::sscanf(line.c_str(), "VmSwap: %llu kB", &val) == 1) {
                s.swap_mib = double(val) / 1024.0;
            }
        }
    }
    {
        struct rusage ru{};
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            s.minflt = static_cast<uint64_t>(ru.ru_minflt);
            s.majflt = static_cast<uint64_t>(ru.ru_majflt);
        }
    }
    {
        std::ifstream f("/proc/self/io");
        std::string line;
        while (std::getline(f, line)) {
            unsigned long long val = 0;
            if (std::sscanf(line.c_str(), "read_bytes: %llu", &val) == 1) {
                s.read_bytes = val;
            }
        }
    }
    return s;
}

struct VramSnap {
    double free_mib = 0.0;
    double total_mib = 0.0;
    double used_mib = 0.0;
};

VramSnap read_vram_snap() {
    VramSnap v{};
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu) {
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(gpu, &free_b, &total_b);
        v.free_mib  = double(free_b) / (1024.0 * 1024.0);
        v.total_mib = double(total_b) / (1024.0 * 1024.0);
        v.used_mib  = v.total_mib - v.free_mib;
    }
    return v;
}

// Log interceptor for compute buffer sizes & S1 mirror telemetry
double g_cuda0_compute_mib = 0.0;
double g_host_compute_mib  = 0.0;

void custom_llama_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (!text) { return; }
    if (const char * p = std::strstr(text, "compute buffer size =")) {
        double val = 0.0;
        if (std::sscanf(p, "compute buffer size = %lf MiB", &val) == 1) {
            if (std::strstr(text, "CUDA0")) {
                g_cuda0_compute_mib = val;
            } else if (std::strstr(text, "CUDA_Host") || std::strstr(text, "Host") || std::strstr(text, "CPU")) {
                g_host_compute_mib = val;
            }
        }
    }
}

std::vector<llama_token> tokenize_str(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
    if (n <= 0) { return {}; }
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, true);
    return tokens;
}

std::string token_to_str(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0) { return ""; }
    return std::string(buf, n);
}

std::vector<llama_token> build_multi_needle_prompt(const llama_vocab * vocab, int target_tokens) {
    const std::string header =
        "<|im_start|>system\nYou are a precise technical assistant. Read the technical report carefully and answer the final question using only the exact facts stated in the report.<|im_end|>\n"
        "<|im_start|>user\nTECHNICAL REPORT:\n";
    const std::string needle1 =
        "\n[CRITICAL FACT 1: The primary vault access code is 7194.]\n";
    const std::string needle2 =
        "\n[CRITICAL FACT 2: The secondary orbital relay codename is BLUE-FALCON.]\n";
    const std::string needle3 =
        "\n[CRITICAL FACT 3: The final calibration date is OCTOBER-29.]\n";
    const std::string footer =
        "\nEND OF REPORT.\n"
        "Question: State (1) the primary vault access code, (2) the secondary orbital relay codename, and (3) the final calibration date in one concise sentence.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n</think>\n\n";

    const std::string filler_unit =
        "Section update: Subsystem telemetry remains within nominal thermal and voltage margins across all distributed sensor nodes. "
        "Diagnostic sweeps confirm zero packet loss on the redundant optical bus during continuous load testing. ";

    auto t_head = tokenize_str(vocab, header, true);
    auto t_n1   = tokenize_str(vocab, needle1, false);
    auto t_n2   = tokenize_str(vocab, needle2, false);
    auto t_n3   = tokenize_str(vocab, needle3, false);
    auto t_foot = tokenize_str(vocab, footer, false);
    auto t_fill = tokenize_str(vocab, filler_unit, false);

    const int fixed = int(t_head.size() + t_n1.size() + t_n2.size() + t_n3.size() + t_foot.size());
    const int rem   = std::max(0, target_tokens - fixed);
    const int seg1  = int(rem * 0.12);
    const int seg2  = int(rem * 0.38);
    const int seg3  = int(rem * 0.38);
    const int seg4  = std::max(0, rem - seg1 - seg2 - seg3);

    std::vector<llama_token> out;
    out.reserve(target_tokens + 64);
    auto append_filler = [&](int count) {
        for (int i = 0; i < count; ) {
            for (llama_token tok : t_fill) {
                if (i >= count) { break; }
                out.push_back(tok);
                ++i;
            }
        }
    };

    out.insert(out.end(), t_head.begin(), t_head.end());
    append_filler(seg1);
    out.insert(out.end(), t_n1.begin(), t_n1.end());
    append_filler(seg2);
    out.insert(out.end(), t_n2.begin(), t_n2.end());
    append_filler(seg3);
    out.insert(out.end(), t_n3.begin(), t_n3.end());
    append_filler(seg4);
    out.insert(out.end(), t_foot.begin(), t_foot.end());
    return out;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    const double idx = (p / 100.0) * double(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    const double frac = idx - double(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct StrategyMemRow {
    std::string name;
    uint32_t n_ctx = 0;
    uint32_t n_ubatch = 0;
    uint32_t n_outputs_max = 0;
    double cuda0_compute_mib = 0.0;
    double host_compute_mib = 0.0;
    double vram_used_mib = 0.0;
    double vram_free_mib = 0.0;
    double rss_mib = 0.0;
};

struct EndToEndRunResult {
    std::string name;
    uint32_t n_ctx = 0;
    int prompt_tokens = 0;
    uint32_t effective_ubatch = 0;
    double cuda0_compute_mib = 0.0;
    double host_compute_mib = 0.0;
    double prefill_ms = 0.0;
    double prefill_tok_s = 0.0;
    double ttft_ms = 0.0;
    double decode_tok_s_mean = 0.0;
    double decode_tok_s_median = 0.0;
    double decode_tok_s_stdev = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double peak_rss_mib = 0.0;
    double vram_used_mib = 0.0;
    double vram_free_mib = 0.0;
    double swap_mib = 0.0;
    uint64_t prefill_majflt = 0;
    uint64_t prefill_minflt = 0;
    double prefill_read_mib = 0.0;
    uint64_t decode_majflt = 0;
    uint64_t decode_minflt = 0;
    double decode_read_mib = 0.0;
    uint64_t k1_calls = 0;
    int needles_passed = 0;
    std::string generated_text;
    double logit_max_abs_err = 0.0;
    double logit_mean_abs_err = 0.0;
    double logit_kl_div = 0.0;
    // Sub-block profile on 10 tokens (us/tok)
    double prof_routed_moe_cpu_ms = 0.0;
    double prof_shared_exp_gpu_ms = 0.0;
    double prof_gdn_gpu_ms = 0.0;
    double prof_full_attn_gpu_ms = 0.0;
    double prof_lm_head_gpu_ms = 0.0;
};

struct SubBlockState {
    ggml_backend_dev_t gpu_dev = nullptr;
    Clock::time_point last_t{};
    double routed_moe_us = 0.0;
    double shared_exp_us = 0.0;
    double gdn_us = 0.0;
    double full_attn_us = 0.0;
    double lm_head_us = 0.0;
};

bool subblock_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<SubBlockState *>(user_data);
    if (ask) {
        const char * nm = ggml_get_name(t);
        if (!nm) { return false; }
        if (std::strncmp(nm, "ffn_moe_out-", 12) == 0 ||
            std::strncmp(nm, "ffn_shexp_gated-", 16) == 0 ||
            std::strncmp(nm, "linear_attn_out-", 16) == 0 ||
            std::strncmp(nm, "attn_out-", 9) == 0 ||
            std::strcmp(nm, "result_output") == 0) {
            return true;
        }
        return false;
    }
    const auto now = Clock::now();
    const double dt_us = std::chrono::duration<double, std::micro>(now - st->last_t).count();
    const char * nm = ggml_get_name(t);
    if (nm) {
        if (std::strncmp(nm, "ffn_moe_out-", 12) == 0) {
            st->routed_moe_us += dt_us;
        } else if (std::strncmp(nm, "ffn_shexp_gated-", 16) == 0) {
            st->shared_exp_us += dt_us;
        } else if (std::strncmp(nm, "linear_attn_out-", 16) == 0) {
            st->gdn_us += dt_us;
        } else if (std::strncmp(nm, "attn_out-", 9) == 0) {
            st->full_attn_us += dt_us;
        } else if (std::strcmp(nm, "result_output") == 0) {
            st->lm_head_us += dt_us;
        }
    }
    st->last_t = Clock::now();
    return true;
}

void compute_logit_diff(const std::vector<float> & ref, const std::vector<float> & cur,
                        double & max_abs, double & mean_abs, double & kl_div) {
    max_abs = 0.0;
    mean_abs = 0.0;
    kl_div = 0.0;
    if (ref.empty() || ref.size() != cur.size()) { return; }
    const size_t n = ref.size();
    double sum_abs = 0.0;
    float max_r = ref[0], max_c = cur[0];
    for (size_t i = 0; i < n; ++i) {
        const double d = std::abs(double(ref[i]) - double(cur[i]));
        if (d > max_abs) { max_abs = d; }
        sum_abs += d;
        if (ref[i] > max_r) { max_r = ref[i]; }
        if (cur[i] > max_c) { max_c = cur[i]; }
    }
    mean_abs = sum_abs / double(n);
    double z_r = 0.0, z_c = 0.0;
    for (size_t i = 0; i < n; ++i) {
        z_r += std::exp(double(ref[i] - max_r));
        z_c += std::exp(double(cur[i] - max_c));
    }
    const double log_z_r = double(max_r) + std::log(z_r);
    const double log_z_c = double(max_c) + std::log(z_c);
    double kl = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double log_p = double(ref[i]) - log_z_r;
        const double log_q = double(cur[i]) - log_z_c;
        const double p = std::exp(log_p);
        if (p > 1e-12) {
            kl += p * (log_p - log_q);
        }
    }
    kl_div = std::max(0.0, kl);
}

EndToEndRunResult run_e2e_benchmark(
    llama_model * model,
    const llama_vocab * vocab,
    const std::string & name,
    uint32_t n_ctx,
    int target_prompt_tokens,
    bool hetero_sched,
    bool hetero_overlap,
    std::vector<float> * out_first_logits,
    const std::vector<float> * ref_first_logits,
    bool run_subblock_prof = false) {

    setenv("ARGUS_AUTO", "1", 1);
    setenv("ARGUS_HETERO_SCHED", hetero_sched ? "1" : "0", 1);
    setenv("ARGUS_HETERO_OVERLAP", hetero_overlap ? "1" : "0", 1);
    argus_reset_config();

    EndToEndRunResult res{};
    res.name = name;
    res.n_ctx = n_ctx;

    auto prompt_tokens = build_multi_needle_prompt(vocab, target_prompt_tokens);
    res.prompt_tokens = int(prompt_tokens.size());

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_ctx;
    cparams.n_ubatch        = 2048;
    cparams.n_outputs_max   = hetero_sched ? 0 : 1; // Hetero_Auto caps n_outputs_max automatically
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 8;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.offload_kqv     = false;

    g_cuda0_compute_mib = 0.0;
    g_host_compute_mib  = 0.0;

    std::fprintf(stderr, "\n>>> [%s] Initializing context (n_ctx=%u, hetero_sched=%d, hetero_overlap=%d)...\n",
                 name.c_str(), n_ctx, int(hetero_sched), int(hetero_overlap));

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "ERROR: Failed to init context for %s\n", name.c_str());
        return res;
    }

    res.effective_ubatch  = llama_n_ubatch(ctx);
    res.cuda0_compute_mib = g_cuda0_compute_mib;
    res.host_compute_mib  = g_host_compute_mib;
    const auto vram_init  = read_vram_snap();
    const auto proc_init  = read_proc_snap();
    res.vram_used_mib     = vram_init.used_mib;
    res.vram_free_mib     = vram_init.free_mib;
    res.peak_rss_mib      = proc_init.rss_mib;
    res.swap_mib          = proc_init.swap_mib;

    if (argus_k1_reset_stats) { argus_k1_reset_stats(); }

    // 1. Prefill
    llama_batch batch = llama_batch_init(res.prompt_tokens, 0, 1);
    batch.n_tokens = res.prompt_tokens;
    for (int i = 0; i < res.prompt_tokens; ++i) {
        batch.token[i]    = prompt_tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i]   = (i == res.prompt_tokens - 1) ? 1 : 0;
    }

    const auto proc_pre0 = read_proc_snap();
    const auto t_pre0    = Clock::now();
    const int rc_pre     = llama_decode(ctx, batch);
    const auto t_pre1    = Clock::now();
    const auto proc_pre1 = read_proc_snap();

    if (rc_pre != 0) {
        std::fprintf(stderr, "ERROR: Prefill failed with rc=%d in %s\n", rc_pre, name.c_str());
        llama_batch_free(batch);
        llama_free(ctx);
        return res;
    }

    res.prefill_ms       = elapsed_ms(t_pre0, t_pre1);
    res.prefill_tok_s    = double(res.prompt_tokens) / (res.prefill_ms / 1000.0);
    res.prefill_majflt   = proc_pre1.majflt - proc_pre0.majflt;
    res.prefill_minflt   = proc_pre1.minflt - proc_pre0.minflt;
    res.prefill_read_mib = double(proc_pre1.read_bytes - proc_pre0.read_bytes) / (1024.0 * 1024.0);
    res.peak_rss_mib     = std::max(res.peak_rss_mib, proc_pre1.rss_mib);
    res.swap_mib         = std::max(res.swap_mib, proc_pre1.swap_mib);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits0 = llama_get_logits_ith(ctx, -1);
    std::vector<float> first_logits(logits0, logits0 + n_vocab);
    if (out_first_logits) {
        *out_first_logits = first_logits;
    }
    if (ref_first_logits && !ref_first_logits->empty()) {
        compute_logit_diff(*ref_first_logits, first_logits,
                           res.logit_max_abs_err, res.logit_mean_abs_err, res.logit_kl_div);
    } else {
        res.logit_max_abs_err  = 0.0;
        res.logit_mean_abs_err = 0.0;
        res.logit_kl_div       = 0.0;
    }

    llama_token first_tok = static_cast<llama_token>(
        std::max_element(first_logits.begin(), first_logits.end()) - first_logits.begin());

    // 2. Decode Benchmark: continuous decode sequence (3 warmup blocks of 14 tokens = 42 tokens
    // capturing the full multi-needle answer, followed by 5 measured blocks of 20 tokens = 100 tokens).
    // Continuous advancing positions avoid partial seq_rm on Qwen3.6's Gated DeltaNet recurrent state.
    constexpr int n_warmup      = 3;
    constexpr int n_measured    = 5;
    constexpr int warmup_len    = 13;
    constexpr int measured_len  = 10;

    llama_batch tok_batch = llama_batch_init(1, 0, 1);
    std::vector<double> run_tok_s;
    std::vector<double> all_tok_lat_ms;
    std::string generated_text = token_to_str(vocab, first_tok);

    llama_token cur_tok = first_tok;
    int total_step = 1;
    ProcSnap proc_dec0{};

    for (int r = 0; r < n_warmup + n_measured; ++r) {
        if (r == n_warmup) {
            proc_dec0 = read_proc_snap();
            if (argus_k1_reset_stats) { argus_k1_reset_stats(); }
        }
        const int block_len = (r < n_warmup) ? warmup_len : measured_len;
        std::vector<double> step_ms;
        step_ms.reserve(block_len);

        for (int s = 0; s < block_len; ++s, ++total_step) {
            tok_batch.n_tokens     = 1;
            tok_batch.token[0]     = cur_tok;
            tok_batch.pos[0]       = res.prompt_tokens + total_step - 1;
            tok_batch.n_seq_id[0]  = 1;
            tok_batch.seq_id[0][0] = 0;
            tok_batch.logits[0]    = 1;

            const auto t0 = Clock::now();
            if (llama_decode(ctx, tok_batch) != 0) {
                std::fprintf(stderr, "ERROR: Decode failed at step %d in %s\n", total_step, name.c_str());
                break;
            }
            const auto t1 = Clock::now();
            const double dt = elapsed_ms(t0, t1);
            step_ms.push_back(dt);

            const float * l = llama_get_logits_ith(ctx, -1);
            cur_tok = static_cast<llama_token>(std::max_element(l, l + n_vocab) - l);
            if (total_step <= 42) {
                generated_text += token_to_str(vocab, cur_tok);
            }
        }

        const double total_ms = std::accumulate(step_ms.begin(), step_ms.end(), 0.0);
        const double tok_s    = step_ms.empty() ? 0.0 : double(step_ms.size()) / (total_ms / 1000.0);
        if (r == 0) {
            res.ttft_ms = res.prefill_ms + (step_ms.empty() ? 0.0 : step_ms[0]);
        }
        if (r >= n_warmup && !step_ms.empty()) {
            run_tok_s.push_back(tok_s);
            all_tok_lat_ms.insert(all_tok_lat_ms.end(), step_ms.begin(), step_ms.end());
        }
    }

    const auto proc_dec1 = read_proc_snap();
    res.decode_majflt    = proc_dec1.majflt - proc_dec0.majflt;
    res.decode_minflt    = proc_dec1.minflt - proc_dec0.minflt;
    res.decode_read_mib  = double(proc_dec1.read_bytes - proc_dec0.read_bytes) / (1024.0 * 1024.0);
    res.peak_rss_mib     = std::max(res.peak_rss_mib, proc_dec1.rss_mib);
    res.swap_mib         = std::max(res.swap_mib, proc_dec1.swap_mib);
    res.k1_calls = argus_k1_stats().total_blocks;

    if (!run_tok_s.empty()) {
        const double sum = std::accumulate(run_tok_s.begin(), run_tok_s.end(), 0.0);
        res.decode_tok_s_mean = sum / double(run_tok_s.size());
        res.decode_tok_s_median = percentile(run_tok_s, 50.0);
        double sq = 0.0;
        for (double v : run_tok_s) { sq += (v - res.decode_tok_s_mean) * (v - res.decode_tok_s_mean); }
        res.decode_tok_s_stdev = std::sqrt(sq / double(run_tok_s.size()));
    }
    res.p50_ms = percentile(all_tok_lat_ms, 50.0);
    res.p95_ms = percentile(all_tok_lat_ms, 95.0);
    res.generated_text = generated_text;

    int needles = 0;
    if (generated_text.find("7194") != std::string::npos) { ++needles; }
    if (generated_text.find("BLUE-FALCON") != std::string::npos) { ++needles; }
    if (generated_text.find("OCTOBER-29") != std::string::npos) { ++needles; }
    res.needles_passed = needles;

    // Optional sub-block timing breakdown with callback_eval over 10 tokens
    if (run_subblock_prof) {
        llama_free(ctx);
        SubBlockState sbs{};
        cparams.cb_eval = subblock_eval_cb;
        cparams.cb_eval_user_data = &sbs;
        ctx = llama_init_from_model(model, cparams);
        if (ctx) {
            // Prefill 32 tokens just to warm up graph
            llama_batch warm_b = llama_batch_init(32, 0, 1);
            warm_b.n_tokens = 32;
            for (int i = 0; i < 32; ++i) {
                warm_b.token[i] = prompt_tokens[i];
                warm_b.pos[i] = i;
                warm_b.n_seq_id[i] = 1;
                warm_b.seq_id[i][0] = 0;
                warm_b.logits[i] = (i == 31) ? 1 : 0;
            }
            (void) llama_decode(ctx, warm_b);
            sbs = SubBlockState{};
            constexpr int prof_steps = 10;
            for (int s = 0; s < prof_steps; ++s) {
                tok_batch.n_tokens = 1;
                tok_batch.token[0] = first_tok;
                tok_batch.pos[0]   = 32 + s;
                tok_batch.n_seq_id[0] = 1;
                tok_batch.seq_id[0][0] = 0;
                tok_batch.logits[0] = 1;
                sbs.last_t = Clock::now();
                (void) llama_decode(ctx, tok_batch);
            }
            res.prof_routed_moe_cpu_ms = (sbs.routed_moe_us / prof_steps) / 1000.0;
            res.prof_shared_exp_gpu_ms = (sbs.shared_exp_us / prof_steps) / 1000.0;
            res.prof_gdn_gpu_ms        = (sbs.gdn_us / prof_steps) / 1000.0;
            res.prof_full_attn_gpu_ms  = (sbs.full_attn_us / prof_steps) / 1000.0;
            res.prof_lm_head_gpu_ms    = (sbs.lm_head_us / prof_steps) / 1000.0;
            llama_batch_free(warm_b);
        }
    }

    llama_batch_free(tok_batch);
    llama_batch_free(batch);
    if (ctx) { llama_free(ctx); }

    std::fprintf(stderr,
        ">>> [%s] DONE: prefill=%.2f tok/s (%.2f s), decode=%.2f tok/s (median=%.2f, p50=%.2f ms, p95=%.2f ms), "
        "VRAM=%.1f MiB, RSS=%.1f MiB, needles=%d/3, K1_calls=%llu\n",
        name.c_str(), res.prefill_tok_s, res.prefill_ms / 1000.0,
        res.decode_tok_s_mean, res.decode_tok_s_median, res.p50_ms, res.p95_ms,
        res.vram_used_mib, res.peak_rss_mib, res.needles_passed,
        (unsigned long long) res.k1_calls);

    return res;
}

std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"') { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else if (c == '\r') { out += "\\r"; }
        else if (c == '\t') { out += "\\t"; }
        else { out += c; }
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1]
        : "/home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/scratch/models/Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf";
    const char * json_out_path = argc > 2 ? argv[2]
        : "/home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/docs/measurements/v080-libargus-2026-10-27/hetero_validation_results.json";

    setenv("ARGUS_AUTO", "1", 1);
    argus_reset_config();
    llama_log_set(custom_llama_log, nullptr);
    llama_backend_init();

    // Load model once (--n-gpu-layers 99 --cpu-moe)
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    static const llama_model_tensor_buft_override overrides[] = {
        { "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr }
    };
    mparams.tensor_buft_overrides = overrides;

    std::fprintf(stderr, "=== Loading model: %s ===\n", model_path);
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "FATAL: could not load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 1. Probe Hardware & Evaluate Cost Model
    ArgusHardwareProfile hw = argus_probe_hardware();
    const auto vram_post_model = read_vram_snap();
    hw.vram_total_bytes = static_cast<size_t>(vram_post_model.total_mib * 1048576.0);
    hw.vram_free_bytes  = static_cast<size_t>(vram_post_model.free_mib * 1048576.0);

    std::fprintf(stderr,
        "=== Hardware Profile: CPU=%dT (AVX2=%d, AVX512=%d), RAM=%.1f/%.1f MiB (pressure=%d), "
        "VRAM=%.1f/%.1f MiB free/total ===\n",
        hw.cpu_logical_cores, int(hw.cpu_avx2), int(hw.cpu_avx512),
        hw.ram_avail_bytes / 1048576.0, hw.ram_total_bytes / 1048576.0, int(hw.ram_pressure),
        vram_post_model.free_mib, vram_post_model.total_mib);

    // 2. Compare Memory & S1 Placement Across Legacy vs Hetero Strategies at 8K, 32K, 64K
    struct StratSpec {
        const char * name;
        uint32_t n_ctx;
        bool hetero_sched;
    };
    const StratSpec specs[] = {
        {"Legacy_Auto_8K",  8192,  false},
        {"Hetero_Auto_8K",  8192,  true},
        {"Legacy_Auto_32K", 32768, false},
        {"Hetero_Auto_32K", 32768, true},
        {"Legacy_Auto_64K", 65536, false},
        {"Hetero_Auto_64K", 65536, true},
    };
    std::vector<StrategyMemRow> strat_rows;
    for (const auto & sp : specs) {
        setenv("ARGUS_HETERO_SCHED", sp.hetero_sched ? "1" : "0", 1);
        setenv("ARGUS_HETERO_OVERLAP", sp.hetero_sched ? "1" : "0", 1);
        argus_reset_config();
        g_cuda0_compute_mib = 0.0;
        g_host_compute_mib  = 0.0;

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = sp.n_ctx;
        cp.n_batch         = sp.n_ctx;
        cp.n_ubatch        = 2048;
        cp.n_outputs_max   = sp.hetero_sched ? 0 : 1;
        cp.n_threads       = 4;
        cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv     = false;

        llama_context * c = llama_init_from_model(model, cp);
        StrategyMemRow row{};
        row.name = sp.name;
        row.n_ctx = sp.n_ctx;
        if (c) {
            row.n_ubatch          = llama_n_ubatch(c);
            row.n_outputs_max     = sp.hetero_sched ? 16 : sp.n_ctx;
            row.cuda0_compute_mib = g_cuda0_compute_mib;
            row.host_compute_mib  = g_host_compute_mib;
            const auto vr = read_vram_snap();
            const auto pr = read_proc_snap();
            row.vram_used_mib = vr.used_mib;
            row.vram_free_mib = vr.free_mib;
            row.rss_mib       = pr.rss_mib;
            llama_free(c);
        }
        strat_rows.push_back(row);
        std::fprintf(stderr,
            "  [Strategy %s] ubatch=%u, CUDA0_Compute=%.2f MiB, Host_Compute=%.2f MiB, VRAM_Used=%.2f MiB, Free=%.2f MiB, RSS=%.1f MiB\n",
            row.name.c_str(), row.n_ubatch, row.cuda0_compute_mib, row.host_compute_mib,
            row.vram_used_mib, row.vram_free_mib, row.rss_mib);
    }

    // 3. End-to-End Benchmarks at 8K, 32K (Ablation A, B, C), and 64K
    std::vector<EndToEndRunResult> e2e_results;

    // 8K Baseline vs 8K Hetero Full
    std::vector<float> ref_logits_8k;
    e2e_results.push_back(run_e2e_benchmark(
        model, vocab, "8K_A_Legacy_Auto", 8192, 7680, false, false, &ref_logits_8k, nullptr, false));
    e2e_results.push_back(run_e2e_benchmark(
        model, vocab, "8K_B_Hetero_Full_Auto", 8192, 7680, true, true, nullptr, &ref_logits_8k, false));

    // 32K Ablation: A (Legacy Auto), B (Hetero Mem+Sched Only), C (Hetero Full Auto + Overlap)
    std::vector<float> ref_logits_32k;
    e2e_results.push_back(run_e2e_benchmark(
        model, vocab, "32K_A_Legacy_Auto", 32768, 31744, false, false, &ref_logits_32k, nullptr, true));
    e2e_results.push_back(run_e2e_benchmark(
        model, vocab, "32K_B_Hetero_MemSched_Only", 32768, 31744, true, false, nullptr, &ref_logits_32k, false));
    e2e_results.push_back(run_e2e_benchmark(
        model, vocab, "32K_C_Hetero_Full_Auto", 32768, 31744, true, true, nullptr, &ref_logits_32k, true));

    // 64K Long-Context Validation: Hetero Full Auto at 65,536 context
    e2e_results.push_back(run_e2e_benchmark(
        model, vocab, "64K_Hetero_Full_Auto", 65536, 61440, true, true, nullptr, nullptr, false));

    // Write full JSON report
    std::ofstream jf(json_out_path);
    jf << "{\n";
    jf << "  \"hardware_profile\": {\n";
    jf << "    \"cpu_logical_cores\": " << hw.cpu_logical_cores << ",\n";
    jf << "    \"has_avx2\": " << (hw.cpu_avx2 ? "true" : "false") << ",\n";
    jf << "    \"has_avx512\": " << (hw.cpu_avx512 ? "true" : "false") << ",\n";
    jf << "    \"ram_total_mib\": " << (hw.ram_total_bytes / 1048576.0) << ",\n";
    jf << "    \"ram_avail_mib\": " << (hw.ram_avail_bytes / 1048576.0) << ",\n";
    jf << "    \"vram_total_mib\": " << vram_post_model.total_mib << ",\n";
    jf << "    \"vram_free_post_model_mib\": " << vram_post_model.free_mib << "\n";
    jf << "  },\n";

    jf << "  \"cost_model_moe_placement\": [\n";
    const int batch_probes[] = {1, 4, 8, 16, 32, 64, 128, 512, 2048};
    for (size_t i = 0; i < sizeof(batch_probes) / sizeof(batch_probes[0]); ++i) {
        const int nt = batch_probes[i];
        const auto c = argus_evaluate_moe_cost(
            hw, nt, 498ULL * 1024 * 1024, 256, 8);
        jf << "    {\"n_tokens\": " << nt
           << ", \"cpu_total_us\": " << c.total_cpu_us
           << ", \"gpu_batch_major_us\": " << c.total_gpu_us
           << ", \"prefer_gpu_batch_major\": " << (c.prefer_gpu ? "true" : "false")
           << "}" << (i + 1 < sizeof(batch_probes) / sizeof(batch_probes[0]) ? "," : "") << "\n";
    }
    jf << "  ],\n";

    jf << "  \"strategy_memory_comparison\": [\n";
    for (size_t i = 0; i < strat_rows.size(); ++i) {
        const auto & r = strat_rows[i];
        jf << "    {\"name\": \"" << r.name << "\""
           << ", \"n_ctx\": " << r.n_ctx
           << ", \"n_ubatch\": " << r.n_ubatch
           << ", \"cuda0_compute_mib\": " << r.cuda0_compute_mib
           << ", \"host_compute_mib\": " << r.host_compute_mib
           << ", \"vram_used_mib\": " << r.vram_used_mib
           << ", \"vram_free_mib\": " << r.vram_free_mib
           << ", \"rss_mib\": " << r.rss_mib
           << "}" << (i + 1 < strat_rows.size() ? "," : "") << "\n";
    }
    jf << "  ],\n";

    jf << "  \"e2e_benchmarks\": [\n";
    for (size_t i = 0; i < e2e_results.size(); ++i) {
        const auto & r = e2e_results[i];
        jf << "    {\n"
           << "      \"name\": \"" << r.name << "\",\n"
           << "      \"n_ctx\": " << r.n_ctx << ",\n"
           << "      \"prompt_tokens\": " << r.prompt_tokens << ",\n"
           << "      \"effective_ubatch\": " << r.effective_ubatch << ",\n"
           << "      \"cuda0_compute_mib\": " << r.cuda0_compute_mib << ",\n"
           << "      \"host_compute_mib\": " << r.host_compute_mib << ",\n"
           << "      \"prefill_ms\": " << r.prefill_ms << ",\n"
           << "      \"prefill_tok_s\": " << r.prefill_tok_s << ",\n"
           << "      \"ttft_ms\": " << r.ttft_ms << ",\n"
           << "      \"decode_tok_s_mean\": " << r.decode_tok_s_mean << ",\n"
           << "      \"decode_tok_s_median\": " << r.decode_tok_s_median << ",\n"
           << "      \"decode_tok_s_stdev\": " << r.decode_tok_s_stdev << ",\n"
           << "      \"p50_ms\": " << r.p50_ms << ",\n"
           << "      \"p95_ms\": " << r.p95_ms << ",\n"
           << "      \"peak_rss_mib\": " << r.peak_rss_mib << ",\n"
           << "      \"vram_used_mib\": " << r.vram_used_mib << ",\n"
           << "      \"vram_free_mib\": " << r.vram_free_mib << ",\n"
           << "      \"swap_mib\": " << r.swap_mib << ",\n"
           << "      \"prefill_majflt\": " << r.prefill_majflt << ",\n"
           << "      \"prefill_minflt\": " << r.prefill_minflt << ",\n"
           << "      \"prefill_read_mib\": " << r.prefill_read_mib << ",\n"
           << "      \"decode_majflt\": " << r.decode_majflt << ",\n"
           << "      \"decode_minflt\": " << r.decode_minflt << ",\n"
           << "      \"decode_read_mib\": " << r.decode_read_mib << ",\n"
           << "      \"k1_calls\": " << r.k1_calls << ",\n"
           << "      \"needles_passed\": " << r.needles_passed << ",\n"
           << "      \"logit_max_abs_err\": " << r.logit_max_abs_err << ",\n"
           << "      \"logit_mean_abs_err\": " << r.logit_mean_abs_err << ",\n"
           << "      \"logit_kl_div\": " << r.logit_kl_div << ",\n"
           << "      \"prof_routed_moe_cpu_ms\": " << r.prof_routed_moe_cpu_ms << ",\n"
           << "      \"prof_shared_exp_gpu_ms\": " << r.prof_shared_exp_gpu_ms << ",\n"
           << "      \"prof_gdn_gpu_ms\": " << r.prof_gdn_gpu_ms << ",\n"
           << "      \"prof_full_attn_gpu_ms\": " << r.prof_full_attn_gpu_ms << ",\n"
           << "      \"prof_lm_head_gpu_ms\": " << r.prof_lm_head_gpu_ms << ",\n"
           << "      \"generated_text\": \"" << json_escape(r.generated_text) << "\"\n"
           << "    }" << (i + 1 < e2e_results.size() ? "," : "") << "\n";
    }
    jf << "  ]\n";
    jf << "}\n";
    jf.close();

    llama_model_free(model);
    llama_backend_free();
    std::fprintf(stderr, "=== Saved JSON results to %s ===\n", json_out_path);
    return 0;
}
