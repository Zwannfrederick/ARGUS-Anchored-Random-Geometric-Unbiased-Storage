// ARGUS v0.8 — Real Qwen3.6-35B-A3B SOTA-3 Ablation & Validation Harness
// Compares across 8192 and 32768 contexts (3 warmup + 5 measured runs):
//   0. Stock llama.cpp (-nkvo)
//   1. ARGUS AUTO (E1 OFF) — exact K1/S1/P2/P3 reference, all 3 new features OFF
//   2. Mode A: ARGUS AUTO (SOTA-3 OFF) — E1 base, MADV_RANDOM=0, MOE_CACHE=0, MULTIPOLE=0, TOP_R=0
//   3. Mode B: A + MADV_RANDOM
//   4. Mode C: B + Hot-Expert Cache
//   5. Mode D: C + Multipole / V-Centroid
//   6. Mode E: D + Top-r
//   7. Mode F: All ON (Low RAM Pressure)

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml_decode_attention.h"
#include "ggml_layer_prefill.h"
#include "argus_config.h"

#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifndef MADV_COLD
#define MADV_COLD 20
#endif

namespace {

constexpr int FULL_ATTN_LAYERS[10] = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};
constexpr int N_DECODE_TOKENS = 32;
constexpr int WARMUP_RUNS = 3;
constexpr int MEASURED_RUNS = 5;

struct SysSnapshot {
    double rss_mib = 0.0;
    double hwm_mib = 0.0;
    double swap_mib = 0.0;
    double lck_mib = 0.0;
    double mem_avail_mib = 0.0;
    double vram_used_mib = 0.0;
    uint64_t majflt = 0;
    uint64_t minflt = 0;
    uint64_t io_read_bytes = 0;
};

SysSnapshot read_sys_snapshot() {
    SysSnapshot s{};
    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind("VmRSS:", 0) == 0) {
                s.rss_mib = double(std::strtol(line.c_str() + 6, nullptr, 10)) / 1024.0;
            } else if (line.rfind("VmHWM:", 0) == 0) {
                s.hwm_mib = double(std::strtol(line.c_str() + 6, nullptr, 10)) / 1024.0;
            } else if (line.rfind("VmSwap:", 0) == 0) {
                s.swap_mib = double(std::strtol(line.c_str() + 7, nullptr, 10)) / 1024.0;
            } else if (line.rfind("VmLck:", 0) == 0) {
                s.lck_mib = double(std::strtol(line.c_str() + 6, nullptr, 10)) / 1024.0;
            }
        }
    }
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.rfind("MemAvailable:", 0) == 0) {
                s.mem_avail_mib = double(std::strtol(line.c_str() + 13, nullptr, 10)) / 1024.0;
                break;
            }
        }
    }
    {
        struct rusage ru{};
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            s.majflt = static_cast<uint64_t>(ru.ru_majflt);
            s.minflt = static_cast<uint64_t>(ru.ru_minflt);
        }
    }
    {
        std::ifstream io("/proc/self/io");
        std::string line;
        while (std::getline(io, line)) {
            if (line.rfind("read_bytes:", 0) == 0) {
                s.io_read_bytes = std::strtoull(line.c_str() + 11, nullptr, 10);
                break;
            }
        }
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        s.vram_used_mib = double(total_b - free_b) / (1024.0 * 1024.0);
    }
    return s;
}

double percentile_sorted(const std::vector<double> & sorted, double p) {
    if (sorted.empty()) { return 0.0; }
    const double idx = (p / 100.0) * double(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    const double frac = idx - double(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

std::vector<llama_token> tokenize_str(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n = llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
    if (n < 0) { n = -n; }
    std::vector<llama_token> tokens(n);
    int actual = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, true);
    if (actual < 0) { throw std::runtime_error("Tokenization failed"); }
    tokens.resize(actual);
    return tokens;
}

std::string detokenize_tokens(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string out;
    for (llama_token t : tokens) {
        char buf[256];
        int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (n > 0) { out.append(buf, n); }
    }
    return out;
}

std::vector<llama_token> build_multi_needle_prompt(const llama_vocab * vocab, int target_prompt_tokens) {
    const std::string prefix =
        "<|im_start|>system\nYou are a precise technical assistant. Answer concisely with the exact requested numbers.<|im_end|>\n"
        "<|im_start|>user\nRead the following operational log carefully and answer the question at the end.\n\n";
    const std::string needle1 =
        "\n[CRITICAL RECORD 1: The vault access code for Sector Alpha is 657446.]\n";
    const std::string needle2 =
        "\n[CRITICAL RECORD 2: The override frequency for Station Beta is 839204.]\n";
    const std::string needle3 =
        "\n[CRITICAL RECORD 3: The calibration serial for Unit Gamma is 412975.]\n";
    const std::string suffix =
        "\n\nQuestion: What is the 6-digit vault access code for Sector Alpha, the 6-digit override frequency for Station Beta, "
        "and the 6-digit calibration serial for Unit Gamma? State the three 6-digit numbers in order.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nSector Alpha: ";

    const std::string filler_sentence =
        "Engineers measured the suspension cables every spring after the mountain thaw had passed, while farmers in the lower valley "
        "rotated barley, clover, and lentils to keep the alluvial soil productive across dry summers. ";

    auto prefix_toks = tokenize_str(vocab, prefix, false);
    auto n1_toks     = tokenize_str(vocab, needle1, false);
    auto n2_toks     = tokenize_str(vocab, needle2, false);
    auto n3_toks     = tokenize_str(vocab, needle3, false);
    auto suffix_toks = tokenize_str(vocab, suffix, false);
    auto fill_toks   = tokenize_str(vocab, filler_sentence, false);

    const int fixed_len = int(prefix_toks.size() + n1_toks.size() + n2_toks.size() + n3_toks.size() + suffix_toks.size());
    const int needed_filler = std::max(64, target_prompt_tokens - fixed_len);

    std::vector<llama_token> filler_pool;
    filler_pool.reserve(needed_filler + fill_toks.size());
    while (int(filler_pool.size()) < needed_filler) {
        filler_pool.insert(filler_pool.end(), fill_toks.begin(), fill_toks.end());
    }
    filler_pool.resize(needed_filler);

    const int pos1 = int(0.10 * needed_filler);
    const int pos2 = int(0.50 * needed_filler);
    const int pos3 = int(0.85 * needed_filler);

    std::vector<llama_token> prompt;
    prompt.reserve(target_prompt_tokens);
    prompt.insert(prompt.end(), prefix_toks.begin(), prefix_toks.end());
    prompt.insert(prompt.end(), filler_pool.begin(), filler_pool.begin() + pos1);
    prompt.insert(prompt.end(), n1_toks.begin(), n1_toks.end());
    prompt.insert(prompt.end(), filler_pool.begin() + pos1, filler_pool.begin() + pos2);
    prompt.insert(prompt.end(), n2_toks.begin(), n2_toks.end());
    prompt.insert(prompt.end(), filler_pool.begin() + pos2, filler_pool.begin() + pos3);
    prompt.insert(prompt.end(), n3_toks.begin(), n3_toks.end());
    prompt.insert(prompt.end(), filler_pool.begin() + pos3, filler_pool.end());
    prompt.insert(prompt.end(), suffix_toks.begin(), suffix_toks.end());
    return prompt;
}

struct EvalCapture {
    bool active = false;
    std::map<int, std::vector<float>> kqv_out;
};

bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = static_cast<EvalCapture *>(user_data);
    if (!cap || !cap->active || !t || !t->name[0]) { return !ask; }
    int il = -1;
    if (std::strncmp(t->name, "kqv_out-", 8) == 0) {
        il = std::atoi(t->name + 8);
    } else if (std::strncmp(t->name, "attn_pregate-", 13) == 0) {
        il = std::atoi(t->name + 13);
    } else {
        return !ask;
    }
    if (ask) { return true; }
    const size_t n_floats = ggml_nelements(t);
    std::vector<float> buf(n_floats);
    ggml_backend_tensor_get(t, buf.data(), 0, n_floats * sizeof(float));
    cap->kqv_out[il] = std::move(buf);
    return true;
}

struct LayerError {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double l2_rel = 0.0;
};

LayerError compute_tensor_error(const std::vector<float> & ref, const std::vector<float> & cur) {
    if (ref.empty() || ref.size() != cur.size()) { return {}; }
    double max_abs = 0.0, sum_abs = 0.0, diff_sq = 0.0, ref_sq = 1e-30;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(double(cur[i]) - double(ref[i]));
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        diff_sq += d * d;
        ref_sq  += double(ref[i]) * double(ref[i]);
    }
    return {max_abs, sum_abs / double(ref.size()), std::sqrt(diff_sq / ref_sq)};
}

struct LogitMetrics {
    double max_abs_err = 0.0;
    double mean_abs_err = 0.0;
    double mean_kl_div = 0.0;
    double max_kl_div = 0.0;
};

LogitMetrics compute_logit_metrics(const std::vector<std::vector<float>> & ref_steps,
                                   const std::vector<std::vector<float>> & cur_steps) {
    LogitMetrics m{};
    if (ref_steps.empty() || ref_steps.size() != cur_steps.size()) { return m; }
    const size_t T = ref_steps.size(), V = ref_steps[0].size();
    double total_abs = 0.0, total_kl = 0.0;
    for (size_t t = 0; t < T; ++t) {
        const auto & r = ref_steps[t];
        const auto & c = cur_steps[t];
        double r_max = -1e30, c_max = -1e30;
        for (size_t v = 0; v < V; ++v) {
            const double diff = std::fabs(double(c[v]) - double(r[v]));
            m.max_abs_err = std::max(m.max_abs_err, diff);
            total_abs += diff;
            r_max = std::max(r_max, double(r[v]));
            c_max = std::max(c_max, double(c[v]));
        }
        double r_sum = 0.0, c_sum = 0.0;
        for (size_t v = 0; v < V; ++v) {
            r_sum += std::exp(double(r[v]) - r_max);
            c_sum += std::exp(double(c[v]) - c_max);
        }
        const double log_zr = r_max + std::log(r_sum);
        const double log_zc = c_max + std::log(c_sum);
        double step_kl = 0.0;
        for (size_t v = 0; v < V; ++v) {
            const double log_pr = double(r[v]) - log_zr;
            const double log_pc = double(c[v]) - log_zc;
            const double pr = std::exp(log_pr);
            if (pr > 1e-30) {
                step_kl += pr * (log_pr - log_pc);
            }
        }
        step_kl = std::max(0.0, step_kl);
        m.max_kl_div = std::max(m.max_kl_div, step_kl);
        total_kl += step_kl;
    }
    m.mean_abs_err = total_abs / double(T * V);
    m.mean_kl_div  = total_kl / double(T);
    return m;
}

struct ModeSpec {
    std::string name;
    bool use_argus_auto;
    bool use_k1;
    bool use_p2;
    bool use_p3;
    float skip_eps;
    bool madv_random;
    bool moe_cache;
    bool multipole;
    bool top_r;
    bool low_ram_pressure;
    bool reuse_prefill;
};

struct ModeResult {
    std::string name;
    int context_len = 0;
    int prompt_tokens = 0;
    double prefill_s = 0.0;
    double prefill_tok_s = 0.0;
    double ttft_s = 0.0;
    std::vector<double> decode_tok_s_runs;
    double decode_median_tok_s = 0.0;
    double decode_mean_tok_s = 0.0;
    double decode_min_tok_s = 0.0;
    double decode_max_tok_s = 0.0;
    double token_lat_p50_ms = 0.0;
    double token_lat_p95_ms = 0.0;
    SysSnapshot snap_after;
    uint64_t majflt_delta = 0;
    uint64_t minflt_delta = 0;
    double disk_read_mib = 0.0;
    double swap_delta_mib = 0.0;
    double pressure_buf_mib = 0.0;
    bool ram_swap_bottleneck = false;
    ArgusMoeStats moe_stats{};
    std::map<int, ArgusK1Stats> layer_stats;
    ArgusK1Stats total_stats;
    std::map<int, LayerError> layer_attn_err_stock;
    std::map<int, LayerError> layer_attn_err_e1off;
    LogitMetrics logit_err_stock;
    LogitMetrics logit_err_e1off;
    std::string generated_text;
    bool needle1_found = false;
    bool needle2_found = false;
    bool needle3_found = false;
    int needles_found_count = 0;
};

void apply_mode_env(const ModeSpec & spec) {
    argus_reset_config();
    unsetenv("ARGUS_AUTO");
    unsetenv("ARGUS_KV_CPU_ATTENTION");
    unsetenv("ARGUS_KV_VRAM_BYTES");
    unsetenv("ARGUS_KV_PREFILL");
    unsetenv("ARGUS_PREFILL");
    unsetenv("ARGUS_KV_SKIP_EPS");
    unsetenv("ARGUS_E1");
    unsetenv("ARGUS_MADV_RANDOM");
    unsetenv("ARGUS_MOE_CACHE");
    unsetenv("ARGUS_MULTIPOLE");
    unsetenv("ARGUS_TOP_R");

    if (spec.use_argus_auto) {
        setenv("ARGUS_AUTO", "1", 1);
    } else {
        if (spec.use_k1) { setenv("ARGUS_KV_CPU_ATTENTION", "k1", 1); }
        if (spec.use_p2) { setenv("ARGUS_KV_PREFILL", "gpu", 1); }
        if (spec.use_p3) { setenv("ARGUS_PREFILL", "layers", 1); }
    }
    if (spec.skip_eps > 0.0f) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", spec.skip_eps);
        setenv("ARGUS_KV_SKIP_EPS", buf, 1);
    } else {
        setenv("ARGUS_KV_SKIP_EPS", "0", 1);
    }
    setenv("ARGUS_MADV_RANDOM", spec.madv_random ? "1" : "0", 1);
    setenv("ARGUS_MOE_CACHE",   spec.moe_cache   ? "1" : "0", 1);
    setenv("ARGUS_MULTIPOLE",   spec.multipole   ? "1" : "0", 1);
    setenv("ARGUS_TOP_R",       spec.top_r       ? "1" : "0", 1);
}

void run_prefill(llama_context * ctx, const std::vector<llama_token> & prompt, int ubatch_size, bool layer_major) {
    const int n_tokens = int(prompt.size());
    const int step = layer_major ? n_tokens : ubatch_size;
    llama_batch batch = llama_batch_init(step, 0, 1);
    for (int i = 0; i < n_tokens; i += step) {
        const int cur = std::min(step, n_tokens - i);
        batch.n_tokens = cur;
        for (int j = 0; j < cur; ++j) {
            batch.token[j]     = prompt[i + j];
            batch.pos[j]       = i + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = (i + j == n_tokens - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed during prefill");
        }
    }
    llama_batch_free(batch);
}

llama_token greedy_argmax(const float * logits, int n_vocab) {
    return static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "scratch/models/Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf";
    std::vector<int> contexts = {8192, 32768};
    if (argc > 2) {
        contexts.clear();
        for (int i = 2; i < argc; ++i) { contexts.push_back(std::atoi(argv[i])); }
    }

    llama_backend_init();

    static const llama_model_tensor_buft_override cpu_moe_overrides[] = {
        { "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr }
    };

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.tensor_buft_overrides = cpu_moe_overrides;

    std::fprintf(stderr, "=== Loading model: %s ===\n", model_path);
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "Failed to load model %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Warm up OS page cache for mmap'd MoE expert weights once before timed runs.
    {
        llama_context_params wparams = llama_context_default_params();
        wparams.n_ctx = 256;
        wparams.n_batch = 64;
        wparams.n_ubatch = 64;
        wparams.n_threads = 4;
        wparams.n_threads_batch = 8;
        wparams.offload_kqv = false;
        llama_context * wctx = llama_init_from_model(model, wparams);
        if (wctx) {
            std::vector<llama_token> wtoks(64, 100);
            run_prefill(wctx, wtoks, 64, false);
            llama_free(wctx);
        }
    }

    const std::vector<ModeSpec> modes = {
        // name                              auto   k1     p2     p3     eps    madv   moe    mp     top_r  lowram reuse_pf
        {"Stock llama.cpp",                  false, false, false, false, 0.0f,  false, false, false, false, false, false},
        {"ARGUS AUTO (E1 OFF)",              true,  true,  true,  true,  0.0f,  false, false, false, false, false, false},
        {"A: ARGUS AUTO (SOTA-3 OFF)",       true,  true,  true,  true,  1e-4f, false, false, false, false, false, true},
        {"B: A + MADV_RANDOM",               true,  true,  true,  true,  1e-4f, true,  false, false, false, false, true},
        {"C: B + Hot-Expert Cache",          true,  true,  true,  true,  1e-4f, true,  true,  false, false, false, true},
        {"D: C + Multipole / V-Centroid",    true,  true,  true,  true,  1e-4f, true,  true,  true,  false, false, true},
        {"E: D + Top-r",                     true,  true,  true,  true,  1e-4f, true,  true,  true,  true,  false, true},
        {"F: All ON (Low RAM Pressure)",     true,  true,  true,  true,  1e-4f, true,  true,  true,  true,  true,  true},
    };

    std::vector<ModeResult> all_results;

    for (int n_ctx : contexts) {
        const int target_prompt = n_ctx - 128;
        const auto prompt = build_multi_needle_prompt(vocab, target_prompt);
        const int n_prompt = int(prompt.size());
        std::fprintf(stderr, "\n========================================\n");
        std::fprintf(stderr, "Context: %d (Prompt tokens: %d, Decode tokens: %d)\n", n_ctx, n_prompt, N_DECODE_TOKENS);
        std::fprintf(stderr, "========================================\n");

        std::vector<uint8_t> saved_prefill_state;
        llama_token first_token_from_prefill = 0;
        double prev_prefill_s = 0.0;
        std::map<int, std::vector<float>> stock_attn_out;
        std::vector<std::vector<float>> stock_logits;
        std::map<int, std::vector<float>> e1off_attn_out;
        std::vector<std::vector<float>> e1off_logits;

        for (size_t m_idx = 0; m_idx < modes.size(); ++m_idx) {
            const auto & spec = modes[m_idx];
            apply_mode_env(spec);
            std::fprintf(stderr, "\n--- Running Mode [%zu/%zu]: %s (ctx=%d) ---\n",
                         m_idx + 1, modes.size(), spec.name.c_str(), n_ctx);

            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx           = n_ctx;
            cparams.n_batch         = spec.use_p3 ? n_ctx : 2048;
            cparams.n_ubatch        = 2048;
            cparams.n_seq_max       = 1;
            cparams.n_outputs_max   = 1;
            cparams.n_threads       = 4;
            cparams.n_threads_batch = 8;
            cparams.offload_kqv     = false; // -nkvo
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cparams.cb_eval         = nullptr;
            cparams.cb_eval_user_data = nullptr;

            llama_context * ctx = llama_init_from_model(model, cparams);
            if (!ctx) { throw std::runtime_error("Failed to create llama_context"); }

            ModeResult res;
            res.name = spec.name;
            res.context_len = n_ctx;
            res.prompt_tokens = n_prompt;

            // 1. Measure Prefill + TTFT
            if (spec.reuse_prefill && !saved_prefill_state.empty()) {
                res.prefill_s = prev_prefill_s;
                res.prefill_tok_s = double(n_prompt) / res.prefill_s;
            } else {
                const auto t_pf0 = std::chrono::steady_clock::now();
                run_prefill(ctx, prompt, 2048, spec.use_p3);
                const auto t_pf1 = std::chrono::steady_clock::now();
                res.prefill_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();
                res.prefill_tok_s = double(n_prompt) / res.prefill_s;
                prev_prefill_s = res.prefill_s;

                const float * pf_logits = llama_get_logits_ith(ctx, -1);
                const llama_token first_tok = greedy_argmax(pf_logits, n_vocab);

                if (saved_prefill_state.empty()) {
                    const size_t state_sz = llama_state_get_size(ctx);
                    saved_prefill_state.resize(state_sz);
                    const size_t written = llama_state_get_data(ctx, saved_prefill_state.data(), saved_prefill_state.size());
                    if (written == 0) {
                        throw std::runtime_error("llama_state_get_data failed");
                    }
                    saved_prefill_state.resize(written);
                    first_token_from_prefill = first_tok;
                }
            }

            // Optional Low RAM Pressure simulation for Mode F:
            // Allocate a bounded anonymous pressure buffer leaving a safe >= 3.5 GiB MemAvailable margin.
            void * pressure_ptr = nullptr;
            size_t pressure_bytes = 0;
            if (spec.low_ram_pressure) {
                const SysSnapshot pre_p = read_sys_snapshot();
                if (pre_p.mem_avail_mib > 4096.0) {
                    const double target_mib = std::min(2048.0, pre_p.mem_avail_mib - 3584.0);
                    if (target_mib >= 256.0) {
                        pressure_bytes = static_cast<size_t>(target_mib) << 20;
                        pressure_ptr = mmap(nullptr, pressure_bytes, PROT_READ | PROT_WRITE,
                                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                        if (pressure_ptr != MAP_FAILED) {
                            std::memset(pressure_ptr, 0x5A, pressure_bytes);
                            res.pressure_buf_mib = double(pressure_bytes) / (1024.0 * 1024.0);
                            std::fprintf(stderr, "  [Mode F Low-RAM Pressure] Allocated %.0f MiB pressure buffer (MemAvail was %.0f MiB)\n",
                                         res.pressure_buf_mib, pre_p.mem_avail_mib);
                        } else {
                            pressure_ptr = nullptr;
                            pressure_bytes = 0;
                        }
                    }
                }
            }

            // 2. Timed Decode Benchmark: 3 Warmup + 5 Measured runs
            std::vector<std::vector<float>> mode_logits(N_DECODE_TOKENS, std::vector<float>(n_vocab));
            std::vector<llama_token> generated_tokens;
            std::vector<double> measured_token_lats_ms;
            measured_token_lats_ms.reserve(MEASURED_RUNS * N_DECODE_TOKENS);
            llama_batch dec_batch = llama_batch_init(1, 0, 1);

            if (argus_moe_reset_stats) { argus_moe_reset_stats(); }
            SysSnapshot snap_before_measured{};

            for (int run = 0; run < WARMUP_RUNS + MEASURED_RUNS; ++run) {
                if (run > 0 || m_idx > 0) {
                    if (llama_state_set_data(ctx, saved_prefill_state.data(), saved_prefill_state.size()) == 0) {
                        throw std::runtime_error("llama_state_set_data failed before timed run");
                    }
                }
                if (run == WARMUP_RUNS) {
                    argus_k1_reset_stats();
                    snap_before_measured = read_sys_snapshot();
                }
                if (run == 0) {
                    generated_tokens.clear();
                    generated_tokens.push_back(first_token_from_prefill);
                }
                llama_token tok = first_token_from_prefill;
                const auto t_d0 = std::chrono::steady_clock::now();
                for (int step = 0; step < N_DECODE_TOKENS; ++step) {
                    const auto t_s0 = std::chrono::steady_clock::now();
                    dec_batch.n_tokens = 1;
                    dec_batch.token[0]     = tok;
                    dec_batch.pos[0]       = n_prompt + step;
                    dec_batch.n_seq_id[0]  = 1;
                    dec_batch.seq_id[0][0] = 0;
                    dec_batch.logits[0]    = 1;
                    if (llama_decode(ctx, dec_batch) != 0) {
                        llama_batch_free(dec_batch);
                        throw std::runtime_error("llama_decode failed during timed decode");
                    }
                    const auto t_s1 = std::chrono::steady_clock::now();
                    if (run >= WARMUP_RUNS) {
                        measured_token_lats_ms.push_back(std::chrono::duration<double, std::milli>(t_s1 - t_s0).count());
                    }
                    const float * l_ptr = llama_get_logits_ith(ctx, -1);
                    if (run == 0) {
                        std::memcpy(mode_logits[step].data(), l_ptr, n_vocab * sizeof(float));
                    }
                    tok = greedy_argmax(l_ptr, n_vocab);
                    if (run == 0) {
                        generated_tokens.push_back(tok);
                    }
                }
                const auto t_d1 = std::chrono::steady_clock::now();
                const double dec_s = std::chrono::duration<double>(t_d1 - t_d0).count();
                const double tok_s = double(N_DECODE_TOKENS) / dec_s;
                if (run >= WARMUP_RUNS) {
                    res.decode_tok_s_runs.push_back(tok_s);
                }
                std::fprintf(stderr, "  [%s] run %d/%d (%s): %.2f tok/s (%.3f s)\n",
                             spec.name.c_str(), run + 1, WARMUP_RUNS + MEASURED_RUNS,
                             run < WARMUP_RUNS ? "warmup" : "measured", tok_s, dec_s);
            }

            cudaDeviceSynchronize();
            res.snap_after = read_sys_snapshot();
            res.majflt_delta = res.snap_after.majflt >= snap_before_measured.majflt
                ? (res.snap_after.majflt - snap_before_measured.majflt) : 0;
            res.minflt_delta = res.snap_after.minflt >= snap_before_measured.minflt
                ? (res.snap_after.minflt - snap_before_measured.minflt) : 0;
            res.disk_read_mib = res.snap_after.io_read_bytes >= snap_before_measured.io_read_bytes
                ? double(res.snap_after.io_read_bytes - snap_before_measured.io_read_bytes) / (1024.0 * 1024.0) : 0.0;
            res.swap_delta_mib = std::max(0.0, res.snap_after.swap_mib - snap_before_measured.swap_mib);
            res.ram_swap_bottleneck = (res.swap_delta_mib > 16.0 || res.majflt_delta > 50 || res.disk_read_mib > 64.0);
            res.moe_stats = argus_moe_stats ? argus_moe_stats() : ArgusMoeStats{};

            if (pressure_ptr && pressure_bytes > 0) {
                munmap(pressure_ptr, pressure_bytes);
                pressure_ptr = nullptr;
            }

            res.total_stats = argus_k1_stats();
            for (int il : FULL_ATTN_LAYERS) {
                res.layer_stats[il] = argus_k1_layer_stats(il);
            }
            llama_free(ctx);

            // 3. 1-step Diagnostic pass with cb_eval to capture exact per-layer kqv_out-%d
            EvalCapture cap;
            cap.active = true;
            llama_context_params dparams = cparams;
            dparams.n_batch   = 32;
            dparams.n_ubatch  = 32;
            dparams.cb_eval   = eval_callback;
            dparams.cb_eval_user_data = &cap;
            llama_context * ctx_diag = llama_init_from_model(model, dparams);
            if (!ctx_diag) { throw std::runtime_error("Failed to create diagnostic llama_context"); }
            if (llama_state_set_data(ctx_diag, saved_prefill_state.data(), saved_prefill_state.size()) == 0) {
                throw std::runtime_error("llama_state_set_data failed before diagnostic step");
            }
            dec_batch.n_tokens = 1;
            dec_batch.token[0]     = first_token_from_prefill;
            dec_batch.pos[0]       = n_prompt;
            dec_batch.n_seq_id[0]  = 1;
            dec_batch.seq_id[0][0] = 0;
            dec_batch.logits[0]    = 1;
            // If Multipole/Top-r is active, run 1 warmup decode on ctx_diag so boxes are populated before capturing Pass 2!
            if (spec.skip_eps > 0.0f && (spec.multipole || spec.top_r)) {
                cap.active = false;
                (void) llama_decode(ctx_diag, dec_batch);
                (void) llama_state_set_data(ctx_diag, saved_prefill_state.data(), saved_prefill_state.size());
                cap.active = true;
            }
            if (llama_decode(ctx_diag, dec_batch) != 0) {
                llama_batch_free(dec_batch);
                throw std::runtime_error("llama_decode failed during 1-step diagnostic pass");
            }
            llama_batch_free(dec_batch);
            llama_free(ctx_diag);

            if (m_idx == 0) {
                stock_attn_out = cap.kqv_out;
                stock_logits   = mode_logits;
            }
            if (m_idx == 1) {
                e1off_attn_out = cap.kqv_out;
                e1off_logits   = mode_logits;
            }
            for (int il : FULL_ATTN_LAYERS) {
                res.layer_attn_err_stock[il] = compute_tensor_error(stock_attn_out[il], cap.kqv_out[il]);
                res.layer_attn_err_e1off[il] = e1off_attn_out.empty()
                    ? LayerError{}
                    : compute_tensor_error(e1off_attn_out[il], cap.kqv_out[il]);
            }
            res.logit_err_stock = compute_logit_metrics(stock_logits, mode_logits);
            res.logit_err_e1off = e1off_logits.empty()
                ? LogitMetrics{}
                : compute_logit_metrics(e1off_logits, mode_logits);

            res.generated_text = detokenize_tokens(vocab, generated_tokens);
            res.needle1_found = res.generated_text.find("657446") != std::string::npos;
            res.needle2_found = res.generated_text.find("839204") != std::string::npos;
            res.needle3_found = res.generated_text.find("412975") != std::string::npos;
            res.needles_found_count = int(res.needle1_found) + int(res.needle2_found) + int(res.needle3_found);

            auto sorted_runs = res.decode_tok_s_runs;
            std::sort(sorted_runs.begin(), sorted_runs.end());
            res.decode_min_tok_s    = sorted_runs.front();
            res.decode_max_tok_s    = sorted_runs.back();
            res.decode_median_tok_s = sorted_runs[sorted_runs.size() / 2];
            res.decode_mean_tok_s   = std::accumulate(sorted_runs.begin(), sorted_runs.end(), 0.0) / double(sorted_runs.size());
            res.ttft_s              = res.prefill_s + (1.0 / res.decode_median_tok_s);

            std::sort(measured_token_lats_ms.begin(), measured_token_lats_ms.end());
            res.token_lat_p50_ms = percentile_sorted(measured_token_lats_ms, 50.0);
            res.token_lat_p95_ms = percentile_sorted(measured_token_lats_ms, 95.0);

            const double moe_hit_pct = res.moe_stats.total_routed
                ? 100.0 * double(res.moe_stats.hot_hits) / double(res.moe_stats.total_routed) : 0.0;

            std::fprintf(stderr,
                         "  => Prefill: %.2f s (%.1f tok/s) | TTFT: %.2f s | Decode median: %.2f tok/s (mean %.2f, min %.2f, max %.2f)\n"
                         "  => Latency p50: %.2f ms | p95: %.2f ms | Bottleneck: %s (majflt=%llu, disk_read=%.1f MiB, swap_d=%.1f MiB)\n"
                         "  => MoE Cache: hits=%llu/%llu (%.2f%%), promo=%llu, demo=%llu, locked=%.2f MiB (VmLck=%.2f MiB)\n"
                         "  => K-skip: %llu / %llu (%.2f%%) | V-cell skip: %llu / %llu (%.2f%%)\n"
                         "  => vs Stock  Logit max: %.3e, mean: %.3e | KL mean: %.3e, max: %.3e\n"
                         "  => vs E1-OFF Logit max: %.3e, mean: %.3e | KL mean: %.3e, max: %.3e\n"
                         "  => Needles (%d/3): [%s]\n",
                         res.prefill_s, res.prefill_tok_s, res.ttft_s,
                         res.decode_median_tok_s, res.decode_mean_tok_s, res.decode_min_tok_s, res.decode_max_tok_s,
                         res.token_lat_p50_ms, res.token_lat_p95_ms,
                         res.ram_swap_bottleneck ? "YES (INVALID)" : "NO (VALID)",
                         (unsigned long long) res.majflt_delta, res.disk_read_mib, res.swap_delta_mib,
                         (unsigned long long) res.moe_stats.hot_hits, (unsigned long long) res.moe_stats.total_routed, moe_hit_pct,
                         (unsigned long long) res.moe_stats.promotions, (unsigned long long) res.moe_stats.demotions,
                         double(res.moe_stats.locked_bytes) / (1024.0 * 1024.0), res.snap_after.lck_mib,
                         (unsigned long long) res.total_stats.skipped_k_blocks,
                         (unsigned long long) res.total_stats.total_blocks,
                         res.total_stats.total_blocks ? 100.0 * double(res.total_stats.skipped_k_blocks) / double(res.total_stats.total_blocks) : 0.0,
                         (unsigned long long) res.total_stats.skipped_v_cells,
                         (unsigned long long) res.total_stats.total_v_cells,
                         res.total_stats.total_v_cells ? 100.0 * double(res.total_stats.skipped_v_cells) / double(res.total_stats.total_v_cells) : 0.0,
                         res.logit_err_stock.max_abs_err, res.logit_err_stock.mean_abs_err,
                         res.logit_err_stock.mean_kl_div, res.logit_err_stock.max_kl_div,
                         res.logit_err_e1off.max_abs_err, res.logit_err_e1off.mean_abs_err,
                         res.logit_err_e1off.mean_kl_div, res.logit_err_e1off.max_kl_div,
                         res.needles_found_count, res.generated_text.c_str());

            all_results.push_back(res);

            // Incrementally write JSON report after each mode completes
            {
                std::ofstream out("docs/measurements/v080-libargus-2026-10-27/real_a3b_validation.json");
                out << "{\n  \"model\": \"Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf\",\n"
                    << "  \"seed\": 2026,\n"
                    << "  \"sampling\": \"deterministic_greedy\",\n"
                    << "  \"warmup_runs\": " << WARMUP_RUNS << ",\n"
                    << "  \"measured_runs\": " << MEASURED_RUNS << ",\n"
                    << "  \"decode_tokens\": " << N_DECODE_TOKENS << ",\n"
                    << "  \"results\": [\n";
                for (size_t i = 0; i < all_results.size(); ++i) {
                    const auto & r = all_results[i];
                    const double k_skip_pct = r.total_stats.total_blocks
                        ? 100.0 * double(r.total_stats.skipped_k_blocks) / double(r.total_stats.total_blocks) : 0.0;
                    const double v_skip_pct = r.total_stats.total_v_cells
                        ? 100.0 * double(r.total_stats.skipped_v_cells) / double(r.total_stats.total_v_cells) : 0.0;
                    const double hit_pct = r.moe_stats.total_routed
                        ? 100.0 * double(r.moe_stats.hot_hits) / double(r.moe_stats.total_routed) : 0.0;
                    out << "    {\n"
                        << "      \"mode\": \"" << r.name << "\",\n"
                        << "      \"context\": " << r.context_len << ",\n"
                        << "      \"prompt_tokens\": " << r.prompt_tokens << ",\n"
                        << "      \"prefill_s\": " << r.prefill_s << ",\n"
                        << "      \"prefill_tok_s\": " << r.prefill_tok_s << ",\n"
                        << "      \"ttft_s\": " << r.ttft_s << ",\n"
                        << "      \"decode_median_tok_s\": " << r.decode_median_tok_s << ",\n"
                        << "      \"decode_mean_tok_s\": " << r.decode_mean_tok_s << ",\n"
                        << "      \"decode_min_tok_s\": " << r.decode_min_tok_s << ",\n"
                        << "      \"decode_max_tok_s\": " << r.decode_max_tok_s << ",\n"
                        << "      \"token_lat_p50_ms\": " << r.token_lat_p50_ms << ",\n"
                        << "      \"token_lat_p95_ms\": " << r.token_lat_p95_ms << ",\n"
                        << "      \"rss_mib\": " << r.snap_after.rss_mib << ",\n"
                        << "      \"hwm_mib\": " << r.snap_after.hwm_mib << ",\n"
                        << "      \"vram_used_mib\": " << r.snap_after.vram_used_mib << ",\n"
                        << "      \"swap_mib\": " << r.snap_after.swap_mib << ",\n"
                        << "      \"swap_delta_mib\": " << r.swap_delta_mib << ",\n"
                        << "      \"vmlck_mib\": " << r.snap_after.lck_mib << ",\n"
                        << "      \"mem_avail_mib\": " << r.snap_after.mem_avail_mib << ",\n"
                        << "      \"pressure_buf_mib\": " << r.pressure_buf_mib << ",\n"
                        << "      \"majflt_delta\": " << r.majflt_delta << ",\n"
                        << "      \"minflt_delta\": " << r.minflt_delta << ",\n"
                        << "      \"disk_read_mib\": " << r.disk_read_mib << ",\n"
                        << "      \"ram_swap_bottleneck\": " << (r.ram_swap_bottleneck ? "true" : "false") << ",\n"
                        << "      \"moe_total_routed\": " << r.moe_stats.total_routed << ",\n"
                        << "      \"moe_hot_hits\": " << r.moe_stats.hot_hits << ",\n"
                        << "      \"moe_hit_rate_pct\": " << hit_pct << ",\n"
                        << "      \"moe_promotions\": " << r.moe_stats.promotions << ",\n"
                        << "      \"moe_demotions\": " << r.moe_stats.demotions << ",\n"
                        << "      \"moe_locked_mib\": " << (double(r.moe_stats.locked_bytes) / (1024.0 * 1024.0)) << ",\n"
                        << "      \"k_blocks_total\": " << r.total_stats.total_blocks << ",\n"
                        << "      \"k_blocks_skipped\": " << r.total_stats.skipped_k_blocks << ",\n"
                        << "      \"k_block_skip_pct\": " << k_skip_pct << ",\n"
                        << "      \"v_cells_total\": " << r.total_stats.total_v_cells << ",\n"
                        << "      \"v_cells_skipped\": " << r.total_stats.skipped_v_cells << ",\n"
                        << "      \"v_cell_skip_pct\": " << v_skip_pct << ",\n"
                        << "      \"logit_max_abs_err_vs_stock\": " << r.logit_err_stock.max_abs_err << ",\n"
                        << "      \"logit_mean_abs_err_vs_stock\": " << r.logit_err_stock.mean_abs_err << ",\n"
                        << "      \"kl_div_mean_vs_stock\": " << r.logit_err_stock.mean_kl_div << ",\n"
                        << "      \"kl_div_max_vs_stock\": " << r.logit_err_stock.max_kl_div << ",\n"
                        << "      \"logit_max_abs_err_vs_e1off\": " << r.logit_err_e1off.max_abs_err << ",\n"
                        << "      \"logit_mean_abs_err_vs_e1off\": " << r.logit_err_e1off.mean_abs_err << ",\n"
                        << "      \"kl_div_mean_vs_e1off\": " << r.logit_err_e1off.mean_kl_div << ",\n"
                        << "      \"kl_div_max_vs_e1off\": " << r.logit_err_e1off.max_kl_div << ",\n"
                        << "      \"needle1_657446\": " << (r.needle1_found ? "true" : "false") << ",\n"
                        << "      \"needle2_839204\": " << (r.needle2_found ? "true" : "false") << ",\n"
                        << "      \"needle3_412975\": " << (r.needle3_found ? "true" : "false") << ",\n"
                        << "      \"needles_found\": " << r.needles_found_count << ",\n"
                        << "      \"per_layer\": {\n";
                    for (size_t li = 0; li < 10; ++li) {
                        const int il = FULL_ATTN_LAYERS[li];
                        const auto & ls = r.layer_stats.at(il);
                        const auto & le_s = r.layer_attn_err_stock.at(il);
                        const auto & le_e = r.layer_attn_err_e1off.at(il);
                        const double lk_pct = ls.total_blocks ? 100.0 * double(ls.skipped_k_blocks) / double(ls.total_blocks) : 0.0;
                        const double lv_pct = ls.total_v_cells ? 100.0 * double(ls.skipped_v_cells) / double(ls.total_v_cells) : 0.0;
                        out << "        \"" << il << "\": {"
                            << "\"k_skip_pct\": " << lk_pct << ", "
                            << "\"v_skip_pct\": " << lv_pct << ", "
                            << "\"k_skipped\": " << ls.skipped_k_blocks << ", "
                            << "\"k_total\": " << ls.total_blocks << ", "
                            << "\"v_skipped\": " << ls.skipped_v_cells << ", "
                            << "\"v_total\": " << ls.total_v_cells << ", "
                            << "\"attn_max_abs_vs_stock\": " << le_s.max_abs << ", "
                            << "\"attn_mean_abs_vs_stock\": " << le_s.mean_abs << ", "
                            << "\"attn_l2_rel_vs_stock\": " << le_s.l2_rel << ", "
                            << "\"attn_max_abs_vs_e1off\": " << le_e.max_abs << ", "
                            << "\"attn_mean_abs_vs_e1off\": " << le_e.mean_abs << ", "
                            << "\"attn_l2_rel_vs_e1off\": " << le_e.l2_rel
                            << "}" << (li + 1 < 10 ? "," : "") << "\n";
                    }
                    out << "      }\n"
                        << "    }" << (i + 1 < all_results.size() ? "," : "") << "\n";
                }
                out << "  ]\n}\n";
                out.close();
            }
        }
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
