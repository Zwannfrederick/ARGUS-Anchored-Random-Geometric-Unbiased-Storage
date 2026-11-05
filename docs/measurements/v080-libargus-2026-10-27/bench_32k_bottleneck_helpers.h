#pragma once

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml_decode_attention.h"
#include "ggml_layer_prefill.h"
#include "argus_config.h"

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
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace argus_audit {

constexpr int N_LAYERS = 40;
constexpr int FULL_ATTN_LAYERS[10] = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};
constexpr int N_CTX_32K = 32768;

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

inline SysSnapshot read_sys_snapshot(bool query_vram = true) {
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
    if (query_vram) {
        if (ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(gpu, &free_b, &total_b);
            if (total_b >= free_b) {
                s.vram_used_mib = double(total_b - free_b) / (1024.0 * 1024.0);
            }
        }
    }
    return s;
}

inline double percentile_sorted(const std::vector<double> & sorted, double p) {
    if (sorted.empty()) { return 0.0; }
    const double idx = (p / 100.0) * double(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    const double frac = idx - double(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

inline int g_last_splits_pp = 0;
inline int g_last_splits_tg = 0;

inline void split_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (!text) { return; }
    const char * p = std::strstr(text, "graph splits = ");
    if (p) {
        int pp = 0, bs = 0, tg = 0;
        if (std::sscanf(p, "graph splits = %d (with bs=%d), %d (with bs=1)", &pp, &bs, &tg) == 3) {
            g_last_splits_pp = pp;
            g_last_splits_tg = tg;
        }
    }
}

inline std::vector<llama_token> tokenize_str(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n = llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
    if (n < 0) { n = -n; }
    std::vector<llama_token> tokens(n);
    int actual = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, true);
    if (actual < 0) { throw std::runtime_error("Tokenization failed"); }
    tokens.resize(actual);
    return tokens;
}

inline std::vector<llama_token> build_multi_needle_prompt(const llama_vocab * vocab, int target_prompt_tokens) {
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

inline llama_token greedy_argmax(const float * logits, int n_vocab) {
    return static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
}

inline void run_prefill(llama_context * ctx, const std::vector<llama_token> & prompt, int ubatch_size, bool layer_major) {
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

struct LayerResidencySummary {
    double total_mib = 0.0;
    double resident_mib = 0.0;
    double resident_pct = 0.0;
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    std::vector<double> layer_resident_mib;
    std::vector<double> layer_resident_pct;
};

inline LayerResidencySummary query_all_moe_residency() {
    LayerResidencySummary sum{};
    sum.layer_resident_mib.resize(N_LAYERS, 0.0);
    sum.layer_resident_pct.resize(N_LAYERS, 0.0);
    for (int il = 0; il < N_LAYERS; ++il) {
        const ArgusMoeLayerResidency r = argus_moe_layer_residency(il);
        const double tot_m = double(r.total_bytes) / (1024.0 * 1024.0);
        const double res_m = double(r.resident_bytes) / (1024.0 * 1024.0);
        const double pct   = r.total_pages > 0 ? 100.0 * double(r.resident_pages) / double(r.resident_pages ? r.total_pages : 1) : 0.0;
        sum.total_mib      += tot_m;
        sum.resident_mib   += res_m;
        sum.total_pages    += r.total_pages;
        sum.resident_pages += r.resident_pages;
        sum.layer_resident_mib[il] = res_m;
        sum.layer_resident_pct[il] = pct;
    }
    sum.resident_pct = sum.total_pages > 0 ? 100.0 * double(sum.resident_pages) / double(sum.total_pages) : 0.0;
    return sum;
}

struct LayerFaultStats {
    uint64_t majflt = 0;
    uint64_t minflt = 0;
    uint64_t io_read_bytes = 0;
    double elapsed_ms = 0.0;
    std::set<int32_t> unique_experts;
};

struct LayerFaultTracker {
    bool active = false;
    SysSnapshot last_snap{};
    std::chrono::steady_clock::time_point last_t{};
    LayerFaultStats layers[N_LAYERS]{};

    void reset() {
        for (int i = 0; i < N_LAYERS; ++i) {
            layers[i] = LayerFaultStats{};
        }
        last_snap = read_sys_snapshot(false);
        last_t = std::chrono::steady_clock::now();
    }
};

inline bool layer_fault_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * tr = static_cast<LayerFaultTracker *>(user_data);
    if (!tr || !tr->active || !t || !t->name[0]) { return false; }

    if (std::strcmp(t->name, "model.input_embed") == 0) {
        if (ask) { return true; }
        tr->last_snap = read_sys_snapshot(false);
        tr->last_t = std::chrono::steady_clock::now();
        return true;
    }
    if (std::strncmp(t->name, "ffn_moe_topk-", 13) == 0) {
        if (ask) { return true; }
        const int il = std::atoi(t->name + 13);
        if (il >= 0 && il < N_LAYERS) {
            const int64_t n_el = ggml_nelements(t);
            std::vector<int32_t> ids(n_el);
            ggml_backend_tensor_get(t, ids.data(), 0, n_el * sizeof(int32_t));
            for (int32_t id : ids) {
                if (id >= 0 && id < 256) {
                    tr->layers[il].unique_experts.insert(id);
                }
            }
        }
        return true;
    }
    if (std::strncmp(t->name, "ffn_moe_out-", 12) == 0) {
        if (ask) { return true; }
        const int il = std::atoi(t->name + 12);
        const auto now_t = std::chrono::steady_clock::now();
        const SysSnapshot now_s = read_sys_snapshot(false);
        if (il >= 0 && il < N_LAYERS) {
            tr->layers[il].majflt += (now_s.majflt >= tr->last_snap.majflt) ? (now_s.majflt - tr->last_snap.majflt) : 0;
            tr->layers[il].minflt += (now_s.minflt >= tr->last_snap.minflt) ? (now_s.minflt - tr->last_snap.minflt) : 0;
            tr->layers[il].io_read_bytes += (now_s.io_read_bytes >= tr->last_snap.io_read_bytes)
                ? (now_s.io_read_bytes - tr->last_snap.io_read_bytes) : 0;
            tr->layers[il].elapsed_ms += std::chrono::duration<double, std::milli>(now_t - tr->last_t).count();
        }
        tr->last_snap = now_s;
        tr->last_t = now_t;
        return true;
    }
    return false;
}

inline void clear_argus_env() {
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
    unsetenv("ARGUS_MOE_SPLIT_OPT");
    unsetenv("ARGUS_RS_GPU_OPT");
    unsetenv("ARGUS_PROFILE_KERNELS");
}

} // namespace argus_audit
