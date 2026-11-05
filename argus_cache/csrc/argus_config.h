// libargus unified configuration and autonomous resource planner.
// Replaces scattered environment checks with one coherent configuration struct, programmatic
// override hooks, and an autonomous mode (ARGUS_AUTO=1) that activates K1 CPU decode, S1
// device mirrors, P2 GPU prefill, and P3 layer-major prefill with phase-shared VRAM sizing.
#pragma once

#include "argus_hetero.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

struct ArgusConfig {
    bool auto_mode = false;          // ARGUS_AUTO=1: zero-config autonomous orchestration
    bool k1_enabled = false;         // ARGUS_KV_CPU_ATTENTION=k1 (or auto_mode)
    int64_t k1_max_tokens = 16;      // ARGUS_K1_MAX_TOKENS
    bool vram_auto = false;          // ARGUS_KV_VRAM_BYTES=auto (or auto_mode)
    size_t vram_bytes = 0;           // Explicit ARGUS_KV_VRAM_BYTES budget
    size_t vram_margin = size_t(256) << 20; // ARGUS_KV_VRAM_MARGIN (256 MiB default)
    bool margin_explicit = false;    // True when ARGUS_KV_VRAM_MARGIN was set by caller
    bool prefill_on_gpu = false;     // ARGUS_KV_PREFILL=gpu (or auto_mode)
    bool layer_prefill = false;      // ARGUS_PREFILL=layers (or auto_mode)
    float skip_eps = 0.0f;           // ARGUS_KV_SKIP_EPS: E1 bounded-mass page/cell skipping threshold (0 = exact)
    bool madv_random = false;        // ARGUS_MADV_RANDOM=1: apply POSIX_MADV_RANDOM to host MoE expert tensors
    bool moe_cache_enabled = false;  // ARGUS_MOE_CACHE=1: track and pin top-K hot MoE experts per layer
    int hot_experts_per_layer = 32;  // ARGUS_HOT_EXPERTS: top-K hot experts per layer pinned in active RAM
    bool multipole_enabled = false;  // ARGUS_MULTIPOLE=1: V-centroid compensation + monopole/dipole K-box check (opt-in)
    bool top_r_enabled = false;      // ARGUS_TOP_R=1: Top-r channel + Cauchy-Schwarz tail bound for K-boxes (opt-in)
    bool hetero_sched = false;       // ARGUS_HETERO_SCHED=1: adaptive VRAM/ubatch reservation + async split DMA (or auto_mode)
    bool hetero_overlap = false;     // ARGUS_HETERO_OVERLAP=1: parallel GPU shared-expert & CPU routed-expert execution (or auto_mode)
    int64_t p3_min_tokens = 128;     // ARGUS_P3_MIN_TOKENS: min batch tokens for P3 GPU prefill (64/128/256/512 or "auto")
    bool p3_adaptive = true;         // True when ARGUS_P3_MIN_TOKENS is unset or "auto" (residency-aware 64..128)
    bool p3_log = false;             // ARGUS_P3_LOG=1: log every P3 sub-batch engagement to stderr for benchmarking
    int mmap_policy = 0;             // ARGUS_MMAP_POLICY: 0=default(madv_random), 1=normal, 2=sequential
    bool dynamic_ram = true;         // ARGUS_DYNAMIC_RAM=1: elastic host MoE RAM window expanding/shrinking with MemAvailable
    size_t ram_reserve_bytes = size_t(3456) << 20; // ARGUS_RAM_RESERVE_MIB: non-MoE RAM reserve for IDE/OS Active(file) + prefetch cushion (3.375 GiB default)
};

namespace argus_detail {
inline std::mutex & config_mutex() {
    static std::mutex m;
    return m;
}

inline ArgusConfig & override_slot() {
    static ArgusConfig cfg{};
    return cfg;
}

inline bool & has_override_slot() {
    static bool flag = false;
    return flag;
}

inline size_t parse_bytes(const char * name, const char * text) {
    size_t value = 0;
    const char * end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{}) {
        throw std::invalid_argument(std::string(name) + " must be a byte count");
    }
    if (*parsed.ptr) {
        const std::string suffix(parsed.ptr);
        if (suffix == "K" || suffix == "k" || suffix == "KiB") { value <<= 10; }
        else if (suffix == "M" || suffix == "m" || suffix == "MiB") { value <<= 20; }
        else if (suffix == "G" || suffix == "g" || suffix == "GiB") { value <<= 30; }
        else { throw std::invalid_argument(std::string(name) + " must be a byte count"); }
    }
    return value;
}

inline bool parse_auto_flag() {
    const char * text = std::getenv("ARGUS_AUTO");
    if (!text || !*text || std::string(text) == "0" || std::string(text) == "off") { return false; }
    if (std::string(text) == "1" || std::string(text) == "on") { return true; }
    throw std::invalid_argument("ARGUS_AUTO must be 0, 1, off, or on");
}
} // namespace argus_detail

inline void argus_set_config(const ArgusConfig & cfg) {
    std::lock_guard<std::mutex> guard(argus_detail::config_mutex());
    argus_detail::override_slot() = cfg;
    argus_detail::has_override_slot() = true;
}

inline void argus_reset_config() {
    std::lock_guard<std::mutex> guard(argus_detail::config_mutex());
    argus_detail::override_slot() = ArgusConfig{};
    argus_detail::has_override_slot() = false;
}

inline ArgusConfig argus_get_config() {
    {
        std::lock_guard<std::mutex> guard(argus_detail::config_mutex());
        if (argus_detail::has_override_slot()) { return argus_detail::override_slot(); }
    }
    ArgusConfig cfg{};
    cfg.auto_mode = argus_detail::parse_auto_flag();

    if (const char * k1 = std::getenv("ARGUS_KV_CPU_ATTENTION"); k1 && *k1) {
        if (std::string(k1) == "k1") { cfg.k1_enabled = true; }
        else if (std::string(k1) == "stock") { cfg.k1_enabled = false; }
        else { throw std::runtime_error("ARGUS_KV_CPU_ATTENTION must be stock or k1"); }
    } else {
        cfg.k1_enabled = cfg.auto_mode;
    }

    if (const char * mt = std::getenv("ARGUS_K1_MAX_TOKENS"); mt && *mt) {
        char * end = nullptr;
        const long val = std::strtol(mt, &end, 10);
        if (end == mt || *end || val <= 0) {
            throw std::runtime_error("ARGUS_K1_MAX_TOKENS must be a positive integer");
        }
        cfg.k1_max_tokens = val;
    }

    if (const char * vb = std::getenv("ARGUS_KV_VRAM_BYTES"); vb && *vb) {
        if (std::string(vb) == "auto") { cfg.vram_auto = true; }
        else { cfg.vram_bytes = argus_detail::parse_bytes("ARGUS_KV_VRAM_BYTES", vb); }
    } else if (cfg.auto_mode) {
        cfg.vram_auto = true;
    }

    if (const char * vm = std::getenv("ARGUS_KV_VRAM_MARGIN"); vm && *vm) {
        cfg.vram_margin = argus_detail::parse_bytes("ARGUS_KV_VRAM_MARGIN", vm);
        cfg.margin_explicit = true;
    }

    if (const char * pf = std::getenv("ARGUS_KV_PREFILL"); pf && *pf) {
        if (std::string(pf) == "gpu") { cfg.prefill_on_gpu = true; }
        else if (std::string(pf) == "cpu") { cfg.prefill_on_gpu = false; }
        else { throw std::runtime_error("ARGUS_KV_PREFILL must be cpu or gpu"); }
    } else {
        cfg.prefill_on_gpu = cfg.auto_mode;
    }

    if (const char * lp = std::getenv("ARGUS_PREFILL"); lp && *lp) {
        if (std::string(lp) == "layers") { cfg.layer_prefill = true; }
        else if (std::string(lp) == "batches") { cfg.layer_prefill = false; }
        else { throw std::runtime_error("ARGUS_PREFILL must be batches or layers"); }
    } else {
        cfg.layer_prefill = cfg.auto_mode;
    }

    if (const char * eps = std::getenv("ARGUS_KV_SKIP_EPS"); eps && *eps) {
        char * end = nullptr;
        const float val = std::strtof(eps, &end);
        if (end == eps || *end || !std::isfinite(val) || val < 0.0f || val >= 1.0f) {
            throw std::runtime_error("ARGUS_KV_SKIP_EPS must be a finite float in [0, 1)");
        }
        cfg.skip_eps = val;
    } else if (const char * e1 = std::getenv("ARGUS_E1"); e1 && *e1) {
        if (std::string(e1) == "1" || std::string(e1) == "on") { cfg.skip_eps = 1e-4f; }
        else if (std::string(e1) == "0" || std::string(e1) == "off") { cfg.skip_eps = 0.0f; }
        else { throw std::runtime_error("ARGUS_E1 must be 0, 1, off, or on"); }
    }

    if (const char * mr = std::getenv("ARGUS_MADV_RANDOM"); mr && *mr) {
        if (std::string(mr) == "1" || std::string(mr) == "on") { cfg.madv_random = true; }
        else if (std::string(mr) == "0" || std::string(mr) == "off") { cfg.madv_random = false; }
        else { throw std::runtime_error("ARGUS_MADV_RANDOM must be 0, 1, off, or on"); }
    } else {
        cfg.madv_random = cfg.auto_mode || cfg.k1_enabled;
    }

    if (const char * mc = std::getenv("ARGUS_MOE_CACHE"); mc && *mc) {
        if (std::string(mc) == "1" || std::string(mc) == "on") { cfg.moe_cache_enabled = true; }
        else if (std::string(mc) == "0" || std::string(mc) == "off") { cfg.moe_cache_enabled = false; }
        else { throw std::runtime_error("ARGUS_MOE_CACHE must be 0, 1, off, or on"); }
    } else {
        cfg.moe_cache_enabled = false;
    }

    if (const char * he = std::getenv("ARGUS_HOT_EXPERTS"); he && *he) {
        char * end = nullptr;
        const long val = std::strtol(he, &end, 10);
        if (end == he || *end || val < 0) {
            throw std::runtime_error("ARGUS_HOT_EXPERTS must be a non-negative integer");
        }
        cfg.hot_experts_per_layer = static_cast<int>(val);
    }

    if (const char * mp = std::getenv("ARGUS_MULTIPOLE"); mp && *mp) {
        if (std::string(mp) == "1" || std::string(mp) == "on") { cfg.multipole_enabled = true; }
        else if (std::string(mp) == "0" || std::string(mp) == "off") { cfg.multipole_enabled = false; }
        else { throw std::runtime_error("ARGUS_MULTIPOLE must be 0, 1, off, or on"); }
    } else {
        cfg.multipole_enabled = false;
    }

    if (const char * tr = std::getenv("ARGUS_TOP_R"); tr && *tr) {
        if (std::string(tr) == "1" || std::string(tr) == "on") { cfg.top_r_enabled = true; }
        else if (std::string(tr) == "0" || std::string(tr) == "off") { cfg.top_r_enabled = false; }
        else { throw std::runtime_error("ARGUS_TOP_R must be 0, 1, off, or on"); }
    } else {
        cfg.top_r_enabled = false;
    }

    if (const char * hs = std::getenv("ARGUS_HETERO_SCHED"); hs && *hs) {
        if (std::string(hs) == "1" || std::string(hs) == "on") { cfg.hetero_sched = true; }
        else if (std::string(hs) == "0" || std::string(hs) == "off") { cfg.hetero_sched = false; }
        else { throw std::runtime_error("ARGUS_HETERO_SCHED must be 0, 1, off, or on"); }
    } else {
        cfg.hetero_sched = cfg.auto_mode;
    }

    if (const char * ho = std::getenv("ARGUS_HETERO_OVERLAP"); ho && *ho) {
        if (std::string(ho) == "1" || std::string(ho) == "on") { cfg.hetero_overlap = true; }
        else if (std::string(ho) == "0" || std::string(ho) == "off") { cfg.hetero_overlap = false; }
        else { throw std::runtime_error("ARGUS_HETERO_OVERLAP must be 0, 1, off, or on"); }
    } else {
        cfg.hetero_overlap = cfg.auto_mode || cfg.hetero_sched;
    }

    if (const char * pm = std::getenv("ARGUS_P3_MIN_TOKENS"); pm && *pm) {
        if (std::string(pm) == "auto") {
            cfg.p3_adaptive = true;
            cfg.p3_min_tokens = 128;
        } else {
            char * end = nullptr;
            const long val = std::strtol(pm, &end, 10);
            if (end == pm || *end || val <= 0) {
                throw std::runtime_error("ARGUS_P3_MIN_TOKENS must be 'auto' or a positive integer");
            }
            cfg.p3_adaptive = false;
            cfg.p3_min_tokens = val;
        }
    }

    if (const char * pl = std::getenv("ARGUS_P3_LOG"); pl && *pl) {
        if (std::string(pl) == "1" || std::string(pl) == "on") { cfg.p3_log = true; }
        else if (std::string(pl) == "0" || std::string(pl) == "off") { cfg.p3_log = false; }
        else { throw std::runtime_error("ARGUS_P3_LOG must be 0, 1, off, or on"); }
    }

    if (const char * pol = std::getenv("ARGUS_MMAP_POLICY"); pol && *pol) {
        const std::string s(pol);
        if (s == "random")          { cfg.madv_random = true;  cfg.mmap_policy = 0; }
        else if (s == "normal")     { cfg.madv_random = false; cfg.mmap_policy = 1; }
        else if (s == "sequential") { cfg.madv_random = false; cfg.mmap_policy = 2; }
        else { throw std::runtime_error("ARGUS_MMAP_POLICY must be random, normal, or sequential"); }
    }

    if (const char * dr = std::getenv("ARGUS_DYNAMIC_RAM"); dr && *dr) {
        if (std::string(dr) == "1" || std::string(dr) == "on") { cfg.dynamic_ram = true; }
        else if (std::string(dr) == "0" || std::string(dr) == "off") { cfg.dynamic_ram = false; }
        else { throw std::runtime_error("ARGUS_DYNAMIC_RAM must be 0, 1, off, or on"); }
    }

    if (const char * rm = std::getenv("ARGUS_RAM_RESERVE_MIB"); rm && *rm) {
        char * end = nullptr;
        const long val = std::strtol(rm, &end, 10);
        if (end == rm || *end || val < 512) {
            throw std::runtime_error("ARGUS_RAM_RESERVE_MIB must be an integer >= 512");
        }
        cfg.ram_reserve_bytes = static_cast<size_t>(val) << 20;
    }

    return cfg;
}

// Compute the phase-shared VRAM arena budget after GGML compute buffers are already reserved.
inline size_t argus_plan_vram_budget(size_t free_vram, size_t shared_expert_bytes, const ArgusConfig & cfg,
                                     size_t full_kv_bytes = 0) {
    return argus_plan_hetero_vram_budget(free_vram, shared_expert_bytes, full_kv_bytes,
                                         cfg.vram_margin, cfg.margin_explicit);
}

