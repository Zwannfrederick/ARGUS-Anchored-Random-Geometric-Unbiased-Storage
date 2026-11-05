// ARGUS Heterogeneous Execution Scheduler, Hardware-Aware Runtime Profiler,
// Cost-Based Workload Partitioner, and Adaptive RAM/VRAM Memory Pressure Controller.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __linux__
#include <fstream>
#include <unistd.h>
#endif

enum ArgusPressureLevel : int {
    ARGUS_PRESSURE_NORMAL   = 0,
    ARGUS_PRESSURE_ELEVATED = 1,
    ARGUS_PRESSURE_CRITICAL = 2,
};

struct ArgusHardwareProfile {
    int    cpu_logical_cores = 1;
    bool   cpu_avx2          = false;
    bool   cpu_avx512        = false;
    bool   cpu_fma           = false;
    double cpu_mem_bw_gbps   = 34.0;   // Measured DDR4/DDR5 host read bandwidth
    double pcie_h2d_gbps     = 3.05;   // Pinned async H2D DMA bandwidth
    double pcie_d2h_gbps     = 3.20;   // Pinned async D2H DMA bandwidth
    double sync_overhead_us  = 5.50;   // Per-cudaStreamSynchronize round-trip cost
    double async_copy_us     = 1.80;   // Stream-ordered async copy enqueue cost

    size_t ram_total_bytes   = 0;
    size_t ram_avail_bytes   = 0;
    size_t swap_total_bytes  = 0;
    size_t swap_free_bytes   = 0;
    size_t proc_rss_bytes    = 0;
    size_t proc_vmswap_bytes = 0;

    bool   has_gpu           = false;
    size_t vram_total_bytes  = 0;
    size_t vram_free_bytes   = 0;

    ArgusPressureLevel ram_pressure  = ARGUS_PRESSURE_NORMAL;
    ArgusPressureLevel vram_pressure = ARGUS_PRESSURE_NORMAL;
};

struct ArgusOpPlacementCost {
    double gpu_compute_us  = 0.0;
    double cpu_compute_us  = 0.0;
    double pcie_xfer_us    = 0.0;
    double sync_us         = 0.0;
    double total_gpu_us    = 0.0;
    double total_cpu_us    = 0.0;
    bool   prefer_gpu      = true;
    bool   overlap_with_cpu = false;
};

inline ArgusHardwareProfile argus_probe_hardware(size_t free_vram = 0, size_t total_vram = 0) {
    ArgusHardwareProfile prof{};
#ifdef __linux__
    const long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs > 0) { prof.cpu_logical_cores = static_cast<int>(nprocs); }
#if defined(__x86_64__) || defined(_M_X64)
    prof.cpu_avx2   = __builtin_cpu_supports("avx2");
    prof.cpu_avx512 = __builtin_cpu_supports("avx512f");
    prof.cpu_fma    = __builtin_cpu_supports("fma");
#endif
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string key, unit;
        size_t val_kb = 0;
        while (meminfo >> key >> val_kb) {
            std::getline(meminfo, unit);
            if (key == "MemTotal:")     { prof.ram_total_bytes  = val_kb << 10; }
            if (key == "MemAvailable:") { prof.ram_avail_bytes  = val_kb << 10; }
            if (key == "SwapTotal:")    { prof.swap_total_bytes = val_kb << 10; }
            if (key == "SwapFree:")     { prof.swap_free_bytes  = val_kb << 10; }
        }
    }
    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            unsigned long long kb = 0;
            if (std::sscanf(line.c_str(), "VmRSS: %llu kB", &kb) == 1)  { prof.proc_rss_bytes    = static_cast<size_t>(kb) << 10; }
            if (std::sscanf(line.c_str(), "VmSwap: %llu kB", &kb) == 1) { prof.proc_vmswap_bytes = static_cast<size_t>(kb) << 10; }
        }
    }
#endif
    prof.vram_free_bytes  = free_vram;
    prof.vram_total_bytes = total_vram;
    prof.has_gpu          = total_vram > 0;

    constexpr size_t gi_b = size_t(1) << 30;
    constexpr size_t mi_b = size_t(1) << 20;
    if (prof.ram_avail_bytes > 0) {
        if (prof.ram_avail_bytes < (1536 * mi_b) || prof.proc_vmswap_bytes >= (64 * mi_b)) {
            prof.ram_pressure = ARGUS_PRESSURE_CRITICAL;
        } else if (prof.ram_avail_bytes < (3 * gi_b) || prof.proc_vmswap_bytes > 0) {
            prof.ram_pressure = ARGUS_PRESSURE_ELEVATED;
        }
    }
    if (prof.has_gpu) {
        if (prof.vram_free_bytes < (224 * mi_b)) {
            prof.vram_pressure = ARGUS_PRESSURE_CRITICAL;
        } else if (prof.vram_free_bytes < (512 * mi_b)) {
            prof.vram_pressure = ARGUS_PRESSURE_ELEVATED;
        }
    }
    return prof;
}

// Unified cost model evaluating whether a routed MoE FFN execution for `n_tokens` is faster on
// CPU (reading active expert slices in-place from DDR4/DDR5 with MADV_RANDOM) or on GPU (streaming
// weights across PCIe + MMQ kernel execution).
inline ArgusOpPlacementCost argus_evaluate_moe_cost(
        const ArgusHardwareProfile & prof,
        int64_t n_tokens,
        size_t layer_expert_bytes,
        int n_experts,
        int n_expert_used,
        double shared_expert_gpu_us = 46.2) {
    ArgusOpPlacementCost c{};
    const double p_miss = 1.0 - double(n_expert_used) / std::max(1, n_experts);
    const double active_frac = 1.0 - std::pow(p_miss, double(std::max<int64_t>(1, n_tokens)));
    const double active_bytes = double(layer_expert_bytes) * active_frac;

    // CPU execution reads only active expert slices from host RAM + small activation D2H/H2D (16 KiB/tok)
    const double d2h_h2d_bytes = double(std::max<int64_t>(1, n_tokens)) * 16384.0;
    c.cpu_compute_us = (active_bytes / (prof.cpu_mem_bw_gbps * 1e3)) + double(std::max<int64_t>(1, n_tokens)) * 180.0;
    c.pcie_xfer_us   = (d2h_h2d_bytes / (prof.pcie_d2h_gbps * 1e3)) + 2.0 * prof.async_copy_us;
    c.sync_us        = prof.sync_overhead_us; // Single batched stream sync per layer
    c.total_cpu_us   = c.cpu_compute_us + c.pcie_xfer_us + c.sync_us;

    // GPU execution streams active experts over PCIe (unless P3 layer-major prefill amortizes across ubatches)
    const double gpu_pcie_us = active_bytes / (prof.pcie_h2d_gbps * 1e3);
    c.gpu_compute_us = double(std::max<int64_t>(1, n_tokens)) * 12.0;
    c.total_gpu_us   = gpu_pcie_us + c.gpu_compute_us + prof.sync_overhead_us;

    c.prefer_gpu       = (n_tokens >= 32) || (c.total_gpu_us < c.total_cpu_us);
    c.overlap_with_cpu = !c.prefer_gpu && (shared_expert_gpu_us > 0.0) && (c.cpu_compute_us > shared_expert_gpu_us);
    return c;
}

// Select optimal micro-batch size (`n_ubatch`) before compute buffer reservation so that
// `self_kq_mask` ([n_ctx, n_ubatch]) and intermediate activation buffers leave enough VRAM
// for 100% S1 GPU KV mirroring at 8K, 32K, and 64K contexts without OOM.
inline uint32_t argus_hetero_select_ubatch(
        uint32_t requested_ubatch,
        uint32_t n_ctx,
        size_t free_vram_before_reserve,
        size_t total_vram,
        bool ubatch_explicitly_forced = false) {
    if (ubatch_explicitly_forced || total_vram == 0) {
        return requested_ubatch;
    }
    const ArgusHardwareProfile prof = argus_probe_hardware(free_vram_before_reserve, total_vram);
    uint32_t target = requested_ubatch;

    // On <= 6 GiB VRAM GPUs (e.g. 4 GB RTX 3050 Ti where ~1.74 GiB is free after model load):
    // - At 64K context (1,280 MiB KV cache), n_ubatch = 512 keeps self_kq_mask to 64 MiB and
    //   1-layer P3 compute buffer to ~150 MiB, leaving > 1,300 MiB for S1 so 100% of 64K KV fits on GPU.
    // - At 32K context (640 MiB KV cache), n_ubatch = 512 keeps self_kq_mask to 32 MiB and
    //   reduces peak VRAM by > 320 MiB while keeping 100% of 32K KV in S1 on GPU.
    constexpr size_t vram_6gb = size_t(6) << 30;
    if (total_vram <= vram_6gb) {
        if (n_ctx >= 49152 && target > 512) {
            target = 512;
        } else if (n_ctx >= 24576 && target > 512 && free_vram_before_reserve < (size_t(2048) << 20)) {
            target = 512;
        }
    }
    if (prof.ram_pressure == ARGUS_PRESSURE_CRITICAL && target > 256) {
        target = 256;
    } else if (prof.ram_pressure == ARGUS_PRESSURE_ELEVATED && target > 512) {
        target = 512;
    }
    return std::max<uint32_t>(32u, std::min(requested_ubatch, target));
}

// Compute the phase-shared VRAM arena budget after GGML compute buffers are already reserved.
// Coordinates three consumers of device memory:
// 1. OS/driver headroom (`cfg.vram_margin`, default 256 MiB, adaptive floor 144 MiB when full KV fits),
// 2. P3 layer-major prefill expert cache (`shared_expert_bytes`, e.g. 498 MiB),
// 3. S1 full-attention KV mirrors (`full_kv_bytes`, e.g. 160 MiB @ 8K, 640 MiB @ 32K, 1,280 MiB @ 64K).
inline size_t argus_plan_hetero_vram_budget(
        size_t free_vram,
        size_t shared_expert_bytes,
        size_t full_kv_bytes,
        size_t configured_margin,
        bool margin_explicit) {
    const size_t base_budget = free_vram > configured_margin ? free_vram - configured_margin : 0;
    if (margin_explicit) {
        return base_budget;
    }
    // If adapting the margin down to a safe 144 MiB driver floor allows 100% of `full_kv_bytes`
    // to fit in the S1 GPU mirror (eliminating all 10 CPU K1 attention splits during decode),
    // grant `full_kv_bytes`.
    constexpr size_t full_kv_driver_floor = size_t(144) << 20;
    if (full_kv_bytes > base_budget && free_vram >= full_kv_bytes + full_kv_driver_floor) {
        return std::max(full_kv_bytes, shared_expert_bytes);
    }
    // If `shared_expert_bytes` (P3 expert cache) needs slightly more than `base_budget`,
    // adapt down to a 64 MiB driver floor so P3 layer-major prefill still shares the arena.
    constexpr size_t min_driver_margin = size_t(64) << 20;
    if (shared_expert_bytes > base_budget && free_vram >= shared_expert_bytes + min_driver_margin) {
        return shared_expert_bytes;
    }
    return base_budget;
}

