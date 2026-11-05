// P3: layer-major prefill for models whose experts stay in host memory (--cpu-moe).
//
// llama.cpp prefills a prompt one batch at a time through every layer, and for each batch
// copies every expert the batch routes to across PCIe (15.6 GB per 2048 tokens for
// Qwen3.6). Layer-major prefill runs every batch of a llama_decode call through one layer
// before the next, so each layer's experts cross once per call: ARGUS keeps that layer's
// expert tensors in a device cache, and graphs use the cached copies in place of the host
// tensors. Activations between layers wait in host memory.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using ArgusNamedTensors = std::vector<std::pair<std::string, ggml_tensor *>>;

struct ArgusMoeStats {
    uint64_t total_routed = 0;
    uint64_t hot_hits = 0;
    uint64_t promotions = 0;
    uint64_t demotions = 0;
    uint64_t advised_tensors = 0;
    uint64_t advised_bytes = 0;
    uint64_t locked_bytes = 0;
};

struct ArgusMoeLayerResidency {
    size_t total_bytes = 0;
    size_t resident_bytes = 0;
    size_t total_pages = 0;
    size_t resident_pages = 0;
};

// ARGUS_PREFILL=layers.
bool argus_layer_prefill_requested();

// Decide whether a prompt batch of `n_tokens_all` should engage P3 GPU layer-major prefill,
// honoring ARGUS_P3_MIN_TOKENS (explicit integer >= threshold or residency-adaptive 64..128).
bool argus_p3_should_engage(uint32_t n_tokens_all, uint32_t n_ubatch);
uint64_t argus_p3_engaged_calls();

// Apply POSIX_MADV_RANDOM to all host-resident MoE expert tensors and initialize
// per-layer hot-expert tracking tables.
void argus_moe_init_host_experts(const ArgusNamedTensors & model_tensors, int n_layer);

// Query OS page-cache residency (via mincore) of one layer's host MoE expert tensors.
ArgusMoeLayerResidency argus_moe_layer_residency(int il);

// Evict one layer's host MoE expert pages from OS page cache (MADV_DONTNEED) for cold-cache I/O benchmarks.
void argus_moe_evict_layer_pages(int il);

// Attach an in-place CPU routing observer on `cur` that reads `selected_experts` and maintains
// the per-layer top-K hot expert working set in active OS page cache.
ggml_tensor * argus_moe_observe_routing(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * selected_experts,
                                        ggml_tensor * up_exps, ggml_tensor * gate_exps,
                                        ggml_tensor * down_exps, ggml_tensor * gate_up_exps, int il);

__attribute__((weak)) ArgusMoeStats argus_moe_stats();
__attribute__((weak)) void argus_moe_reset_stats();

// Host-resident expert tensors ("blk.<layer>.*_exps*") of one layer; empty if none.
std::vector<ggml_tensor *> argus_layer_experts(const ArgusNamedTensors & model_tensors, int layer);

// Bytes the device cache needs: the largest layer's host-resident experts.
size_t argus_expert_cache_bytes(const ArgusNamedTensors & model_tensors, int n_layer, ggml_backend_dev_t dev);

// Sets up the cache once the KV mirrors are allocated: it borrows their arena when that is
// large enough (VRAM shared by phase: experts during prefill, mirrors during decode; the
// caller detaches the mirrors for a prefill), and allocates its own device memory otherwise.
// False when the model keeps no experts on the host or neither fits: prefill stays batch-major.
bool argus_expert_cache_reserve(const ArgusNamedTensors & model_tensors, int n_layer, ggml_backend_dev_t dev);
bool argus_expert_cache_ready();

// Elastic host-RAM controller: expands active MoE layers into unused RAM when MemAvailable grows,
// and immediately sheds upper MoE layers (MADV_DONTNEED + POSIX_FADV_DONTNEED) when desktop/IDE needs RAM.
void argus_dynamic_ram_rebalance(bool force = false);

struct ArgusDecodeScope {
    explicit ArgusDecodeScope(uint32_t n_tokens);
    ~ArgusDecodeScope();
    ArgusDecodeScope(const ArgusDecodeScope &) = delete;
    ArgusDecodeScope & operator=(const ArgusDecodeScope &) = delete;
};

// Prepare overflow headroom and prefetch the first cold MoE layers at the start of a P3 pass.
void argus_expert_cache_prepare_p3();

// Asynchronously prefetch `experts` into OS page cache ahead of GPU execution.
void argus_expert_cache_prefetch(const std::vector<ggml_tensor *> & experts);

// Copies `experts` into the cache; graphs built afterwards use the copies. Replaces the
// previous layer's. When `layer_idx` >= dynamic RAM window, releases clean overflow pages after copy.
void argus_expert_cache_load(const std::vector<ggml_tensor *> & experts, int layer_idx = -1);
// Graphs use the host tensors again.
void argus_expert_cache_unload();

// Bind/unbind lightweight dummy device-resident expert tensors during sched_reserve() so
// ggml_backend_sched reserves P3 1-layer prefill graphs without allocating 150 MiB of dead
// host-to-device MoE weight copies in the CUDA compute buffer.
void argus_expert_cache_bind_dummy(const ArgusNamedTensors & model_tensors, int n_layer, ggml_backend_dev_t dev);
void argus_expert_cache_unbind_dummy();

// Graph build: the cached copy of `tensor`, or `tensor` itself when it is not cached.
ggml_tensor * argus_expert_tensor(ggml_tensor * tensor);
