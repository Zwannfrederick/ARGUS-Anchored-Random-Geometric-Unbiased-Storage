// v0.8 KV tiers over llama.cpp's own KV cache (-nkvo).
//
// Host memory holds every cell of every layer: it is the truth, written by llama.cpp as
// usual. With ARGUS_KV_VRAM_BYTES set, device memory holds a write-through copy of the
// first G cells of each layer (its mirror): every append is written to both, and the copy
// is refreshed from the host after anything else rewrites cells (K shift, state restore).
// Invariant: a mirror's rows [0, G) equal its host rows [0, G) whenever a graph reads them.
// Mirrors are sized and allocated once the context knows its compute buffers (P1).
//
// Decode attention then runs where each part lives (S1): the GPU over [0, G), K1 over
// [G, n_kv), merged by log-sum-exp, only q and partials crossing PCIe. A context that fits
// the mirror never leaves the GPU. Prefill batches run on the GPU over the whole view (P2).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>

// ARGUS_KV_VRAM_BYTES is set: a byte budget for mirrors, or "auto" (the device's free memory
// once the context has reserved its compute buffers, minus ARGUS_KV_VRAM_MARGIN, 256 MiB by
// default).
bool argus_kv_tiers_requested();

// KV cache construction: registers `host`, a layer's K or V tensor, to be mirrored on `dev`.
// The registration lasts as long as the returned handle, which the cache keeps.
std::shared_ptr<void> argus_kv_register(const ggml_tensor * host, ggml_backend_dev_t dev);

// After a context has reserved its compute buffers: mirrors for every registered tensor that
// has none, the same leading cells for every layer, within the budget, filled from the host.
// Tensors left without a mirror keep host KV and K1. The arena holding them is at least
// `shared_bytes` when the budget allows, for a user that borrows it while they are detached.
void argus_kv_allocate_mirrors(size_t shared_bytes = 0);

// The mirrors' device memory, if it is at least `bytes`; nullptr otherwise.
ggml_backend_buffer_t argus_kv_arena(size_t bytes);

// Lend the arena out (P3 prefill): mirrors are neither read nor written until attached again;
// the tiers are off meanwhile (prefill attention reads host KV) and appends write the host only. Attaching fills every
// mirror from its host rows, which restores the invariant.
void argus_kv_mirrors_detach();
void argus_kv_mirrors_attach();

// Host storage for KV that ARGUS mirrors: pinned memory, which the GPU writes directly (CUDA
// builds); nullptr to keep llama.cpp's choice.
ggml_backend_buffer_type_t argus_kv_host_buffer_type();

// Appends of `rows` into `host` at `indices` when `host` has a mirror: in pinned host storage
// one GPU node writes both (for decode and for prefill batches that fit inside the mirror);
// otherwise llama.cpp's set_rows followed by the mirror's (when min_idx < mirror.cells).
ggml_tensor * argus_kv_append(ggml_context * ctx, ggml_tensor * host, ggml_tensor * rows, ggml_tensor * indices,
                              int64_t n_kv = 0, int64_t min_idx = 0);

// Replace host K/V views with F16 GPU mirror views during prefill when [0, n_kv) fits in the mirror.
bool argus_kv_prefill_mirror_views(ggml_context * ctx, const ggml_tensor * q, ggml_tensor * & k, ggml_tensor * & v,
                                   bool unsupported_features);

// Host cells were rewritten outside appends: copy host rows [0, G) of `host` to its mirror.
void argus_kv_mirror_refresh(const ggml_tensor * host);

// Decode attention over host views k/v whose storage has mirrors; nullptr otherwise (and
// for prefill batches). Its GPU nodes are assigned to the CUDA backend and its K1 nodes to
// `cpu` (GGML would otherwise spread the CPU from K1 to its GPU neighbours).
ggml_tensor * argus_kv_tiered_attention(ggml_context * ctx, ggml_backend_sched_t sched, ggml_backend_t cpu,
                                        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
                                        bool unsupported_features, float scale);

// ARGUS manages the KV cache (ARGUS_KV_CPU_ATTENTION=k1 or ARGUS_KV_VRAM_BYTES set). Then
// state of fixed size that every token reads in full (recurrent state, sliding-window
// caches) stays in device memory under -nkvo: only the growing full-attention KV is placed
// by ARGUS. Stock -nkvo moves both to host memory and runs those layers on the CPU.
bool argus_kv_managed();

// P2: a batch of n_tokens over host KV should run its attention on the GPU (ARGUS_KV_PREFILL=gpu).
bool argus_kv_prefill_on_gpu(const ggml_tensor * k, int64_t n_tokens);
