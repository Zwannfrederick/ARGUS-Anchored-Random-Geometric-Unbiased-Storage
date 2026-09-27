// v0.8 KV tiers over llama.cpp's own KV cache (-nkvo).
//
// Host memory holds every cell of every layer: it is the truth, written by llama.cpp as
// usual. With ARGUS_KV_VRAM_BYTES set, device memory holds a write-through copy of the
// first G cells of each layer (its mirror): every append is written to both, and the copy
// is refreshed from the host after anything else rewrites cells (K shift, state restore).
// Invariant: a mirror's rows [0, G) equal its host rows [0, G) whenever a graph reads them.
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

// KV cache construction. Cells to mirror in a cache of kv_size cells whose layers need
// bytes_per_cell (K and V rows of every layer) per cell: whole 256-cell steps within what is
// left of ARGUS_KV_VRAM_BYTES, which this call reserves. 0 when tiers are off.
int64_t argus_kv_mirror_cells(int64_t kv_size, size_t bytes_per_cell);

// Registers `device` (cells + 1 rows; the last is scratch for rows outside the mirror) as
// the mirror of `host`, a layer's K or V cache tensor. The registration lasts as long as
// the returned handle, which the cache keeps.
std::shared_ptr<void> argus_kv_mirror_register(const ggml_tensor * host, ggml_tensor * device, int64_t cells);

// Host storage for KV that ARGUS mirrors: pinned memory, which the GPU writes directly (CUDA
// builds); nullptr to keep llama.cpp's choice.
ggml_backend_buffer_type_t argus_kv_host_buffer_type();

// Appends of `rows` into `host` at `indices` when `host` has a mirror: in pinned host storage
// one GPU node writes both (no CPU work on the decode path); otherwise, and for P2 prefill
// batches, llama.cpp's set_rows followed by the mirror's. nullptr when `host` has no mirror.
ggml_tensor * argus_kv_append(ggml_context * ctx, ggml_tensor * host, ggml_tensor * rows, ggml_tensor * indices);

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
