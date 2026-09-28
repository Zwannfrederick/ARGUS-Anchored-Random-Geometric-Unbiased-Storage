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
#include <string>
#include <utility>
#include <vector>

using ArgusNamedTensors = std::vector<std::pair<std::string, ggml_tensor *>>;

// ARGUS_PREFILL=layers.
bool argus_layer_prefill_requested();

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

// Copies `experts` into the cache; graphs built afterwards use the copies. Replaces the
// previous layer's.
void argus_expert_cache_load(const std::vector<ggml_tensor *> & experts);
// Graphs use the host tensors again.
void argus_expert_cache_unload();

// Graph build: the cached copy of `tensor`, or `tensor` itself when it is not cached.
ggml_tensor * argus_expert_tensor(ggml_tensor * tensor);
