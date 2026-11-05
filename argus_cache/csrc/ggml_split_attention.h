// S1, GPU side: decode-attention partials over the KV cells that live in GPU memory, in
// K1's partial format ({max, sum, unnormalized output[dv]} per row; rows ordered by KV head,
// then token, then query head within the group). The CPU side (K1) covers the other cells
// and merges this result first, so one attention call is computed where its data lives and
// only q and partials cross PCIe.
#pragma once

#include "ggml.h"

// q [dk, n_head, n_tokens] F32; k/v [d, n_head_kv, cells] F16 in device memory (d 64, 128,
// 256 or 512); the mask is the whole [n_kv, >= n_tokens] F16/F32 tensor and cell c of k/v
// reads its column mask_cell0 + c (a mask view would not survive a copy between backends).
// Result [dv + 2, n_head_kv * rows, 1]. Returns nullptr outside what the kernels cover.
ggml_tensor * argus_ggml_cuda_decode_partials(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                              ggml_tensor * mask, int64_t mask_cell0, float scale);

// Tier appends: F32 rows into F16 host storage `host` (pinned, written by the device through
// its mapping) at `indices`, and into its mirror (cells + 1 rows; rows past the mirror land on
// its last, scratch row), in one kernel on the GPU stream. nullptr if the layout is unsupported.
ggml_tensor * argus_ggml_cuda_tier_append(ggml_context * ctx, ggml_tensor * mirror, ggml_tensor * rows,
                                          ggml_tensor * indices, const ggml_tensor * host);

// The same over every cell of the call, normalized on the GPU: result [dv, n_head, n_tokens]
// like GGML flash attention's, so a context held by the mirror never leaves the GPU.
ggml_tensor * argus_ggml_cuda_decode_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                               ggml_tensor * mask, float scale);

// Copy/dequantize `cells` rows from pinned `host` (Q8_0 or Q4_0) directly into `device` (F16) on the GPU.
bool argus_cuda_sync_mirror_from_host(const ggml_tensor * host, ggml_tensor * device, int64_t cells);
