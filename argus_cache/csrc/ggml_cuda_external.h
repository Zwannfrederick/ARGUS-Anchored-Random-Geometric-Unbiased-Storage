// ARGUS CUDA graph nodes: every one is the same external custom operation registered with
// the CUDA backend, told apart by a kind stored in op_params after the attention scale.
// Not an API.
#pragma once

#include "ggml.h"

#include <cstdint>

enum ArgusCudaKind : int32_t {
    argus_kind_attention = 0,        // v0.7 paged attention over the ARGUS store
    argus_kind_set_rows = 1,         // v0.7 GPU appends into the ARGUS store
    argus_kind_decode_partials = 2,  // S1: per-chunk decode partials over GPU-resident cells
    argus_kind_partials_reduce = 3,  // S1: chunk partials merged into one per row
    argus_kind_tier_append = 4,      // tiers: new K/V rows into pinned host storage and its mirror
};

// A CUDA node of `kind` with an F32 result of shape [ne0, ne1, ne2]. `scale` and `extra`
// are stored for the kinds that read them back.
ggml_tensor * argus_cuda_external_op(ggml_context * ctx, int64_t ne0, int64_t ne1, int64_t ne2,
                                     ggml_tensor ** args, int n_args, ArgusCudaKind kind, float scale,
                                     int64_t extra = 0);
// The same as an in-place node on `a` (the result is a view of it, so GGML's scheduler places
// the node wherever `a` lives); args become src[1..].
ggml_tensor * argus_cuda_external_inplace(ggml_context * ctx, ggml_tensor * a, ggml_tensor ** args, int n_args,
                                          ArgusCudaKind kind, float scale, int64_t extra = 0);
int32_t argus_cuda_external_kind(const ggml_tensor * tensor); // -1 for any other node
float argus_cuda_external_scale(const ggml_tensor * tensor);
int64_t argus_cuda_external_extra(const ggml_tensor * tensor);

// Kind handlers, run on the CUDA backend's stream.
void argus_cuda_attention_compute(ggml_tensor * dst, int device, void * stream);
void argus_cuda_set_rows_compute(ggml_tensor * dst, int device, void * stream);
void argus_cuda_split_compute(ggml_tensor * dst, int device, void * stream);
