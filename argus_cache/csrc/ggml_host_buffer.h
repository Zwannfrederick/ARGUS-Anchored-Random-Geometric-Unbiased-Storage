#pragma once

// llama.cpp ownership seam: exact GGML layout in ARGUS-owned, file-backed memory.
// With ARGUS_KV_RESIDENT_BYTES set, attention reads that memory block by block
// and returns cold blocks to storage; the tensor type stays GGML's.
#include "ggml-backend.h"

#include <cstddef>

ggml_backend_buffer_type_t argus_ggml_host_buffer_type();

// Target for resident KV file pages, not a hard RSS limit; 0 disables paging.
size_t argus_kv_resident_budget();
// Fraction of each KV tensor's cells that may stay resident after attention.
double argus_kv_hot_fraction();
void argus_kv_page_out(const void * address, size_t bytes, bool cold_before = false);
// Request readahead for this range; the OS controls timing and residency.
void argus_kv_prefetch(const void * address, size_t bytes);
void argus_kv_note_attention(size_t bytes_read, size_t bytes_paged_out);
void argus_kv_publish_stats();

// Returns nullptr when the KV does not live in an ARGUS paged buffer.
// Throws std::runtime_error for an ARGUS-paged KV with an unsupported contract.
ggml_tensor * argus_ggml_paged_attention(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * mask, bool unsupported_features, float scale);
