// K1: exact decode attention over KV in host memory that reads every KV byte once per call.
// One pass over a KV head's cells serves every query head that shares it (GQA), so the
// bytes cross the memory bus once instead of once per query head.
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

struct ArgusK1Problem {
    int64_t dk, dv, n_head, n_head_kv, n_tokens, n_kv;
    ggml_type k_type, v_type;
    const char * q;    size_t q_head_stride, q_token_stride; // F32 rows of dk
    const char * k;    size_t k_head_stride, k_cell_stride;
    const char * v;    size_t v_head_stride, v_cell_stride;
    const char * mask; ggml_type mask_type; size_t mask_token_stride; // mask[token][cell], F16 or F32; nullptr: none
    float scale;

    int64_t group() const { return n_head / n_head_kv; }
    int64_t rows() const { return n_tokens * group(); } // rows served by one KV head
};

// Cells are split into fixed chunks that depend only on n_kv, so the result is the same
// bit for bit whatever the thread count. A task is one (KV head, chunk).
int64_t argus_k1_chunk_cells(int64_t n_kv);
int64_t argus_k1_tasks(const ArgusK1Problem & p);

// A partial is {max, sum, unnormalized output[dv]} for one row over a task's cells;
// rows of a KV head are ordered token-major: row = token * group + (query head % group).
inline int64_t argus_k1_partial_floats(const ArgusK1Problem & p) { return p.dv + 2; }

// Writes rows() partials for one task at partials + task * rows() * (dv + 2).
void argus_k1_task(const ArgusK1Problem & p, int64_t task, float * partials);

// Merges a KV head's chunk partials in chunk order and writes normalized rows. Output row of
// (token, head) is at out + head * out_head_stride + token * out_token_stride (bytes).
void argus_k1_reduce(const ArgusK1Problem & p, const float * partials, int64_t head_row0, int64_t head_row1,
                     char * out, size_t out_head_stride, size_t out_token_stride);

// Graph op: q [dk, n_head, n_tokens], k/v [d, n_head_kv, n_kv] in host memory, mask [n_kv, >= n_tokens].
// Returns nullptr when K1 does not apply (disabled, too many tokens, unsupported features or layout).
ggml_tensor * argus_ggml_decode_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                          ggml_tensor * mask, bool unsupported_features, float scale);
