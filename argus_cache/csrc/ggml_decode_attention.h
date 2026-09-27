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

// Writes the n_head_kv * rows() partials of one task (a chunk of cells across every KV head)
// at partials + task * n_head_kv * rows() * (dv + 2).
void argus_k1_task(const ArgusK1Problem & p, int64_t task, float * partials);

// Partials of several chunks laid out as [chunk][n_head_kv * rows()][dv + 2].
struct ArgusK1Partials { const float * data; int64_t chunks; };

// Merges every row's partials, part after part and chunk after chunk, and writes normalized
// rows. Output row of (token, head) is at out + head * out_head_stride + token *
// out_token_stride (bytes). Rows are numbered head_kv * rows() + token * group + head % group.
void argus_k1_reduce(const ArgusK1Problem & p, const ArgusK1Partials * parts, int n_parts, int64_t head_row0,
                     int64_t head_row1, char * out, size_t out_head_stride, size_t out_token_stride);

bool argus_k1_enabled();      // ARGUS_KV_CPU_ATTENTION=k1
bool argus_ggml_is_k1(const ggml_tensor * tensor); // a K1 node: must run on the CPU backend
int64_t argus_k1_max_tokens(); // ARGUS_K1_MAX_TOKENS, default 16: larger batches are prefill

// Graph op: q [dk, n_head, n_tokens], k/v [d, n_head_kv, n_kv] in host memory, mask [n_kv, >= n_tokens].
// Returns nullptr when K1 does not apply (disabled, too many tokens, unsupported features or layout).
ggml_tensor * argus_ggml_decode_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                          ggml_tensor * mask, bool unsupported_features, float scale);

// S1, CPU side: K1 over the host cells of a split call (k/v are views of those cells; cell c
// reads mask column mask_cell0 + c), merged after `prior` — the partials of the cells before
// them, one per row in K1's layout [dv + 2, n_head * n_tokens, chunks], or nullptr for none.
// Returns nullptr if unsupported.
ggml_tensor * argus_ggml_decode_attention_after(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                                ggml_tensor * mask, int64_t mask_cell0, ggml_tensor * prior, float scale);
