#pragma once
#include "ggml_disk_buffer.h"

class ArgusTierBuffer {
public:
    ArgusTierBuffer(ArgusTier tier, size_t bytes);
    ~ArgusTierBuffer();
    ArgusTierBuffer(const ArgusTierBuffer &) = delete;
    ArgusTierBuffer & operator=(const ArgusTierBuffer &) = delete;
    void * data() const { return data_; }
    size_t size() const { return bytes_; }
    ArgusTier tier() const { return tier_; }
    void read(void * target, size_t offset, size_t bytes) const;
    void write(const void * source, size_t bytes);
private:
    ArgusTier tier_;
    size_t bytes_;
    void * data_ = nullptr;
};

struct ArgusTierUsage {
    size_t gpu, pinned, ram, peak_gpu, peak_pinned, peak_ram;
    size_t attention_calls;
};
ArgusTierUsage argus_tier_usage();
struct ArgusTierBudget { size_t limit, live; };
// An unset tier has zero capacity; allocation still enforces the configured budget.
ArgusTierBudget argus_tier_budget(ArgusTier tier);
// Callbacks inspect snapshots only and must not call storage APIs.
void argus_disk_visit_resident_pages(void (*visit)(const ArgusDiskPageDescriptor &, void *), void * context);
// Allocation identity is resolved under the store registry lock, including teardown races.
void argus_disk_move_page(ArgusDiskPageRevision expected, ArgusTier tier);
// One aligned physical page per transaction. A range may span several transactions.
void argus_disk_move_page(const ggml_tensor * tensor, size_t offset, ArgusTier tier,
                         ArgusDiskPageRevision expected);
// Source stays locked until the transfer stream completes. Host staging belongs to caller.
void argus_disk_stage_cuda(const ggml_tensor * tensor, void * host, void * device,
                           size_t offset, size_t bytes, void * stream);
void argus_cuda_copy(void * device, const void * source, size_t bytes, bool device_source, void * stream);
// GPU-control page commit. Uploads `count` pages of `page_bytes` from `source` to
// `targets[i]`, reads every one of them back, and waits once. `staging` is
// 2 * count * page_bytes of caller-owned pinned host memory. Returns the read-back
// bytes inside it for the caller to verify, or null when the commit stream is
// unavailable — the caller then keeps its per-page path.
const void * argus_cuda_commit_pages(void * staging, void * const * targets, const void * source,
                                     size_t count, size_t page_bytes);
void argus_cuda_wait(void * stream);

// Written pages that are not GPU-resident, copied for this one read only. Under the
// locks, reserve() receives their count and returns host space (4096 bytes each) or
// null to decline; upload() enqueues the checksum-verified host pages on the
// consumer's stream and returns their device copy. Placement, revisions and access
// history are untouched: temporary staging is not a promotion.
struct ArgusColdStaging {
    void * (*reserve)(size_t pages, void * context);
    const char * (*upload)(size_t pages, void * context);
};
// Callback borrows GPU pages under registry + both store locks. Null means logical
// zero, never a missing written page. It must drain GPU work before returning or
// throwing and must not call storage/policy APIs. False leaves access counts intact.
// Without cold staging any written non-GPU page declines the read.
bool argus_disk_read_resident(const ggml_tensor * k, const ggml_tensor * v,
    const void ** pages, size_t capacity,
    void (*consume)(size_t k_pages, size_t v_pages, void * context), void * context,
    const ArgusColdStaging * cold = nullptr);

ggml_tensor * argus_ggml_cuda_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                       ggml_tensor * v, ggml_tensor * mask, float scale);
bool argus_ggml_is_cuda_attention(const ggml_tensor * tensor);

// KV appends written on the GPU. Only for GPU-control (GPU-authoritative) stores whose
// rows never straddle a page.
bool argus_disk_gpu_appendable(const ggml_tensor * target);
// Under the store lock, resolves each row of `target` to its device address. Unwritten
// pages get zeroed GPU pages (queued on `stream`). New revisions are published with a
// pending digest, which the first host read computes. The caller must write the rows on
// `stream` before anything else reads those pages.
void argus_disk_gpu_rows(const ggml_tensor * target, const int64_t * rows, size_t count,
                         void ** destinations, void * stream);
void argus_cuda_zero(void * device, size_t bytes, void * stream);
ggml_tensor * argus_ggml_cuda_set_rows(ggml_context * ctx, ggml_tensor * target, ggml_tensor * source,
                                       ggml_tensor * indices);
bool argus_ggml_is_cuda_set_rows(const ggml_tensor * tensor);
