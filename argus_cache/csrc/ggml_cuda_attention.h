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

// KV appends written on the GPU: GPU-control stores, and policy-managed stores with a
// GPU tier, whose rows never straddle a page.
bool argus_disk_gpu_appendable(const ggml_tensor * target);
// Under the store lock: gives every touched page a GPU copy (zeroed if unwritten,
// uploaded if its content lives elsewhere), calls `encode` with each row's device
// address (it must finish the writes before returning), then publishes the new
// revisions. The digest is pending until the first host read or flush computes it.
// Disk-backed pages become dirty and a background flusher publishes their disk slots.
// Returns false, touching nothing, when the GPU budget cannot hold the pages; the
// caller then appends through argus_disk_append_rows.
bool argus_disk_gpu_rows(const ggml_tensor * target, const int64_t * rows, size_t count, void * stream,
                         void (*encode)(void * const * destinations, void * context), void * context);
void argus_cuda_zero(void * device, size_t bytes, void * stream);
// GPU control, asynchronous: once the whole store is on the GPU (done here on first use,
// within budget), `launch` enqueues the encode on `stream` with the device page-address
// table, the tensor's byte offset in the store and a device-visible error flag it sets for
// a row outside the tensor; that is refused by the next append. Returns false, touching
// nothing, for other stores or when the budget cannot hold every page.
bool argus_disk_gpu_rows_on_device(const ggml_tensor * target, void * stream,
        void (*launch)(void * const * pages, size_t base, int * error, void * context), void * context);
// GPU control with the device page table built: tables for k and v, indexed from the page
// holding each tensor's first byte, and those bytes' offsets within their pages. Nothing is
// borrowed: replacing a page drops the table, and freeing GPU memory synchronizes first.
bool argus_disk_gpu_table(const ggml_tensor * k, const ggml_tensor * v, const void * const ** keys,
                          const void * const ** values, size_t * key_offset, size_t * value_offset);
// Policy stores whose written K/V pages are all on the GPU (and whose whole store, page table
// and scratch headroom fit the GPU budget): calls `launch` with page-address tables for k and
// v and their first-byte offsets, without waiting, then records the table's read event and
// the same access history as argus_disk_read_resident. False, touching nothing, otherwise or
// if `launch` declines.
bool argus_disk_policy_table_read(const ggml_tensor * k, const ggml_tensor * v, void * stream,
        bool (*launch)(const void * const * keys, const void * const * values, size_t key_offset,
                       size_t value_offset, void * context), void * context);
// Stores `value` into a device pointer slot, in order on `stream`.
void argus_cuda_store_pointer(void ** slot, void * value, void * stream);
void * argus_cuda_event_create();
void argus_cuda_event_record(void * event, void * stream);
void argus_cuda_event_wait(void * event);
void argus_cuda_event_destroy(void * event);
ggml_tensor * argus_ggml_cuda_set_rows(ggml_context * ctx, ggml_tensor * target, ggml_tensor * source,
                                       ggml_tensor * indices);
bool argus_ggml_is_cuda_set_rows(const ggml_tensor * tensor);
// Publishes every dirty page of the tensor's store to disk now (verified slots), and
// rethrows a background flush failure only if this synchronous attempt fails too.
void argus_disk_flush(const ggml_tensor * tensor);
