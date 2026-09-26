#pragma once
// ARGUS disk-store internals shared by ggml_disk_buffer.cpp (store lifetime, GGML buffer,
// host I/O) and ggml_disk_gpu.cpp (GPU residency, appends, write-back). Not an API.
#include "ggml_disk_buffer.h"
#include "ggml-backend-impl.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <sys/mman.h>
#include <sys/types.h>
#ifdef ARGUS_CUDA
#include "ggml_cuda_attention.h"
#endif

namespace argus_disk {
constexpr size_t page_size = 4096;
struct Page {
    uint64_t content_revision = 0;
    uint64_t placement_revision = 0;
    uint32_t checksum = 0;
    uint32_t active = 0;
    ggml_type codec = GGML_TYPE_COUNT;
    uint64_t last_access_step = 0, access_count = 0;
#ifdef ARGUS_CUDA
    ArgusTierBuffer * resident = nullptr;
    // Written on the GPU: `checksum` is unknown until the first host read computes it.
    bool digest_pending = false;
    // The resident copy is newer than the disk slot; the flusher publishes it.
    bool dirty = false;
#endif
};
#ifdef ARGUS_CUDA
struct Flusher;
#endif
struct Store {
    void * address = MAP_FAILED;
    Page * pages = nullptr;
    size_t bytes = 0, disk_bytes = 0, metadata_bytes = 0, metadata_charge = 0;
    int fd = -1;
    uint64_t id = 0;
    uint64_t content_revision = 0;
    uint64_t access_step = 0;
    size_t last_written_page = SIZE_MAX; // Diagnostic write frontier for the residency census.
#ifdef ARGUS_CUDA
    Store * next = nullptr; // Intrusive registry: charged with the store metadata.
    bool gpu_control = false; // Diagnostic GPU-authoritative storage; no disk fallback.
    // Pinned upload + read-back staging for whole-page flushes; released with the store.
    std::unique_ptr<ArgusTierBuffer> commit_host;
    std::vector<size_t> dirty_pages; // may hold pages flushed since; `dirty` decides
    Flusher * flusher = nullptr;      // started by the first GPU append of a disk-backed store
    // GPU control, once every page is on the GPU: the device array of page addresses that
    // lets appends resolve their rows on the device. Dropped whenever a page is replaced.
    std::unique_ptr<ArgusTierBuffer> page_table;
    std::unique_ptr<ArgusTierBuffer> append_error; // pinned flag: a device append met a row outside its tensor
    void * write_event = nullptr;                    // after the last device append; host reads wait for it
    // Policy stores: page_table holds each page's GPU address or null, updated only by
    // publish_entry on table_stream, so kernels queued earlier keep the entries they saw.
    void * table_stream = nullptr;
    void * read_event = nullptr; // after the last attention that read the table
#endif
    std::mutex mutex;
    // Disk-slot I/O. `active` changes only with both locks held (taken in this order), so the
    // flusher can write a slot without holding `mutex` while attention reads pages.
    std::mutex io_mutex;
};
#ifdef ARGUS_CUDA
extern std::mutex registry_mutex;
extern Store * registry;
#endif
constexpr size_t descriptor_offset = (sizeof(Store) + alignof(Page) - 1) / alignof(Page) * alignof(Page);

// A page-aligned mapping also makes the physical staging allocation explicit.
struct Bounce {
    ArgusStagingBuffer buffer{page_size};
    void * data = buffer.data();
};

extern std::atomic<size_t> read_bytes, written_bytes, committed_pages;
constexpr size_t max_run_pages = 16;

size_t rounded(size_t bytes);
uint32_t checksum(const void * data);
off_t offset(size_t page, uint32_t active);
Page & page_at(Store & store, size_t index);
// Written content exists: a published slot, GPU control, or a dirty copy not yet flushed.
bool has_content(const Page & page);
void read_disk_page(Store & store, size_t page, void * data);
void read_page(Store & store, size_t page, void * data);
Store & store_for(ggml_backend_buffer_t buffer);
size_t checked_offset(const Store & store, const ggml_tensor * tensor, size_t start, size_t bytes);
size_t migration_page(const Store & store, const ggml_tensor * tensor, size_t start);
// Called under the store lock only after the entire public read succeeds.
void record_access(Store & store, size_t start, size_t bytes);
ArgusDiskPageDescriptor page_descriptor(const Store & store, size_t index);
#ifdef ARGUS_CUDA
// Defined in ggml_disk_gpu.cpp.
bool write_run(Store & store, size_t first, size_t count, const char * source);
void flush_page(Store & store, size_t index);
void destroy_flusher(Flusher * flusher);
// Policy stores with a page table: republish one page's GPU address (or null), in order on
// table_stream. Call after every change of a page's GPU residency. A replaced GPU page may
// then be freed: cudaFree waits for all queued device work, including earlier table reads.
void publish_entry(Store & store, size_t index);
#endif
} // namespace argus_disk
