#pragma once

#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <pthread.h>

// O_DIRECT storage. Tensor addresses are inaccessible handles, never resident KV.
ggml_backend_buffer_type_t argus_ggml_disk_buffer_type();
bool argus_ggml_is_disk_tensor(const ggml_tensor * tensor);
bool argus_ggml_disk_enabled();

// Includes all ARGUS-owned attention scratch and I/O bounce buffers, process-wide.
class ArgusStagingReservation {
public:
    explicit ArgusStagingReservation(size_t bytes);
    ~ArgusStagingReservation();
    ArgusStagingReservation(const ArgusStagingReservation &) = delete;
    ArgusStagingReservation & operator=(const ArgusStagingReservation &) = delete;
private:
    size_t bytes_;
};

class ArgusStagingBuffer {
public:
    explicit ArgusStagingBuffer(size_t bytes);
    ~ArgusStagingBuffer();
    void * data() const { return data_; }
    size_t size() const { return bytes_; }
private:
    ArgusStagingReservation reservation_;
    size_t bytes_;
    void * data_;
};

struct ArgusDiskRevision {
    uint64_t allocation, content_revision;
    bool operator==(const ArgusDiskRevision & other) const {
        return allocation == other.allocation && content_revision == other.content_revision;
    }
};
// Store-wide content snapshot for attention/prefetch; placement does not invalidate it.
ArgusDiskRevision argus_disk_revision(const ggml_tensor * tensor);

// Migration tokens are bound to one physical page and ignore unrelated writes.
struct ArgusDiskPageRevision {
    uint64_t allocation;
    size_t page;
    uint64_t content_revision;
};
ArgusDiskPageRevision argus_disk_page_revision(const ggml_tensor * tensor, size_t offset);

enum class ArgusTier { disk, ram, pinned, gpu };

struct ArgusDiskPageDescriptor {
    ArgusDiskPageRevision revision;
    uint64_t placement_revision;
    ArgusTier placement;
    ggml_type codec; // GGML_TYPE_COUNT until a tensor owns the page; never converted here.
    uint64_t last_access_step, access_count;
    bool written;
};
// Snapshot of the physical page containing offset, including partial tail pages/views.
// Access steps are store-local successful read operations (including prefetch), not tokens.
// Counts saturate at UINT64_MAX; writes/migration preserve them, clear(0) resets them.
ArgusDiskPageDescriptor argus_disk_page_descriptor(const ggml_tensor * tensor, size_t offset);

// One pending block. The explicitly allocated worker stack is charged to staging too.
class ArgusDiskPrefetch {
public:
    // ponytail: 256 KiB stack covers tested CUDA TLS (~104 KiB); raise if linked TLS grows.
    static constexpr size_t stack_bytes = (256 + 4) * 1024; // includes guard page
    ArgusDiskPrefetch();
    ~ArgusDiskPrefetch();
    void submit(const ggml_tensor * k, const ggml_tensor * v, int64_t first, int64_t count, void * keys, void * values);
    void take();
private:
    struct Request {
        const ggml_tensor * k = nullptr, * v = nullptr;
        int64_t first = 0, count = 0;
        void * keys = nullptr, * values = nullptr;
        ArgusDiskRevision k_revision{}, v_revision{};
    } request_;
    ArgusStagingBuffer stack_{stack_bytes};
    pthread_t thread_{};
    std::mutex mutex_;
    std::condition_variable condition_;
    std::exception_ptr error_;
    bool stop_ = false, pending_ = false, ready_ = false, busy_ = false;
    static void * run(void * self);
};

size_t argus_disk_staging_limit();
void argus_disk_publish_stats();
ggml_tensor * argus_ggml_disk_set_rows(
    ggml_context * ctx, ggml_tensor * target, ggml_tensor * source, ggml_tensor * indices);
