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
    uint64_t allocation, generation;
    bool operator==(const ArgusDiskRevision & other) const {
        return allocation == other.allocation && generation == other.generation;
    }
};
ArgusDiskRevision argus_disk_revision(const ggml_tensor * tensor);

// One pending block. The explicitly allocated worker stack is charged to staging too.
class ArgusDiskPrefetch {
public:
    static constexpr size_t stack_bytes = 68 * 1024;
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
