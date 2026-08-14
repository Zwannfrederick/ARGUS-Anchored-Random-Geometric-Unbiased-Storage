#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <string>

class ZeroCopyHostPool {
public:
    int device_id_;
    bool fallback_mode_;
    bool initialized_;
    size_t total_allocated_bytes_;
    size_t peak_allocated_bytes_;
    int gpu_numa_node_;
    int num_nodes_;
    bool is_multi_socket_;

private:

    // Allocations mapping: host_ptr -> (raw_ptr, size)
    std::unordered_map<void*, size_t> allocations_;
    std::unordered_map<void*, void*> device_pointers_;
    std::mutex mutex_;

    // Handle for dynamically loaded libnuma
    void* libnuma_handle_;
    typedef int (*numa_available_t)();
    typedef void (*numa_set_preferred_t)(int);
    numa_available_t numa_available_fn_;
    numa_set_preferred_t numa_set_preferred_fn_;

    void try_init();
    void detect_numa_topology();
    void bind_to_gpu_node();

public:
    ZeroCopyHostPool(int device_id = 0);
    ~ZeroCopyHostPool();

    void* allocate(size_t size_bytes);
    void free(void* ptr);
    void* get_device_pointer(void* host_ptr);

    // Tensor-level API
    torch::Tensor tensor_to_pinned(const torch::Tensor& tensor);
    void free_tensor(const torch::Tensor& tensor);
    bool is_zero_copy_tensor(const torch::Tensor& tensor);

    // Fragmentation report
    py::dict get_fragmentation_report();
};
