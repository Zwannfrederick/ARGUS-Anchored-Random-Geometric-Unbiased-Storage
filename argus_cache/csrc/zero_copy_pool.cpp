#include "zero_copy_pool.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/types.h>

namespace {
bool verbose_from_env() {
    const char* value = std::getenv("ARGUS_VERBOSE");
    return value != nullptr && std::string(value) == "1";
}
}  // namespace

ZeroCopyHostPool::ZeroCopyHostPool(int device_id)
    : device_id_(device_id),
      fallback_mode_(false),
      initialized_(false),
      total_allocated_bytes_(0),
      peak_allocated_bytes_(0),
      gpu_numa_node_(0),
      num_nodes_(1),
      is_multi_socket_(false),
      libnuma_handle_(nullptr),
      numa_available_fn_(nullptr),
      numa_set_preferred_fn_(nullptr) {
    detect_numa_topology();
    try_init();
}

ZeroCopyHostPool::~ZeroCopyHostPool() {
    // Clean up all live allocations
    for (auto& pair : allocations_) {
        cudaFreeHost(pair.first);
    }
    allocations_.clear();
    device_pointers_.clear();

    if (libnuma_handle_) {
        dlclose(libnuma_handle_);
    }
}

void ZeroCopyHostPool::detect_numa_topology() {
    // Detect number of NUMA nodes
    DIR* dir = opendir("/sys/devices/system/node");
    if (dir) {
        int count = 0;
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name(entry->d_name);
            if (name.rfind("node", 0) == 0) { // starts with "node"
                count++;
            }
        }
        closedir(dir);
        if (count > 0) {
            num_nodes_ = count;
        }
    }
    is_multi_socket_ = (num_nodes_ > 1);

    // Detect closest NUMA node to GPU
    DIR* pci_dir = opendir("/sys/bus/pci/devices");
    if (pci_dir) {
        struct dirent* entry;
        while ((entry = readdir(pci_dir)) != nullptr) {
            std::string device(entry->d_name);
            if (device == "." || device == "..") continue;
            
            std::string class_file = "/sys/bus/pci/devices/" + device + "/class";
            std::ifstream f(class_file);
            if (f.is_open()) {
                std::string cls;
                f >> cls;
                // 0x030200 (3D controller) or 0x030000 (VGA compatible)
                if (cls.rfind("0x0302", 0) == 0 || cls.rfind("0x0300", 0) == 0) {
                    std::string numa_file = "/sys/bus/pci/devices/" + device + "/numa_node";
                    std::ifstream nf(numa_file);
                    if (nf.is_open()) {
                        int node = -1;
                        nf >> node;
                        if (node >= 0) {
                            gpu_numa_node_ = node;
                            break;
                        }
                    }
                }
            }
        }
        closedir(pci_dir);
    }
}

void ZeroCopyHostPool::bind_to_gpu_node() {
    if (!is_multi_socket_) return;

    libnuma_handle_ = dlopen("libnuma.so.1", RTLD_GLOBAL | RTLD_NOW);
    if (!libnuma_handle_) {
        libnuma_handle_ = dlopen("libnuma.so", RTLD_GLOBAL | RTLD_NOW);
    }

    if (libnuma_handle_) {
        numa_available_fn_ = (numa_available_t)dlsym(libnuma_handle_, "numa_available");
        numa_set_preferred_fn_ = (numa_set_preferred_t)dlsym(libnuma_handle_, "numa_set_preferred");
        
        if (numa_available_fn_ && numa_set_preferred_fn_) {
            if (numa_available_fn_() >= 0) {
                numa_set_preferred_fn_(gpu_numa_node_);
                if (verbose_from_env()) {
                    std::cout << "[ARGUS ZeroCopyHostPool] NUMA preference set to node " << gpu_numa_node_ << std::endl;
                }
            }
        }
    }
}

void ZeroCopyHostPool::try_init() {
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        fallback_mode_ = true;
        if (verbose_from_env()) {
            std::cout << "[ARGUS ZeroCopyHostPool] cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        }
        return;
    }

    // Capability probe: Allocate a small page-locked mapped test block
    void* probe_ptr = nullptr;
    err = cudaHostAlloc(&probe_ptr, 65536, cudaHostAllocPortable | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fallback_mode_ = true;
        if (verbose_from_env()) {
            std::cout << "[ARGUS ZeroCopyHostPool] Host allocation capability probe failed: "
                      << cudaGetErrorString(err) << " -> Fallback to pin_memory mode." << std::endl;
        }
        return;
    }
    cudaFreeHost(probe_ptr);

    bind_to_gpu_node();
    initialized_ = true;
    if (verbose_from_env()) {
        std::cout << "[ARGUS ZeroCopyHostPool] Initialized successfully. Zero-Copy PCIe Active." << std::endl;
    }
}

void* ZeroCopyHostPool::allocate(size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fallback_mode_ || !initialized_) return nullptr;

    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, size_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        std::cerr << "[ARGUS ZeroCopyHostPool] cudaHostAlloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    void* dev_ptr = nullptr;
    err = cudaHostGetDevicePointer(&dev_ptr, ptr, 0);
    if (err != cudaSuccess) {
        cudaFreeHost(ptr);
        std::cerr << "[ARGUS ZeroCopyHostPool] cudaHostGetDevicePointer failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    allocations_[ptr] = size_bytes;
    device_pointers_[ptr] = dev_ptr;
    total_allocated_bytes_ += size_bytes;
    if (total_allocated_bytes_ > peak_allocated_bytes_) {
        peak_allocated_bytes_ = total_allocated_bytes_;
    }

    return ptr;
}

void ZeroCopyHostPool::free(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ptr) return;

    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        cudaFreeHost(ptr);
        total_allocated_bytes_ -= it->second;
        allocations_.erase(it);
        device_pointers_.erase(ptr);
    }
}

void* ZeroCopyHostPool::get_device_pointer(void* host_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_pointers_.find(host_ptr);
    if (it != device_pointers_.end()) {
        return it->second;
    }
    return nullptr;
}

torch::Tensor ZeroCopyHostPool::tensor_to_pinned(const torch::Tensor& tensor) {
    size_t nbytes = tensor.numel() * tensor.element_size();
    if (nbytes == 0) return tensor;

    void* host_ptr = allocate(nbytes);
    if (!host_ptr) {
        // Fallback to PyTorch pin_memory
        auto cpu_t = tensor.is_cuda() ? tensor.detach().cpu() : tensor.detach();
        return cpu_t.pin_memory();
    }

    // Create a CPU tensor viewing the pinned host memory
    auto options = torch::TensorOptions().device(torch::kCPU).dtype(tensor.dtype());
    auto pinned = torch::from_blob(host_ptr, tensor.sizes(), options);

    // Copy data
    if (tensor.is_cuda()) {
        cudaMemcpy(host_ptr, tensor.data_ptr(), nbytes, cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_ptr, tensor.data_ptr(), nbytes);
    }

    return pinned;
}

void ZeroCopyHostPool::free_tensor(const torch::Tensor& tensor) {
    if (tensor.is_cpu() && tensor.data_ptr()) {
        free(tensor.data_ptr());
    }
}

bool ZeroCopyHostPool::is_zero_copy_tensor(const torch::Tensor& tensor) {
    if (!tensor.is_cpu() || !tensor.data_ptr()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.find(tensor.data_ptr()) != allocations_.end();
}

py::dict ZeroCopyHostPool::get_fragmentation_report() {
    py::dict report;
    report["pool_total_allocated_bytes"] = total_allocated_bytes_;
    report["pool_peak_allocated_bytes"] = peak_allocated_bytes_;
    report["pool_num_allocations"] = allocations_.size();
    report["pytorch_allocated_bytes"] = 0;
    report["pytorch_reserved_bytes"] = 0;
    report["invisible_locked_bytes"] = total_allocated_bytes_;
    report["fragmentation_risk"] = "LOW";
    return report;
}
