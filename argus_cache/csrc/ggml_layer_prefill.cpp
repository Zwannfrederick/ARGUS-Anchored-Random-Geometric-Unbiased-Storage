// P3 layer-major prefill: the device cache of one layer's host-resident experts.
#include "ggml_layer_prefill.h"
#include "ggml_kv_tiers.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace {
struct Cache {
    size_t bytes = 0;                                          // the largest layer's experts
    ggml_backend_buffer_t buffer = nullptr;                    // own device memory, when not borrowing the KV arena
    ggml_context * ctx = nullptr;                              // the loaded layer's copies
    std::unordered_map<const ggml_tensor *, ggml_tensor *> copies;
    // Lives until the process exits: freeing device memory from a static destructor would run
    // after the CUDA backend may be gone.
};

std::mutex cache_lock;
Cache cache;  // one per process: prefill runs one context at a time on this machine

bool is_layer_expert(const std::string & name, int layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    return name.compare(0, prefix.size(), prefix) == 0 && name.find("_exps") != std::string::npos;
}

size_t aligned(size_t bytes, size_t alignment) { return (bytes + alignment - 1) / alignment * alignment; }
} // namespace

bool argus_layer_prefill_requested() {
    const char * text = std::getenv("ARGUS_PREFILL");
    if (!text || !*text || std::string(text) == "batches") { return false; }
    if (std::string(text) == "layers") { return true; }
    throw std::runtime_error("ARGUS_PREFILL must be batches or layers");
}

std::vector<ggml_tensor *> argus_layer_experts(const ArgusNamedTensors & model_tensors, int layer) {
    std::vector<ggml_tensor *> experts;
    for (const auto & [name, tensor] : model_tensors) {
        if (is_layer_expert(name, layer) && tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer)) {
            experts.push_back(tensor);
        }
    }
    return experts;
}

size_t argus_expert_cache_bytes(const ArgusNamedTensors & model_tensors, int n_layer, ggml_backend_dev_t dev) {
    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev));
    size_t largest = 0;
    for (int layer = 0; layer < n_layer; ++layer) {
        size_t bytes = 0;
        for (ggml_tensor * t : argus_layer_experts(model_tensors, layer)) { bytes += aligned(ggml_nbytes(t), alignment); }
        largest = std::max(largest, bytes);
    }
    return largest;
}

bool argus_expert_cache_reserve(const ArgusNamedTensors & model_tensors, int n_layer, ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> guard(cache_lock);
    if (cache.bytes) { return true; }
    const size_t bytes = argus_expert_cache_bytes(model_tensors, n_layer, dev);
    if (!bytes) { return false; }
    // On stderr whatever the log level: runs can prove which prefill path they took.
    if (argus_kv_arena(bytes)) {  // VRAM shared by phase: experts during prefill, KV mirrors during decode
        cache.bytes = bytes;
        std::fprintf(stderr, "ARGUS P3: layer-major prefill, %.1f MiB expert cache sharing the KV mirrors' memory\n", bytes / 1048576.0);
        return true;
    }
    cache.buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), bytes);
    if (!cache.buffer) {
        std::fprintf(stderr, "ARGUS P3: no device memory for a %.1f MiB expert cache: batch-major prefill\n", bytes / 1048576.0);
        return false;
    }
    cache.bytes = bytes;
    std::fprintf(stderr, "ARGUS P3: layer-major prefill, %.1f MiB expert cache\n", bytes / 1048576.0);
    return true;
}

bool argus_expert_cache_ready() {
    std::lock_guard<std::mutex> guard(cache_lock);
    return cache.bytes && (cache.buffer || argus_kv_arena(cache.bytes));
}

void argus_expert_cache_load(const std::vector<ggml_tensor *> & experts) {
    std::lock_guard<std::mutex> guard(cache_lock);
    ggml_backend_buffer_t buffer = cache.buffer ? cache.buffer : argus_kv_arena(cache.bytes);
    if (!buffer) { throw std::logic_error("ARGUS expert cache is not reserved"); }
    if (cache.ctx) { ggml_free(cache.ctx); }
    cache.copies.clear();
    cache.ctx = ggml_init({ggml_tensor_overhead() * (experts.size() + 1), nullptr, true});
    const size_t alignment = ggml_backend_buffer_get_alignment(buffer);
    char * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
    size_t offset = 0;
    for (ggml_tensor * host : experts) {
        ggml_tensor * copy = ggml_dup_tensor(cache.ctx, host);
        if (offset + ggml_nbytes(host) > ggml_backend_buffer_get_size(buffer)) {
            throw std::runtime_error("ARGUS expert cache is smaller than a layer's experts");
        }
        if (ggml_backend_tensor_alloc(buffer, copy, base + offset) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ARGUS expert cache allocation failed");
        }
        ggml_backend_tensor_set(copy, host->data, 0, ggml_nbytes(host));
        cache.copies.emplace(host, copy);
        offset += aligned(ggml_nbytes(host), alignment);
    }
}

void argus_expert_cache_unload() {
    std::lock_guard<std::mutex> guard(cache_lock);
    cache.copies.clear();
}

ggml_tensor * argus_expert_tensor(ggml_tensor * tensor) {
    if (!tensor) { return tensor; }
    std::lock_guard<std::mutex> guard(cache_lock);
    const auto it = cache.copies.find(tensor);
    return it == cache.copies.end() ? tensor : it->second;
}
