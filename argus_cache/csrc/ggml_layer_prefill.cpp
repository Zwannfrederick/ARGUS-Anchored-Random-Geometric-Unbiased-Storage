// P3 layer-major prefill: the device cache of one layer's host-resident experts.
#include "ggml_layer_prefill.h"
#include "ggml_kv_tiers.h"
#include "argus_config.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#ifdef __linux__
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#ifndef MADV_COLD
#define MADV_COLD 20
#endif
#endif

namespace {
struct Cache {
    size_t bytes = 0;                                          // the largest layer's experts
    ggml_backend_dev_t dev = nullptr;                          // device for on-demand prefill arena allocation
    bool on_demand = false;                                    // true when buffer is leased only during prefill
    ggml_backend_buffer_t buffer = nullptr;                    // own device memory, when not borrowing the KV arena
    ggml_backend_buffer_t dummy_buffer = nullptr;              // tiny 4 KiB device buffer used only during sched_reserve()
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

bool moe_was_random_advised = false;
bool moe_is_file_mapped = false;
uintptr_t moe_vma_start = 0;
uintptr_t moe_vma_end = 0;
off_t moe_vma_off = 0;
std::string moe_vma_path;
int moe_total_layers = 0;
std::atomic<int> moe_pinned_layers{-1};
std::atomic<int64_t> last_rebalance_ms{0};

bool find_regular_file_vma(const void * ptr, uintptr_t & out_s, uintptr_t & out_e, off_t & out_off, std::string & out_path) {
#ifdef __linux__
    if (!ptr) { return false; }
    const uintptr_t target = reinterpret_cast<uintptr_t>(ptr);
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long long s = 0, e = 0, off = 0, inode = 0;
        char perms[8]{}, dev[16]{};
        int pos = 0;
        if (std::sscanf(line.c_str(), "%llx-%llx %7s %llx %15s %llu %n", &s, &e, perms, &off, dev, &inode, &pos) >= 6 &&
            target >= static_cast<uintptr_t>(s) && target < static_cast<uintptr_t>(e)) {
            if (inode == 0 || pos <= 0) { return false; }
            while (pos < int(line.size()) && (line[pos] == ' ' || line[pos] == '\t')) { ++pos; }
            if (pos >= int(line.size()) || line[pos] != '/') { return false; }
            const std::string path = line.substr(pos);
            if (path.compare(0, 5, "/dev/") == 0 || path.compare(0, 6, "/proc/") == 0 || path.compare(0, 5, "/sys/") == 0) {
                return false;
            }
            out_s = static_cast<uintptr_t>(s);
            out_e = static_cast<uintptr_t>(e);
            out_off = static_cast<off_t>(off);
            out_path = path;
            return true;
        }
    }
#else
    (void) ptr; (void) out_s; (void) out_e; (void) out_off; (void) out_path;
#endif
    return false;
}

bool is_regular_file_vma(const void * ptr) {
    uintptr_t s = 0, e = 0;
    off_t off = 0;
    std::string path;
    return find_regular_file_vma(ptr, s, e, off, path);
}

void advise_range(void * ptr, size_t len, bool willneed, bool cold, bool random_access, bool normal_access = false, bool seq_access = false) {
#ifdef __linux__
    if (!ptr || len == 0) { return; }
    static const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
    if (random_access || normal_access || seq_access || willneed) {
        const uintptr_t beg = raw & ~(page_sz - 1);
        const uintptr_t end = (raw + len + page_sz - 1) & ~(page_sz - 1);
        if (end > beg) {
            void * aligned_ptr = reinterpret_cast<void *>(beg);
            const size_t aligned_len = end - beg;
            if (normal_access) { (void) posix_madvise(aligned_ptr, aligned_len, POSIX_MADV_NORMAL); }
            if (seq_access)    { (void) posix_madvise(aligned_ptr, aligned_len, POSIX_MADV_SEQUENTIAL); }
            if (random_access) { (void) posix_madvise(aligned_ptr, aligned_len, POSIX_MADV_RANDOM); }
            if (willneed)      { (void) posix_madvise(aligned_ptr, aligned_len, POSIX_MADV_WILLNEED); }
        }
    }
    if (cold) {
        const uintptr_t beg_in = (raw + page_sz - 1) & ~(page_sz - 1);
        const uintptr_t end_in = (raw + len) & ~(page_sz - 1);
        if (end_in > beg_in) {
            (void) madvise(reinterpret_cast<void *>(beg_in), end_in - beg_in, MADV_COLD);
        }
    }
#else
    (void) ptr; (void) len; (void) willneed; (void) cold; (void) random_access; (void) normal_access; (void) seq_access;
#endif
}

size_t query_mem_available_bytes() {
#ifdef __linux__
    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    size_t val_kb = 0;
    while (meminfo >> key >> val_kb) {
        std::getline(meminfo, unit);
        if (key == "MemAvailable:") { return val_kb << 10; }
    }
#endif
    return 0;
}

// Reclaim host OS page-cache pages in the GGUF mmap VMA that belong to GPU-offloaded weights
// or unused layers (already copied to VRAM during model load and never read on host again).
void reclaim_non_host_gguf_pages(const ArgusNamedTensors & model_tensors, int n_layer, const void * sample_host_ptr) {
#ifdef __linux__
    if (!sample_host_ptr) { return; }
    static uintptr_t last_reclaimed_vma = 0;
    static const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    uintptr_t vma_start = 0, vma_end = 0;
    off_t vma_off = 0;
    std::string vma_path;
    if (!find_regular_file_vma(sample_host_ptr, vma_start, vma_end, vma_off, vma_path) ||
        vma_end <= vma_start || vma_start == last_reclaimed_vma) {
        return;
    }
    last_reclaimed_vma = vma_start;
    moe_vma_start = vma_start;
    moe_vma_end   = vma_end;
    moe_vma_off   = vma_off;
    moe_vma_path  = vma_path;

    std::vector<std::pair<uintptr_t, uintptr_t>> host_ranges;
    for (const auto & [name, t] : model_tensors) {
        if (!t || !t->data || (t->buffer && !ggml_backend_buffer_is_host(t->buffer))) { continue; }
        // Exclude MTP / extra layers (>= n_layer) from active host ranges so their mmap pages are reclaimed.
        if (name.compare(0, 4, "blk.") == 0) {
            const int il = std::atoi(name.c_str() + 4);
            if (il >= n_layer) { continue; }
        }
        const uintptr_t raw = reinterpret_cast<uintptr_t>(t->data);
        const size_t len = ggml_nbytes(t);
        if (raw >= vma_start && raw + len <= vma_end) {
            const uintptr_t beg = raw & ~(page_sz - 1);
            const uintptr_t end = (raw + len + page_sz - 1) & ~(page_sz - 1);
            host_ranges.emplace_back(beg, end);
        }
    }
    std::sort(host_ranges.begin(), host_ranges.end());
    std::vector<std::pair<uintptr_t, uintptr_t>> merged;
    for (const auto & r : host_ranges) {
        if (merged.empty() || r.first > merged.back().second) {
            merged.push_back(r);
        } else {
            merged.back().second = std::max(merged.back().second, r.second);
        }
    }

    const int fd = !vma_path.empty() ? open(vma_path.c_str(), O_RDONLY) : -1;
    uintptr_t cur = vma_start;
    size_t reclaimed_bytes = 0;
    for (const auto & r : merged) {
        if (r.first > cur && r.first - cur >= page_sz) {
            const size_t gap = r.first - cur;
            (void) madvise(reinterpret_cast<void *>(cur), gap, MADV_DONTNEED);
            if (fd >= 0) {
                (void) posix_fadvise(fd, vma_off + static_cast<off_t>(cur - vma_start), static_cast<off_t>(gap), POSIX_FADV_DONTNEED);
            }
            reclaimed_bytes += gap;
        }
        cur = std::max(cur, r.second);
    }
    if (vma_end > cur && vma_end - cur >= page_sz) {
        const size_t gap = vma_end - cur;
        (void) madvise(reinterpret_cast<void *>(cur), gap, MADV_DONTNEED);
        if (fd >= 0) {
            (void) posix_fadvise(fd, vma_off + static_cast<off_t>(cur - vma_start), static_cast<off_t>(gap), POSIX_FADV_DONTNEED);
        }
        reclaimed_bytes += gap;
    }
    if (fd >= 0) { close(fd); }
    if (reclaimed_bytes > 0) {
        std::fprintf(stderr, "ARGUS Hetero Mem: reclaimed %.1f MiB of non-active GGUF pages\n", reclaimed_bytes / 1048576.0);
    }
#else
    (void) model_tensors; (void) n_layer; (void) sample_host_ptr;
#endif
}

void drop_tensor_file_pages(const ggml_tensor * t, int gguf_fd) {
#ifdef __linux__
    if (!t || !t->data) { return; }
    static const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t raw = reinterpret_cast<uintptr_t>(t->data);
    const size_t len = ggml_nbytes(t);
    const uintptr_t beg = (raw + page_sz - 1) & ~(page_sz - 1);
    const uintptr_t end = (raw + len) & ~(page_sz - 1);
    if (end <= beg) { return; }
    (void) madvise(reinterpret_cast<void *>(beg), end - beg, MADV_DONTNEED);
    if (gguf_fd >= 0 && beg >= moe_vma_start && end <= moe_vma_end) {
        const off_t f_off = moe_vma_off + static_cast<off_t>(beg - moe_vma_start);
        (void) posix_fadvise(gguf_fd, f_off, static_cast<off_t>(end - beg), POSIX_FADV_DONTNEED);
    }
#else
    (void) t; (void) gguf_fd;
#endif
}
} // namespace

bool argus_layer_prefill_requested() {
    return argus_get_config().layer_prefill;
}

namespace {
std::atomic<uint64_t> stat_p3_engaged_calls{0};
} // namespace

uint64_t argus_p3_engaged_calls() {
    return stat_p3_engaged_calls.load(std::memory_order_relaxed);
}

bool argus_p3_should_engage(uint32_t n_tokens_all, uint32_t n_ubatch) {
    if (!argus_layer_prefill_requested() || !argus_expert_cache_ready()) {
        return false;
    }
    const ArgusConfig cfg = argus_get_config();
    uint32_t eff_min = static_cast<uint32_t>(std::max<int64_t>(1, cfg.p3_min_tokens));
    double residency = 1.0;
    if (cfg.p3_adaptive) {
        if (n_tokens_all < 64) {
            return false;
        }
        if (n_tokens_all < eff_min && moe_is_file_mapped) {
            size_t res_pages = 0, tot_pages = 0;
            for (int probe : {0, 13, 26, 39}) {
                const ArgusMoeLayerResidency r = argus_moe_layer_residency(probe);
                res_pages += r.resident_pages;
                tot_pages += r.total_pages;
            }
            if (tot_pages > 0) {
                residency = static_cast<double>(res_pages) / static_cast<double>(tot_pages);
                if (residency < 0.50) {
                    eff_min = 64;
                }
            }
        }
    }
    if (n_tokens_all < eff_min) {
        return false;
    }
    stat_p3_engaged_calls.fetch_add(1, std::memory_order_relaxed);
    if (cfg.p3_log) {
        std::fprintf(stderr,
                     "ARGUS P3: batch engaged (n_tokens=%u, ubatch=%u, min_tok=%u, residency=%.0f%%%s)\n",
                     n_tokens_all, n_ubatch, eff_min, residency * 100.0, cfg.p3_adaptive ? ", adaptive" : "");
    }
    return true;
}

std::vector<ggml_tensor *> argus_layer_experts(const ArgusNamedTensors & model_tensors, int layer) {
    std::vector<ggml_tensor *> experts;
    for (const auto & [name, tensor] : model_tensors) {
        if (is_layer_expert(name, layer) && tensor && tensor->data &&
            (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer))) {
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
    cache.dev = dev;
    // On stderr whatever the log level: runs can prove which prefill path they took.
    if (argus_kv_arena(bytes)) {  // VRAM shared by phase: experts during prefill, KV mirrors during decode
        cache.bytes = bytes;
        cache.on_demand = false;
        std::fprintf(stderr, "ARGUS P3: layer-major prefill, %.1f MiB expert cache sharing the KV mirrors' memory\n", bytes / 1048576.0);
        return true;
    }
    const ArgusConfig cfg = argus_get_config();
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_dev_memory(dev, &free_vram, &total_vram);
    if (cfg.vram_auto && total_vram > 0) {
        if (free_vram < bytes + cfg.vram_margin) {
            std::fprintf(stderr, "ARGUS P3: preserving %.1f MiB VRAM margin (free %.1f MiB, need %.1f MiB): batch-major prefill\n",
                         cfg.vram_margin / 1048576.0, free_vram / 1048576.0, (bytes + cfg.vram_margin) / 1048576.0);
            return false;
        }
        cache.bytes = bytes;
        cache.on_demand = true;
        std::fprintf(stderr, "ARGUS P3: layer-major prefill, %.1f MiB on-demand expert cache\n", bytes / 1048576.0);
        return true;
    }
    cache.buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), bytes);
    if (!cache.buffer) {
        std::fprintf(stderr, "ARGUS P3: no device memory for a %.1f MiB expert cache: batch-major prefill\n", bytes / 1048576.0);
        return false;
    }
    cache.bytes = bytes;
    cache.on_demand = false;
    std::fprintf(stderr, "ARGUS P3: layer-major prefill, %.1f MiB expert cache\n", bytes / 1048576.0);
    return true;
}

bool argus_expert_cache_ready() {
    std::lock_guard<std::mutex> guard(cache_lock);
    if (!cache.bytes) { return false; }
    if (cache.buffer || argus_kv_arena(cache.bytes)) { return true; }
    if (cache.on_demand && cache.dev) {
        size_t free_vram = 0, total_vram = 0;
        ggml_backend_dev_memory(cache.dev, &free_vram, &total_vram);
        return total_vram == 0 || free_vram >= cache.bytes + argus_get_config().vram_margin;
    }
    return false;
}

void argus_expert_cache_prefetch(const std::vector<ggml_tensor *> & experts) {
#ifdef __linux__
    if (!moe_is_file_mapped || experts.empty()) { return; }
    for (ggml_tensor * host : experts) {
        if (host && host->data) {
            advise_range(host->data, ggml_nbytes(host), true, false, false, true, false);
        }
    }
#else
    (void) experts;
#endif
}

void argus_expert_cache_load(const std::vector<ggml_tensor *> & experts, int layer_idx) {
    std::lock_guard<std::mutex> guard(cache_lock);
    if (cache.ctx) { ggml_free(cache.ctx); cache.ctx = nullptr; }
    cache.copies.clear();
    if (experts.empty()) { return; }
    if (!cache.buffer && cache.on_demand && cache.dev && !argus_kv_arena(cache.bytes)) {
        cache.buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(cache.dev), cache.bytes);
    }
    ggml_backend_buffer_t buffer = cache.buffer ? cache.buffer : argus_kv_arena(cache.bytes);
    if (!buffer) { throw std::logic_error("ARGUS expert cache is not reserved"); }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    cache.ctx = ggml_init({ggml_tensor_overhead() * (experts.size() + 1), nullptr, true});
    const size_t alignment = ggml_backend_buffer_get_alignment(buffer);
    char * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
    size_t offset = 0;
#ifdef __linux__
    const int pinned_k = moe_pinned_layers.load(std::memory_order_relaxed);
    const bool release_overflow = moe_is_file_mapped && pinned_k >= 0 && layer_idx >= pinned_k;
    const int ovf_fd = (release_overflow && !moe_vma_path.empty()) ? open(moe_vma_path.c_str(), O_RDONLY) : -1;
#endif
    for (ggml_tensor * host : experts) {
        ggml_tensor * copy = ggml_dup_tensor(cache.ctx, host);
        const size_t nbytes = ggml_nbytes(host);
        if (offset + nbytes > ggml_backend_buffer_get_size(buffer)) {
            throw std::runtime_error("ARGUS expert cache is smaller than a layer's experts");
        }
        if (ggml_backend_tensor_alloc(buffer, copy, base + offset) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ARGUS expert cache allocation failed");
        }
#ifdef __linux__
        if (moe_is_file_mapped && moe_was_random_advised) {
            advise_range(host->data, nbytes, true, false, false, true, false);
        }
#endif
        ggml_backend_tensor_set(copy, host->data, 0, nbytes);
#ifdef __linux__
        if (release_overflow) {
            drop_tensor_file_pages(host, ovf_fd);
        } else if (moe_is_file_mapped && moe_was_random_advised) {
            advise_range(host->data, nbytes, false, false, true, false, false);
        }
#endif
        cache.copies.emplace(host, copy);
        offset += aligned(nbytes, alignment);
    }
#ifdef __linux__
    if (ovf_fd >= 0) { close(ovf_fd); }
#endif
}

void argus_expert_cache_unload() {
    std::lock_guard<std::mutex> guard(cache_lock);
    cache.copies.clear();
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    if (cache.buffer && cache.on_demand) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    } else if (ggml_backend_buffer_t buffer = cache.buffer ? cache.buffer : argus_kv_arena(cache.bytes)) {
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_ANY);
    }
}

void argus_expert_cache_bind_dummy(const ArgusNamedTensors & model_tensors, int n_layer, ggml_backend_dev_t dev) {
    if (!dev || n_layer <= 0) { return; }
    std::vector<ggml_tensor *> all_experts;
    for (int layer = 0; layer < n_layer; ++layer) {
        for (ggml_tensor * t : argus_layer_experts(model_tensors, layer)) {
            all_experts.push_back(t);
        }
    }
    if (all_experts.empty()) { return; }
    std::lock_guard<std::mutex> guard(cache_lock);
    if (!cache.dummy_buffer) {
        cache.dummy_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), 4096);
        if (!cache.dummy_buffer) { return; }
        ggml_backend_buffer_set_usage(cache.dummy_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    if (cache.ctx) { ggml_free(cache.ctx); }
    cache.copies.clear();
    cache.ctx = ggml_init({ggml_tensor_overhead() * (all_experts.size() + 1), nullptr, true});
    void * base = ggml_backend_buffer_get_base(cache.dummy_buffer);
    for (ggml_tensor * host : all_experts) {
        ggml_tensor * copy = ggml_dup_tensor(cache.ctx, host);
        copy->buffer = cache.dummy_buffer;
        copy->data   = base;
        cache.copies.emplace(host, copy);
    }
}

void argus_expert_cache_unbind_dummy() {
    std::lock_guard<std::mutex> guard(cache_lock);
    cache.copies.clear();
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    if (cache.dummy_buffer) {
        ggml_backend_buffer_free(cache.dummy_buffer);
        cache.dummy_buffer = nullptr;
    }
}

ggml_tensor * argus_expert_tensor(ggml_tensor * tensor) {
    if (!tensor) { return tensor; }
    std::lock_guard<std::mutex> guard(cache_lock);
    const auto it = cache.copies.find(tensor);
    return it == cache.copies.end() ? tensor : it->second;
}

namespace {
constexpr int max_moe_layers = 128;
constexpr int max_moe_experts = 512;

struct MoeLayerState {
    std::vector<ggml_tensor *> tensors;
    int n_experts = 0;
    int hot_count = 0;
    std::array<uint32_t, max_moe_experts> freq{};
    std::array<uint8_t, max_moe_experts> is_hot{};
    std::array<uint8_t, max_moe_experts> is_locked{};
};

std::mutex moe_lock;
std::array<MoeLayerState, max_moe_layers> moe_layers;
int moe_max_hot_per_layer = 32;
size_t moe_mlock_budget = 0;
size_t moe_mlock_used = 0;

std::atomic<uint64_t> stat_moe_routed{0};
std::atomic<uint64_t> stat_moe_hot_hits{0};
std::atomic<uint64_t> stat_moe_promotions{0};
std::atomic<uint64_t> stat_moe_demotions{0};
std::atomic<uint64_t> stat_moe_advised_tensors{0};
std::atomic<uint64_t> stat_moe_advised_bytes{0};
std::atomic<uint64_t> stat_moe_locked_bytes{0};

bool try_lock_expert_slice(void * ptr, size_t len, bool lock) {
#ifdef __linux__
    if (!ptr || len == 0) { return false; }
    static const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
    // Inward page rounding guarantees disjoint page ranges across adjacent expert slices.
    const uintptr_t beg = (raw + page_sz - 1) & ~(page_sz - 1);
    const uintptr_t end = (raw + len) & ~(page_sz - 1);
    if (end <= beg) { return false; }
    const size_t aligned_len = end - beg;
    if (lock) {
        if (moe_mlock_used + aligned_len > moe_mlock_budget) { return false; }
        if (mlock(reinterpret_cast<const void *>(beg), aligned_len) == 0) {
            moe_mlock_used += aligned_len;
            stat_moe_locked_bytes.store(moe_mlock_used, std::memory_order_relaxed);
            return true;
        }
        return false;
    }
    if (munlock(reinterpret_cast<const void *>(beg), aligned_len) == 0 && moe_mlock_used >= aligned_len) {
        moe_mlock_used -= aligned_len;
        stat_moe_locked_bytes.store(moe_mlock_used, std::memory_order_relaxed);
    }
    return true;
#else
    (void) ptr; (void) len; (void) lock;
    return false;
#endif
}

void promote_expert(MoeLayerState & st, int e) {
    bool locked_any = false;
    if (moe_mlock_used < moe_mlock_budget) {
        for (ggml_tensor * t : st.tensors) {
            if (!t || !t->data || e >= t->ne[2]) { continue; }
            char * slice = static_cast<char *>(t->data) + static_cast<size_t>(e) * t->nb[2];
            const size_t slice_bytes = t->nb[2];
            if (try_lock_expert_slice(slice, slice_bytes, true)) { locked_any = true; }
        }
    }
    st.is_hot[e] = 1;
    st.is_locked[e] = locked_any ? 1 : 0;
    ++st.hot_count;
    stat_moe_promotions.fetch_add(1, std::memory_order_relaxed);
}

void demote_expert(MoeLayerState & st, int e) {
    if (st.is_locked[e]) {
        for (ggml_tensor * t : st.tensors) {
            if (!t || !t->data || e >= t->ne[2]) { continue; }
            char * slice = static_cast<char *>(t->data) + static_cast<size_t>(e) * t->nb[2];
            const size_t slice_bytes = t->nb[2];
            (void) try_lock_expert_slice(slice, slice_bytes, false);
        }
    }
    st.is_hot[e] = 0;
    st.is_locked[e] = 0;
    --st.hot_count;
    stat_moe_demotions.fetch_add(1, std::memory_order_relaxed);
}

void unlock_all_moe_slices_locked() {
    for (auto & st : moe_layers) {
        for (int e = 0; e < st.n_experts && e < max_moe_experts; ++e) {
            if (st.is_hot[e]) { demote_expert(st, e); }
        }
        st.freq.fill(0);
        st.is_hot.fill(0);
        st.is_locked.fill(0);
        st.hot_count = 0;
    }
    moe_mlock_used = 0;
    stat_moe_locked_bytes.store(0, std::memory_order_relaxed);
    stat_moe_demotions.store(0, std::memory_order_relaxed);
}

void moe_routing_custom_op(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
                           int ith, int nth, void * userdata) {
    (void) dst; (void) a; (void) nth;
    if (ith != 0 || !b || !b->data || b->type != GGML_TYPE_I32) { return; }
    const int il = static_cast<int>(reinterpret_cast<intptr_t>(userdata));
    if (il < 0 || il >= max_moe_layers) { return; }
    std::lock_guard<std::mutex> guard(moe_lock);
    MoeLayerState & st = moe_layers[il];
    if (st.n_experts <= 0) { return; }

    // Memory Pressure Controller: periodically release mlock slices if host RAM pressure rises.
    static uint32_t pressure_tick = 0;
    if ((++pressure_tick & 63u) == 0 && moe_mlock_used > 0) {
        const ArgusHardwareProfile hw = argus_probe_hardware();
        if (hw.ram_pressure >= ARGUS_PRESSURE_ELEVATED) {
            unlock_all_moe_slices_locked();
            moe_mlock_budget = 0;
        }
    }

    const int64_t n_used = b->ne[0], n_tok = b->ne[1];
    uint64_t routed = 0, hits = 0;
    for (int64_t tok = 0; tok < n_tok; ++tok) {
        const char * row = static_cast<const char *>(b->data) + tok * b->nb[1];
        for (int64_t k = 0; k < n_used; ++k) {
            const int32_t e = *reinterpret_cast<const int32_t *>(row + k * b->nb[0]);
            if (e < 0 || e >= st.n_experts || e >= max_moe_experts) { continue; }
            ++routed;
            const uint32_t f = ++st.freq[e];
            if (st.is_hot[e]) {
                ++hits;
                continue;
            }
            if (moe_max_hot_per_layer <= 0 || f < 2) { continue; }
            if (st.hot_count < moe_max_hot_per_layer) {
                promote_expert(st, e);
            } else {
                int victim = -1;
                uint32_t min_f = f;
                for (int cand = 0; cand < st.n_experts; ++cand) {
                    if (st.is_hot[cand] && st.freq[cand] < min_f) {
                        min_f = st.freq[cand];
                        victim = cand;
                    }
                }
                if (victim >= 0 && f >= min_f + 3) {
                    demote_expert(st, victim);
                    promote_expert(st, e);
                }
            }
        }
    }
    if (routed) { stat_moe_routed.fetch_add(routed, std::memory_order_relaxed); }
    if (hits)   { stat_moe_hot_hits.fetch_add(hits, std::memory_order_relaxed); }
}
std::atomic<int> active_decodes{0};

void warm_layer_active_pages(int il) {
#ifdef __linux__
    static const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    volatile unsigned char sink = 0;
    for (ggml_tensor * t : moe_layers[il].tensors) {
        if (!t || !t->data) { continue; }
        const size_t nbytes = ggml_nbytes(t);
        advise_range(t->data, nbytes, true, false, false, true, false);
        const unsigned char * p = static_cast<const unsigned char *>(t->data);
        for (size_t off = 0; off < nbytes; off += page_sz) {
            sink ^= p[off];
        }
    }
    (void) sink;
#else
    (void) il;
#endif
}

void rebalance_ram_locked(bool force) {
#ifdef __linux__
    if (!moe_is_file_mapped || moe_total_layers <= 0) { return; }
    const ArgusConfig cfg = argus_get_config();
    if (!cfg.dynamic_ram) { return; }
    const size_t avail = query_mem_available_bytes();
    const size_t reserve = cfg.ram_reserve_bytes;
    const size_t usable = (avail > reserve) ? (avail - reserve) : 0;

    size_t cum_bytes = 0, total_model_bytes = 0, pinned_bytes = 0;
    int target_k = 0;
    for (int il = 0; il < moe_total_layers; ++il) {
        size_t lbytes = 0;
        for (ggml_tensor * t : moe_layers[il].tensors) {
            if (t && t->data) { lbytes += ggml_nbytes(t); }
        }
        total_model_bytes += lbytes;
        if (lbytes > 0 && cum_bytes + lbytes <= usable) {
            cum_bytes += lbytes;
            target_k = il + 1;
            pinned_bytes = cum_bytes;
        }
    }

    const int prev_k = moe_pinned_layers.load(std::memory_order_relaxed);
    if (prev_k < 0) {
        const int fd = !moe_vma_path.empty() ? open(moe_vma_path.c_str(), O_RDONLY) : -1;
        for (int il = target_k; il < moe_total_layers; ++il) {
            for (ggml_tensor * t : moe_layers[il].tensors) { drop_tensor_file_pages(t, fd); }
        }
        if (fd >= 0) { close(fd); }
        for (int il = 0; il < target_k; ++il) { warm_layer_active_pages(il); }
        moe_pinned_layers.store(target_k, std::memory_order_relaxed);
        std::fprintf(stderr,
                     "ARGUS Dynamic RAM: %d/%d MoE layers active in RAM (%.1f / %.1f MiB), MemAvailable=%.1f MiB (reserve=%.1f MiB)\n",
                     target_k, moe_total_layers, pinned_bytes / 1048576.0, total_model_bytes / 1048576.0,
                     avail / 1048576.0, reserve / 1048576.0);
        return;
    }

    if (target_k < prev_k) {
        const int fd = !moe_vma_path.empty() ? open(moe_vma_path.c_str(), O_RDONLY) : -1;
        for (int il = target_k; il < prev_k; ++il) {
            for (ggml_tensor * t : moe_layers[il].tensors) { drop_tensor_file_pages(t, fd); }
        }
        if (fd >= 0) { close(fd); }
        moe_pinned_layers.store(target_k, std::memory_order_relaxed);
        std::fprintf(stderr,
                     "ARGUS Dynamic RAM: yielded %d layers -> %d/%d in RAM (%.1f MiB, MemAvailable=%.1f MiB)\n",
                     prev_k - target_k, target_k, moe_total_layers, pinned_bytes / 1048576.0, avail / 1048576.0);
    } else if (target_k > prev_k) {
        const bool has_deadband = force || target_k == moe_total_layers || (usable >= pinned_bytes + (256ULL << 20));
        if (has_deadband) {
            for (int il = prev_k; il < target_k; ++il) { warm_layer_active_pages(il); }
            moe_pinned_layers.store(target_k, std::memory_order_relaxed);
            std::fprintf(stderr,
                         "ARGUS Dynamic RAM: expanded +%d layers -> %d/%d in RAM (%.1f MiB, MemAvailable=%.1f MiB)\n",
                         target_k - prev_k, target_k, moe_total_layers, pinned_bytes / 1048576.0, avail / 1048576.0);
        }
    }
#else
    (void) force;
#endif
}

struct DynamicRamWatchdog {
    std::mutex m;
    std::condition_variable cv;
    std::thread worker;
    bool running = false;
    bool stop = false;

    void start() {
#ifdef __linux__
        std::lock_guard<std::mutex> lk(m);
        if (running) { return; }
        running = true;
        worker = std::thread([this]() {
            std::unique_lock<std::mutex> wlk(m);
            while (!cv.wait_for(wlk, std::chrono::milliseconds(1500), [this]() { return stop; })) {
                wlk.unlock();
                if (moe_is_file_mapped && moe_pinned_layers.load(std::memory_order_relaxed) >= 0) {
                    const bool busy = active_decodes.load(std::memory_order_relaxed) > 0;
                    if (!busy || query_mem_available_bytes() < (argus_get_config().ram_reserve_bytes * 3) / 4) {
                        std::unique_lock<std::mutex> mlk(moe_lock, std::try_to_lock);
                        if (mlk.owns_lock()) { rebalance_ram_locked(false); }
                    }
                }
                wlk.lock();
            }
        });
#endif
    }

    ~DynamicRamWatchdog() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) { worker.join(); }
    }
} ram_watchdog;
} // namespace

void argus_dynamic_ram_rebalance(bool force) {
    if (!moe_is_file_mapped) { return; }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const int64_t prev_ms = last_rebalance_ms.load(std::memory_order_relaxed);
    if (!force && (now_ms - prev_ms) < 500) { return; }
    last_rebalance_ms.store(now_ms, std::memory_order_relaxed);
    std::lock_guard<std::mutex> guard(moe_lock);
    rebalance_ram_locked(force);
}

ArgusDecodeScope::ArgusDecodeScope(uint32_t n_tokens) {
    if (active_decodes.fetch_add(1, std::memory_order_acq_rel) == 0) {
        argus_dynamic_ram_rebalance(false);
    }
#ifdef __linux__
    const int k = moe_pinned_layers.load(std::memory_order_relaxed);
    if (n_tokens >= 32 && moe_is_file_mapped && k >= 0 && k < moe_total_layers) {
        std::lock_guard<std::mutex> guard(moe_lock);
        for (int il = k; il < std::min(moe_total_layers, k + 2); ++il) {
            argus_expert_cache_prefetch(moe_layers[il].tensors);
        }
    }
#else
    (void) n_tokens;
#endif
}

ArgusDecodeScope::~ArgusDecodeScope() {
    active_decodes.fetch_sub(1, std::memory_order_acq_rel);
}

void argus_expert_cache_prepare_p3() {
#ifdef __linux__
    const int k = moe_pinned_layers.load(std::memory_order_relaxed);
    if (!moe_is_file_mapped || k < 0 || k >= moe_total_layers) { return; }
    std::lock_guard<std::mutex> guard(moe_lock);
    if (k + 2 < moe_total_layers) {
        const int fd = !moe_vma_path.empty() ? open(moe_vma_path.c_str(), O_RDONLY) : -1;
        for (int il = k + 2; il < moe_total_layers; ++il) {
            for (ggml_tensor * t : moe_layers[il].tensors) { drop_tensor_file_pages(t, fd); }
        }
        if (fd >= 0) { close(fd); }
    }
    for (int il = k; il < std::min(moe_total_layers, k + 2); ++il) {
        argus_expert_cache_prefetch(moe_layers[il].tensors);
    }
#endif
}

void argus_moe_init_host_experts(const ArgusNamedTensors & model_tensors, int n_layer) {
    const ArgusConfig cfg = argus_get_config();
    bool start_watchdog = false;
    {
        std::lock_guard<std::mutex> guard(moe_lock);
        unlock_all_moe_slices_locked();
        const int layers = std::min(n_layer, max_moe_layers);
        moe_total_layers = layers;
        moe_pinned_layers.store(-1, std::memory_order_relaxed);
        const void * sample_host_ptr = nullptr;
        ggml_backend_buffer_t sample_buf = nullptr;
        for (int il = 0; il < layers; ++il) {
            MoeLayerState & st = moe_layers[il];
            st = MoeLayerState{};
            st.tensors = argus_layer_experts(model_tensors, il);
            for (ggml_tensor * t : st.tensors) {
                if (t && t->data && t->ne[2] > 1) {
                    st.n_experts = std::max(st.n_experts, static_cast<int>(t->ne[2]));
                    if (!sample_host_ptr) { sample_host_ptr = t->data; sample_buf = t->buffer; }
                }
            }
        }
        moe_is_file_mapped = sample_host_ptr ? is_regular_file_vma(sample_host_ptr) : false;
        if (!moe_is_file_mapped && sample_buf != nullptr) {
            size_t count = 0, total_bytes = 0;
            for (int il = 0; il < layers; ++il) {
                for (ggml_tensor * t : moe_layers[il].tensors) {
                    if (t && t->data) { ++count; total_bytes += ggml_nbytes(t); }
                }
            }
            moe_was_random_advised = false;
            stat_moe_advised_tensors.store(count, std::memory_order_relaxed);
            stat_moe_advised_bytes.store(total_bytes, std::memory_order_relaxed);
            if (count > 0) {
                std::fprintf(stderr, "ARGUS Hetero Mem: %zu host expert tensors resident in active RAM (%.1f MiB, buffer=%s, zero SSD faults)\n",
                             count, total_bytes / 1048576.0, ggml_backend_buffer_name(sample_buf));
            }
            return;
        }
        if (sample_host_ptr && moe_is_file_mapped) { reclaim_non_host_gguf_pages(model_tensors, layers, sample_host_ptr); }
        if (!cfg.madv_random && !cfg.moe_cache_enabled) {
            const bool use_seq = (cfg.mmap_policy == 2);
            size_t count = 0, total_bytes = 0;
            for (int il = 0; il < layers; ++il) {
                for (ggml_tensor * t : moe_layers[il].tensors) {
                    if (t && t->data) {
                        const size_t nbytes = ggml_nbytes(t);
                        advise_range(t->data, nbytes, false, false, false, !use_seq, use_seq);
                        ++count;
                        total_bytes += nbytes;
                    }
                }
            }
            moe_was_random_advised = false;
            stat_moe_advised_tensors.store(count, std::memory_order_relaxed);
            stat_moe_advised_bytes.store(total_bytes, std::memory_order_relaxed);
            if (count > 0) {
                std::fprintf(stderr, "ARGUS MoE Cache: %s applied to %zu host expert tensors (%.1f MiB)\n",
                             use_seq ? "MADV_SEQUENTIAL" : "MADV_NORMAL", count, total_bytes / 1048576.0);
            }
            rebalance_ram_locked(true);
            start_watchdog = cfg.dynamic_ram && moe_is_file_mapped;
        } else {
            moe_max_hot_per_layer = cfg.moe_cache_enabled ? std::min(cfg.hot_experts_per_layer, max_moe_experts) : 0;
#ifdef __linux__
            struct rlimit rl{};
            if (cfg.moe_cache_enabled && getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur > (512ULL << 10)) {
                moe_mlock_budget = std::min<size_t>(rl.rlim_cur - (512ULL << 10), size_t(2) << 30);
            } else {
                moe_mlock_budget = 0;
            }
#endif
            size_t count = 0, total_bytes = 0;
            const bool apply_random = cfg.madv_random || cfg.moe_cache_enabled;
            for (int il = 0; il < layers; ++il) {
                for (ggml_tensor * t : moe_layers[il].tensors) {
                    if (!t || !t->data) { continue; }
                    const size_t nbytes = ggml_nbytes(t);
                    if (apply_random) { advise_range(t->data, nbytes, false, false, true, false); }
                    ++count;
                    total_bytes += nbytes;
                }
            }
            if (apply_random && count > 0) {
                moe_was_random_advised = true;
                stat_moe_advised_tensors.store(count, std::memory_order_relaxed);
                stat_moe_advised_bytes.store(total_bytes, std::memory_order_relaxed);
                std::fprintf(stderr, "ARGUS MoE Cache: MADV_RANDOM applied to %zu host expert tensors (%.1f MiB, top-%d hot experts/layer)\n",
                             count, total_bytes / 1048576.0, moe_max_hot_per_layer);
            }
        }
    }
    if (start_watchdog) { ram_watchdog.start(); }
}

ArgusMoeLayerResidency argus_moe_layer_residency(int il) {
    ArgusMoeLayerResidency res{};
    if (il < 0 || il >= max_moe_layers) { return res; }
    std::lock_guard<std::mutex> guard(moe_lock);
#ifdef __linux__
    static const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    std::vector<unsigned char> vec;
    for (ggml_tensor * t : moe_layers[il].tensors) {
        if (!t || !t->data) { continue; }
        const size_t len = ggml_nbytes(t);
        res.total_bytes += len;
        const uintptr_t raw = reinterpret_cast<uintptr_t>(t->data);
        const uintptr_t beg = raw & ~(page_sz - 1);
        const uintptr_t end = (raw + len + page_sz - 1) & ~(page_sz - 1);
        const size_t npages = (end - beg) / page_sz;
        if (npages == 0) { continue; }
        vec.assign(npages, 0);
        if (mincore(reinterpret_cast<void *>(beg), end - beg, vec.data()) == 0) {
            size_t in_ram = 0;
            for (unsigned char v : vec) { if (v & 1) { ++in_ram; } }
            res.total_pages += npages;
            res.resident_pages += in_ram;
            res.resident_bytes += in_ram * page_sz;
        }
    }
#endif
    return res;
}

void argus_moe_evict_layer_pages(int il) {
    if (il < 0 || il >= max_moe_layers) { return; }
    std::lock_guard<std::mutex> guard(moe_lock);
#ifdef __linux__
    const int fd = !moe_vma_path.empty() ? open(moe_vma_path.c_str(), O_RDONLY) : -1;
    for (ggml_tensor * t : moe_layers[il].tensors) { drop_tensor_file_pages(t, fd); }
    if (fd >= 0) { close(fd); }
#endif
}

ggml_tensor * argus_moe_observe_routing(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * selected_experts,
                                        ggml_tensor * up_exps, ggml_tensor * gate_exps,
                                        ggml_tensor * down_exps, ggml_tensor * gate_up_exps, int il) {
    if (!cur || !selected_experts || il < 0 || il >= max_moe_layers || !argus_get_config().moe_cache_enabled) {
        return cur;
    }
    {
        std::lock_guard<std::mutex> guard(moe_lock);
        MoeLayerState & st = moe_layers[il];
        if (st.tensors.empty()) {
            for (ggml_tensor * t : {up_exps, gate_exps, down_exps, gate_up_exps}) {
                if (t && t->data && (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) && t->ne[2] > 1) {
                    st.tensors.push_back(t);
                    st.n_experts = std::max(st.n_experts, static_cast<int>(t->ne[2]));
                }
            }
        }
        if (st.n_experts <= 0) { return cur; }
    }
    return ggml_map_custom2_inplace(ctx, cur, selected_experts, moe_routing_custom_op, 1,
                                    reinterpret_cast<void *>(static_cast<intptr_t>(il)));
}

ArgusMoeStats argus_moe_stats() {
    return {
        stat_moe_routed.load(std::memory_order_relaxed),
        stat_moe_hot_hits.load(std::memory_order_relaxed),
        stat_moe_promotions.load(std::memory_order_relaxed),
        stat_moe_demotions.load(std::memory_order_relaxed),
        stat_moe_advised_tensors.load(std::memory_order_relaxed),
        stat_moe_advised_bytes.load(std::memory_order_relaxed),
        stat_moe_locked_bytes.load(std::memory_order_relaxed),
    };
}

void argus_moe_reset_stats() {
    std::lock_guard<std::mutex> guard(moe_lock);
    stat_moe_routed.store(0, std::memory_order_relaxed);
    stat_moe_hot_hits.store(0, std::memory_order_relaxed);
    stat_moe_promotions.store(0, std::memory_order_relaxed);
    stat_moe_demotions.store(0, std::memory_order_relaxed);
    unlock_all_moe_slices_locked();
}
