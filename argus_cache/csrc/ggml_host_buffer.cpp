#include "ggml_host_buffer.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <mutex>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
struct Mapping {
    ggml_backend_buffer_t inner;
    void * address;
    size_t bytes;
    int fd;
};

std::mutex budget_mutex;
size_t live_bytes = 0;
std::vector<Mapping *> mappings;

std::atomic<size_t> total_read{0};
std::atomic<size_t> total_paged_out{0};
std::atomic<size_t> attention_calls{0};
std::atomic<size_t> peak_resident{0};
std::atomic<int64_t> last_publish_ms{0};

const size_t page_bytes = static_cast<size_t>(sysconf(_SC_PAGESIZE));

bool parse_bytes(const char * text, size_t & value) {
    if (!text || !*text) { return false; }
    for (const char * p = text; *p; ++p) {
        if (*p < '0' || *p > '9') { return false; }
    }
    errno = 0;
    char * end = nullptr;
    const auto parsed = std::strtoull(text, &end, 10);
    if (errno || *end || parsed == 0 || parsed > std::numeric_limits<size_t>::max()) { return false; }
    value = static_cast<size_t>(parsed);
    return true;
}

Mapping * mapping(ggml_backend_buffer_t buffer) {
    return static_cast<Mapping *>(buffer->context);
}

// Caller holds budget_mutex.
size_t resident_bytes_locked() {
    size_t resident = 0;
    std::vector<unsigned char> pages;
    for (const auto * state : mappings) {
        pages.resize((state->bytes + page_bytes - 1) / page_bytes);
        if (mincore(state->address, state->bytes, pages.data()) != 0) { continue; }
        for (const auto page : pages) { resident += (page & 1) * page_bytes; }
    }
    return std::min(resident, live_bytes);
}

void release(ggml_backend_buffer_t buffer) {
    auto * state = mapping(buffer);
    ggml_backend_buffer_free(state->inner);
    munmap(state->address, state->bytes);
    close(state->fd);
    {
        std::lock_guard<std::mutex> lock(budget_mutex);
        live_bytes -= state->bytes;
        mappings.erase(std::find(mappings.begin(), mappings.end(), state));
        std::fprintf(stderr,
            "ARGUS_KV free bytes=%zu live_bytes=%zu read_bytes=%zu paged_out_bytes=%zu "
            "attention_calls=%zu peak_resident_bytes=%zu\n",
            state->bytes, live_bytes, total_read.load(), total_paged_out.load(),
            attention_calls.load(), peak_resident.load());
    }
    delete state;
}

void * base(ggml_backend_buffer_t buffer) { return mapping(buffer)->address; }

void set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto inner = mapping(buffer)->inner;
    inner->iface.set_tensor(inner, tensor, data, offset, size);
    // State restore writes whole tensors; don't let it bypass the resident limit.
    argus_kv_page_out(static_cast<char *>(tensor->data) + offset, size);
}

void get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto inner = mapping(buffer)->inner;
    inner->iface.get_tensor(inner, tensor, data, offset, size);
    argus_kv_page_out(static_cast<char *>(tensor->data) + offset, size);
}

void fill(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto inner = mapping(buffer)->inner;
    inner->iface.memset_tensor(inner, tensor, value, offset, size);
    argus_kv_page_out(static_cast<char *>(tensor->data) + offset, size);
}

bool copy(ggml_backend_buffer_t buffer, const ggml_tensor * source, ggml_tensor * target) {
    auto inner = mapping(buffer)->inner;
    const bool copied = inner->iface.cpy_tensor(inner, source, target);
    if (copied) { argus_kv_page_out(target->data, ggml_nbytes(target)); }
    return copied;
}

void clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * state = mapping(buffer);
    const auto bytes = static_cast<off_t>(state->bytes);
    // Zeroing extents drops page cache and resident pages while keeping the storage
    // reservation, so a later write can't hit ENOSPC as SIGBUS.
    if (value == 0 && fallocate(state->fd, FALLOC_FL_ZERO_RANGE | FALLOC_FL_KEEP_SIZE, 0, bytes) == 0) {
        return;
    }
    if (value == 0 && fallocate(state->fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, 0, bytes) == 0) {
        if (posix_fallocate(state->fd, 0, bytes) != 0) {
            // Writing into an unreserved hole of a shared mapping would SIGBUS later.
            GGML_ABORT("ARGUS_KV could not re-reserve %zu bytes after clearing KV storage", state->bytes);
        }
        return;
    }
    state->inner->iface.clear(state->inner, value);
    argus_kv_page_out(state->address, state->bytes);
}

ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, size_t bytes) {
    const char * directory = std::getenv("ARGUS_KV_DIR");
    size_t limit = 0;
    if (!directory || !*directory || !parse_bytes(std::getenv("ARGUS_KV_MAX_BYTES"), limit) || bytes == 0 ||
        bytes > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
        std::fprintf(stderr, "ARGUS_KV requires a directory, a byte limit and nonzero allocation\n");
        return nullptr;
    }
    if (std::getenv("ARGUS_KV_RESIDENT_BYTES") && !argus_kv_resident_budget()) {
        std::fprintf(stderr, "ARGUS_KV_RESIDENT_BYTES must be a positive byte count\n");
        return nullptr;
    }

    // ponytail: process-wide allocation lock; separate arenas if parallel creation matters.
    std::lock_guard<std::mutex> lock(budget_mutex);
    if (live_bytes > limit || bytes > limit - live_bytes) {
        std::fprintf(stderr, "ARGUS_KV budget exceeded requested=%zu live=%zu limit=%zu\n", bytes, live_bytes, limit);
        return nullptr;
    }
    std::string pattern = std::string(directory) + "/argus-ggml-XXXXXX";
    std::vector<char> filename(pattern.begin(), pattern.end());
    filename.push_back('\0');
    const int fd = mkstemp(filename.data());
    if (fd < 0) { return nullptr; }
    // Reserve physical space before publishing the mapping, so ENOSPC is a
    // normal allocation error rather than a later SIGBUS during attention.
    const int error = posix_fallocate(fd, 0, static_cast<off_t>(bytes));
    if (unlink(filename.data()) != 0) {
        std::fprintf(stderr, "ARGUS_KV could not unlink %s: %s\n", filename.data(), std::strerror(errno));
        close(fd);
        return nullptr;
    }
    if (error != 0) {
        std::fprintf(stderr, "ARGUS_KV could not reserve %zu bytes: %s\n", bytes, std::strerror(error));
        close(fd);
        return nullptr;
    }
    void * address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (address == MAP_FAILED) {
        close(fd);
        return nullptr;
    }
    // Kernel readahead would pull cold neighbours back into RAM; attention asks
    // for exactly the blocks it reads through argus_kv_prefetch instead.
    madvise(address, bytes, MADV_RANDOM);
    posix_fadvise(fd, 0, static_cast<off_t>(bytes), POSIX_FADV_RANDOM);
    auto inner = ggml_backend_cpu_buffer_from_ptr(address, bytes);
    auto * state = inner ? new (std::nothrow) Mapping{inner, address, bytes, fd} : nullptr;
    const ggml_backend_buffer_i interface = {
        release, base, nullptr, fill, set, get, nullptr, nullptr, copy, clear, nullptr,
    };
    auto buffer = state ? ggml_backend_buffer_init(type, interface, state, bytes) : nullptr;
    if (!buffer) {
        if (inner) { ggml_backend_buffer_free(inner); }
        munmap(address, bytes);
        close(fd);
        delete state;
        return nullptr;
    }
    live_bytes += bytes;
    mappings.push_back(state);
    std::fprintf(stderr, "ARGUS_KV allocate bytes=%zu live_bytes=%zu address=%p resident_budget=%zu\n",
                 bytes, live_bytes, address, argus_kv_resident_budget());
    return buffer;
}

const char * name(ggml_backend_buffer_type_t) { return "ARGUS_HOST"; }
size_t alignment(ggml_backend_buffer_type_t) {
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
}
bool is_host(ggml_backend_buffer_type_t) { return true; }
} // namespace

ggml_backend_buffer_type_t argus_ggml_host_buffer_type() {
    static ggml_backend_buffer_type type = {
        {name, allocate, alignment, nullptr, nullptr, is_host}, nullptr, nullptr,
    };
    return &type;
}

size_t argus_kv_resident_budget() {
    size_t value = 0;
    return parse_bytes(std::getenv("ARGUS_KV_RESIDENT_BYTES"), value) ? value : 0;
}

double argus_kv_hot_fraction() {
    const size_t budget = argus_kv_resident_budget();
    std::lock_guard<std::mutex> lock(budget_mutex);
    return live_bytes ? std::min(1.0, static_cast<double>(budget) / static_cast<double>(live_bytes)) : 1.0;
}

void argus_kv_page_out(const void * address, size_t bytes, bool cold_before) {
    if (!argus_kv_resident_budget() || bytes == 0) { return; }
    // Whole pages only: a partial page may hold hot neighbours, unless the
    // caller knows the bytes before the range are cold too.
    auto first = cold_before ? reinterpret_cast<uintptr_t>(address) / page_bytes * page_bytes
                                   : (reinterpret_cast<uintptr_t>(address) + page_bytes - 1) / page_bytes * page_bytes;
    const auto last = (reinterpret_cast<uintptr_t>(address) + bytes) / page_bytes * page_bytes;
    if (last <= first) { return; }
    int fd = -1;
    off_t offset = 0;
    {
        std::lock_guard<std::mutex> lock(budget_mutex);
        for (const auto * state : mappings) {
            const auto begin = reinterpret_cast<uintptr_t>(state->address);
            if (first < begin && last > begin) { first = begin; }
            if (first >= begin && last <= begin + state->bytes) {
                fd = state->fd;
                offset = static_cast<off_t>(first - begin);
                break;
            }
        }
    }
    // Using fd after unlocking is safe: the range belongs to a tensor whose buffer is in
    // use by the caller, and a buffer is never released while its graph is computing.
    if (fd < 0) { return; }
    auto * start = reinterpret_cast<void *>(first);
    // Clean the pages, drop our PTEs, then evict them from the page cache; the
    // file keeps the bytes, so the next attention read faults them back in.
    msync(start, last - first, MS_SYNC);
    madvise(start, last - first, MADV_DONTNEED);
    posix_fadvise(fd, offset, static_cast<off_t>(last - first), POSIX_FADV_DONTNEED);
}

void argus_kv_prefetch(const void * address, size_t bytes) {
    if (!argus_kv_resident_budget() || bytes == 0) { return; }
    const auto first = reinterpret_cast<uintptr_t>(address) / page_bytes * page_bytes;
    const auto last = (reinterpret_cast<uintptr_t>(address) + bytes + page_bytes - 1) / page_bytes * page_bytes;
    madvise(reinterpret_cast<void *>(first), last - first, MADV_WILLNEED);
}

void argus_kv_note_attention(size_t bytes_read, size_t bytes_paged_out) {
    total_read += bytes_read;
    total_paged_out += bytes_paged_out;
    ++attention_calls;
}

void argus_kv_publish_stats() {
    const char * path = std::getenv("ARGUS_KV_STATS_PATH");
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    // mincore walks every mapping; sample at most once per second.
    auto last = last_publish_ms.load();
    if (now - last < 1000 || !last_publish_ms.compare_exchange_strong(last, now)) { return; }
    std::lock_guard<std::mutex> lock(budget_mutex);
    const size_t resident = resident_bytes_locked();
    size_t peak = peak_resident.load();
    while (resident > peak && !peak_resident.compare_exchange_weak(peak, resident)) {}
    if (!path || !*path) { return; }
    const std::string scratch = std::string(path) + ".tmp";
    FILE * stream = std::fopen(scratch.c_str(), "w");
    if (!stream) { return; }
    const int written = std::fprintf(stream,
        "{\"live_bytes\":%zu,\"resident_budget_bytes\":%zu,\"resident_bytes\":%zu,"
        "\"peak_resident_bytes\":%zu,\"read_bytes\":%zu,\"paged_out_bytes\":%zu,"
        "\"attention_calls\":%zu,\"mappings\":%zu}\n",
        live_bytes, argus_kv_resident_budget(), resident, peak_resident.load(), total_read.load(),
        total_paged_out.load(), attention_calls.load(), mappings.size());
    if (std::fclose(stream) == 0 && written > 0) {
        std::rename(scratch.c_str(), path);
    } else {
        std::remove(scratch.c_str());
    }
}
