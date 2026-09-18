#include "ggml_disk_buffer.h"
#include "ggml_profile.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu/traits.h"
#ifdef ARGUS_CUDA
#include "ggml_cuda_attention.h"
#include "ggml_kv_policy.h"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

namespace {
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
#endif
};
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
#endif
    std::mutex mutex;
};
#ifdef ARGUS_CUDA
std::mutex registry_mutex;
Store * registry = nullptr;
#endif
constexpr size_t descriptor_offset = (sizeof(Store) + alignof(Page) - 1) / alignof(Page) * alignof(Page);

void name_mapping(void * address, size_t bytes, const char * name) {
#ifdef PR_SET_VMA_ANON_NAME
    // Naming is for /proc measurement only; allocation accounting does not depend on it.
    prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, address, bytes, name);
#else
    (void) address; (void) bytes; (void) name;
#endif
}

std::mutex budget_mutex;
size_t disk_live = 0, metadata_live = 0, staging_live = 0;
size_t peak_metadata = 0, peak_staging = 0, stores = 0;
uint64_t next_id = 0;
std::atomic<size_t> read_bytes{0}, written_bytes{0}, committed_pages{0};

size_t limit(const char * name) {
    const char * text = std::getenv(name);
    if (!text || !*text) { throw std::runtime_error(std::string(name) + " is required"); }
    for (const char * p = text; *p; ++p) {
        if (*p < '0' || *p > '9') { throw std::runtime_error(std::string(name) + " must be positive bytes"); }
    }
    errno = 0;
    const auto value = std::strtoull(text, nullptr, 10);
    if (errno || !value || value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string(name) + " is out of range");
    }
    return static_cast<size_t>(value);
}

size_t rounded(size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max() - page_size + 1) {
        throw std::runtime_error("ARGUS page size overflow");
    }
    return (bytes + page_size - 1) / page_size * page_size;
}

// A page-aligned mapping also makes the physical staging allocation explicit.
struct Bounce {
    ArgusStagingBuffer buffer{page_size};
    void * data = buffer.data();
};

uint32_t checksum(const void * data) {
    argus_profile::Scope timer(argus_profile::checksum);
    // FNV-1a detects accidental payload corruption; this is not an authenticity check.
    uint32_t hash = 2166136261u;
    const auto * bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < page_size; ++i) { hash = (hash ^ bytes[i]) * 16777619u; }
    return hash;
}

off_t offset(size_t page, uint32_t active) {
    return static_cast<off_t>((page * 2 + active) * page_size);
}

Page & page_at(Store & store, size_t index) {
    argus_profile::Scope timer(argus_profile::page_lookup);
    return store.pages[index];
}

void read_disk_page(Store & store, size_t page, void * data) {
    const Page & descriptor = page_at(store, page);
    if (!descriptor.content_revision || descriptor.active == 2) { std::memset(data, 0, page_size); return; }
    if (descriptor.active == 3) { throw std::runtime_error("ARGUS GPU control has no disk backing"); }
    ssize_t got;
    {
        argus_profile::Scope timer(argus_profile::disk_read);
        do { got = pread(store.fd, data, page_size, offset(page, descriptor.active)); } while (got < 0 && errno == EINTR);
    }
    if (got != static_cast<ssize_t>(page_size)) { throw std::runtime_error("ARGUS short or failed direct page read"); }
    read_bytes += page_size;
    if (argus_profile::enabled()) { argus_profile::disk_read_bytes[argus_profile::phase] += page_size; }
    if (checksum(data) != descriptor.checksum) { throw std::runtime_error("ARGUS disk page checksum mismatch"); }
}

void read_page(Store & store, size_t page, void * data) {
#ifdef ARGUS_CUDA
    const auto & descriptor = page_at(store, page);
    if (descriptor.resident) {
        descriptor.resident->read(data, 0, page_size);
        if (checksum(data) != descriptor.checksum) { throw std::runtime_error("ARGUS resident checksum mismatch"); }
        return;
    }
#endif
    read_disk_page(store, page, data);
}

void write_page(Store & store, size_t page, void * data) {
    argus_profile::Scope timer(argus_profile::write_page);
    Page & descriptor = page_at(store, page);
    if (descriptor.content_revision == std::numeric_limits<uint64_t>::max() ||
        descriptor.placement_revision == std::numeric_limits<uint64_t>::max() || store.content_revision == std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error("ARGUS page generation exhausted");
    }
#ifdef ARGUS_CUDA
    if (store.gpu_control) {
        // Diagnostic control, not a placement policy or durability mode. Preserve the
        // old page until a separately budgeted GPU destination passes verification.
        const auto digest = checksum(data);
        auto target = std::make_unique<ArgusTierBuffer>(ArgusTier::gpu, page_size);
        target->write(data, page_size);
        target->read(data, 0, page_size);
        if (checksum(data) != digest) { throw std::runtime_error("ARGUS GPU control verification failed"); }
        delete descriptor.resident;
        descriptor.resident = target.release();
        ++descriptor.content_revision;
        ++descriptor.placement_revision;
        descriptor.checksum = digest;
        descriptor.active = 3;
        ++store.content_revision;
        store.last_written_page = page;
        ++committed_pages;
        return;
    }
#endif
    const uint32_t inactive = 1 - (descriptor.active & 1);
    ssize_t wrote;
    {
        argus_profile::Scope io(argus_profile::disk_write);
        do { wrote = pwrite(store.fd, data, page_size, offset(page, inactive)); } while (wrote < 0 && errno == EINTR);
    }
    if (wrote != static_cast<ssize_t>(page_size)) {
        // The previously published slot remains untouched, even after a short write.
        throw std::runtime_error("ARGUS short or failed direct page write");
    }
    written_bytes += page_size;
    if (argus_profile::enabled()) { argus_profile::disk_write_bytes[argus_profile::phase] += page_size; }
    const auto digest = checksum(data);
    // Verify the destination before publishing its generation and physical slot.
    ssize_t got;
    {
        argus_profile::Scope io(argus_profile::disk_read);
        do { got = pread(store.fd, data, page_size, offset(page, inactive)); } while (got < 0 && errno == EINTR);
    }
    if (got != static_cast<ssize_t>(page_size) || checksum(data) != digest) {
        throw std::runtime_error("ARGUS disk page write verification failed");
    }
    read_bytes += page_size;
    if (argus_profile::enabled()) { argus_profile::disk_read_bytes[argus_profile::phase] += page_size; }
#ifdef ARGUS_CUDA
    delete descriptor.resident;
    descriptor.resident = nullptr;
#endif
    ++descriptor.content_revision;
    ++descriptor.placement_revision;
    descriptor.checksum = digest;
    descriptor.active = inactive;
    ++store.content_revision;
    store.last_written_page = page;
    ++committed_pages;
}

Store & store_for(ggml_backend_buffer_t buffer) { return *static_cast<Store *>(buffer->context); }

size_t checked_offset(const Store & store, const ggml_tensor * tensor, size_t start, size_t bytes) {
    argus_profile::Scope timer(argus_profile::page_lookup);
    const auto base = reinterpret_cast<uintptr_t>(store.address);
    const auto address = reinterpret_cast<uintptr_t>(tensor->data);
    if (address < base || address - base > store.bytes || start > store.bytes - (address - base) ||
        bytes > store.bytes - (address - base) - start) {
        throw std::out_of_range("ARGUS disk tensor range");
    }
    return address - base + start;
}

size_t migration_page(const Store & store, const ggml_tensor * tensor, size_t start) {
    const size_t position = checked_offset(store, tensor, start, page_size);
    if (position % page_size || start > ggml_nbytes(tensor) || page_size > ggml_nbytes(tensor) - start) {
        throw std::invalid_argument("ARGUS move requires a complete aligned page inside the tensor");
    }
    return position / page_size;
}

ggml_status init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    // Views describe the same encoded storage; their interpretation is not a conversion.
    if (tensor->view_src) { return GGML_STATUS_SUCCESS; }
    auto & store = store_for(buffer);
    const size_t bytes = ggml_nbytes(tensor);
    const size_t start = checked_offset(store, tensor, 0, bytes);
    if (!bytes) { return GGML_STATUS_SUCCESS; }
    if (start % page_size) { return GGML_STATUS_FAILED; }
    std::lock_guard<std::mutex> guard(store.mutex);
    const size_t end = (start + bytes - 1) / page_size;
    for (size_t i = start / page_size; i <= end; ++i) {
        if (store.pages[i].codec != GGML_TYPE_COUNT && store.pages[i].codec != tensor->type) {
            return GGML_STATUS_FAILED;
        }
    }
    for (size_t i = start / page_size; i <= end; ++i) { store.pages[i].codec = tensor->type; }
    return GGML_STATUS_SUCCESS;
}

// Called under the store lock only after the entire public read succeeds.
void record_access(Store & store, size_t start, size_t bytes) {
    argus_profile::Scope timer(argus_profile::page_lookup);
    if (!bytes) { return; }
    if (store.access_step != UINT64_MAX) { ++store.access_step; }
    const size_t end = (start + bytes - 1) / page_size;
    for (size_t i = start / page_size; i <= end; ++i) {
        auto & page = store.pages[i];
        page.last_access_step = store.access_step;
        if (page.access_count != UINT64_MAX) { ++page.access_count; }
    }
}

void transfer(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
              size_t start, size_t bytes, bool writing) {
    auto & store = store_for(buffer);
    size_t position = checked_offset(store, tensor, start, bytes);
    const size_t access_start = position, access_bytes = bytes;
    if (!bytes) { return; }
    Bounce bounce;
    auto * cursor = static_cast<char *>(data);
    std::lock_guard<std::mutex> guard(store.mutex);
    while (bytes) {
        const size_t page = position / page_size, within = position % page_size;
        const size_t count = std::min(bytes, page_size - within);
        if (!writing || within || count != page_size) { read_page(store, page, bounce.data); }
        if (writing) {
            std::memcpy(static_cast<char *>(bounce.data) + within, cursor, count);
            write_page(store, page, bounce.data);
        } else {
            std::memcpy(cursor, static_cast<char *>(bounce.data) + within, count);
        }
        position += count;
        cursor += count;
        bytes -= count;
    }
    if (!writing) { record_access(store, access_start, access_bytes); }
}

void get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t start, size_t bytes) {
    transfer(buffer, tensor, data, start, bytes, false);
}
void set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t start, size_t bytes) {
    transfer(buffer, tensor, const_cast<void *>(data), start, bytes, true);
}
void fill(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t start, size_t bytes) {
    Bounce bounce;
    std::memset(bounce.data, value, page_size);
    while (bytes) {
        const size_t count = std::min(bytes, page_size);
        set(buffer, tensor, bounce.data, start, count);
        bytes -= count;
        start += count;
    }
}
bool copy(ggml_backend_buffer_t buffer, const ggml_tensor * source, ggml_tensor * target) {
    Bounce bounce;
    for (size_t start = 0, total = ggml_nbytes(source); start < total;) {
        const size_t count = std::min(total - start, page_size);
        ggml_backend_tensor_get(source, bounce.data, start, count);
        set(buffer, target, bounce.data, start, count);
        start += count;
    }
    return true;
}
void clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto & store = store_for(buffer);
    if (value == 0) {
        std::lock_guard<std::mutex> guard(store.mutex);
        // No outstanding reads survive this lock; no data pages need to be faulted in.
        const size_t count = rounded(store.bytes) / page_size;
        if (store.content_revision == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("ARGUS store generation exhausted");
        }
        for (size_t i = 0; i < count; ++i) {
            if (store.pages[i].content_revision == std::numeric_limits<uint64_t>::max() ||
                store.pages[i].placement_revision == std::numeric_limits<uint64_t>::max()) {
                throw std::runtime_error("ARGUS page generation exhausted");
            }
        }
        for (size_t i = 0; i < count; ++i) {
#ifdef ARGUS_CUDA
            delete store.pages[i].resident;
#endif
            const auto codec = store.pages[i].codec;
            store.pages[i] = {store.pages[i].content_revision + 1, store.pages[i].placement_revision + 1, 0, 2, codec};
        }
        store.access_step = 0;
        ++store.content_revision;
        return;
    }
    ggml_tensor tensor{};
    tensor.data = store.address;
    fill(buffer, &tensor, value, 0, store.bytes);
}
void * base(ggml_backend_buffer_t buffer) { return store_for(buffer).address; }

void release(ggml_backend_buffer_t buffer) {
    auto * store = static_cast<Store *>(buffer->context);
    const auto metadata = store->metadata_bytes, charged = store->metadata_charge, physical = store->disk_bytes;
    const auto id = store->id;
#ifdef ARGUS_CUDA
    // Registry users finish before this allocation's pages can be destroyed.
    std::unique_lock<std::mutex> registry_guard(registry_mutex);
    Store ** link = &registry;
    while (*link && *link != store) { link = &(*link)->next; }
    if (*link) { *link = store->next; }
    for (size_t i = 0; i < rounded(store->bytes) / page_size; ++i) { delete store->pages[i].resident; }
#endif
    munmap(store->address, rounded(store->bytes));
    if (store->fd >= 0) { close(store->fd); }
    store->~Store();
    munmap(store, metadata);
#ifdef ARGUS_CUDA
    registry_guard.unlock();
#endif
    std::lock_guard<std::mutex> guard(budget_mutex);
    disk_live -= physical;
    metadata_live -= charged;
    --stores;
    std::fprintf(stderr, "ARGUS_DISK free id=%llu disk_live=%zu resident_live=%zu staging_live=%zu\n",
                 static_cast<unsigned long long>(id), disk_live, metadata_live, staging_live);
}

ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, size_t bytes) try {
    const char * control = std::getenv("ARGUS_KV_GPU_CONTROL");
    const bool gpu_control = control && std::strcmp(control, "1") == 0;
    if (control && !gpu_control) { throw std::invalid_argument("ARGUS_KV_GPU_CONTROL must be 1 or unset"); }
#ifdef ARGUS_CUDA
    if (gpu_control && argus_kv_policy_enabled()) { throw std::invalid_argument("ARGUS GPU control requires policy=off"); }
#else
    if (gpu_control) { throw std::invalid_argument("ARGUS GPU control requires CUDA"); }
#endif
    const size_t allocation = rounded(bytes);
    if (!bytes || allocation > static_cast<size_t>(std::numeric_limits<off_t>::max()) / 2) {
        throw std::runtime_error("ARGUS disk allocation size is invalid");
    }
    const size_t physical = gpu_control ? 0 : allocation * 2;
    const size_t metadata = rounded(descriptor_offset + allocation / page_size * sizeof(Page));
    size_t metadata_charge = metadata;
#ifdef ARGUS_CUDA
    // Reserve both handles during a migration; payloads have separate tier budgets.
    metadata_charge += allocation / page_size * 2 * sizeof(ArgusTierBuffer);
#endif
    const size_t max_disk = limit("ARGUS_KV_MAX_BYTES"), max_resident = limit("ARGUS_KV_RESIDENT_BYTES");
    if (argus_disk_staging_limit() < 2 * page_size) { throw std::runtime_error("ARGUS staging needs at least 8192 bytes"); }
    const char * directory = std::getenv("ARGUS_KV_DIR");
    if (!directory || !*directory) { throw std::runtime_error("ARGUS_KV_DIR is required"); }
    std::unique_lock<std::mutex> guard(budget_mutex);
    if (physical > max_disk || disk_live > max_disk - physical ||
        metadata_charge > max_resident || metadata_live > max_resident - metadata_charge) {
        throw std::runtime_error("ARGUS disk or resident metadata budget exceeded");
    }
    std::string pattern = std::string(directory) + "/argus-direct-XXXXXX";
    std::vector<char> filename(pattern.begin(), pattern.end());
    filename.push_back('\0');
    int fd = -1;
    if (!gpu_control) {
        fd = mkstemp(filename.data());
        if (fd < 0) { throw std::runtime_error("ARGUS could not create disk store"); }
        const int reservation = posix_fallocate(fd, 0, static_cast<off_t>(physical));
        const int removed = unlink(filename.data());
        if (reservation || removed || fcntl(fd, F_SETFL, O_DIRECT) < 0 || fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
            close(fd);
            throw std::runtime_error("ARGUS could not reserve direct-I/O storage");
        }
    }
    void * address = mmap(nullptr, allocation, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void * descriptors = mmap(nullptr, metadata, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address == MAP_FAILED || descriptors == MAP_FAILED) {
        if (address != MAP_FAILED) { munmap(address, allocation); }
        if (descriptors != MAP_FAILED) { munmap(descriptors, metadata); }
        close(fd);
        throw std::bad_alloc();
    }
    auto * store = new (descriptors) Store;
#ifdef ARGUS_CUDA
    store->gpu_control = gpu_control;
#endif
    store->fd = fd;
    store->address = address;
    store->pages = reinterpret_cast<Page *>(static_cast<char *>(descriptors) + descriptor_offset);
    name_mapping(address, allocation, "argus-disk-handle");
    name_mapping(descriptors, metadata, "argus-disk-metadata");
    for (size_t i = 0; i < allocation / page_size; ++i) { new (store->pages + i) Page{}; }
    store->bytes = bytes;
    store->disk_bytes = physical;
    store->metadata_bytes = metadata;
    store->metadata_charge = metadata_charge;
    store->id = ++next_id;
    const ggml_backend_buffer_i iface = {release, base, init_tensor, fill, set, get, nullptr, nullptr, copy, clear, nullptr};
    auto * result = ggml_backend_buffer_init(type, iface, store, bytes);
    if (!result) {
        munmap(store->address, allocation);
        close(store->fd);
        store->~Store();
        munmap(descriptors, metadata);
        return nullptr;
    }
    disk_live += physical;
    metadata_live += metadata_charge;
    peak_metadata = std::max(peak_metadata, metadata_live);
    ++stores;
    std::fprintf(stderr, "ARGUS_DISK allocate id=%llu logical_bytes=%zu disk_bytes=%zu metadata_bytes=%zu\n",
                 static_cast<unsigned long long>(store->id), bytes, physical, metadata);
#ifdef ARGUS_CUDA
    guard.unlock(); // Never acquire the registry while holding the budget lock.
    std::lock_guard<std::mutex> registry_guard(registry_mutex);
    store->next = registry;
    registry = store;
#endif
    return result;
} catch (const std::exception & error) {
    std::fprintf(stderr, "ARGUS_DISK allocation refused: %s\n", error.what());
    return nullptr;
}

const char * name(ggml_backend_buffer_type_t) { return "ARGUS_DISK"; }
size_t alignment(ggml_backend_buffer_type_t) { return page_size; }
bool is_host(ggml_backend_buffer_type_t) { return false; }

class DiskSupport final : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override { return op->op == GGML_OP_CUSTOM; }
    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor *) override { return nullptr; }
};

void set_rows(ggml_tensor * dst, int ith, int, void *) try {
    if (ith != 0) { return; }
    const auto * source = dst->src[0], * indices = dst->src[1];
    argus_profile::InPhase phase(source->ne[1] > 1);
    argus_profile::Scope timer(argus_profile::set_rows);
    auto * target = dst->src[2];
    const size_t bytes = ggml_row_size(target->type, target->ne[0]);
    ArgusStagingBuffer encoded(bytes);
    const size_t capacity = target->nb[1] == bytes ? encoded.size() / bytes : 1;
    size_t pending = 0;
    int64_t first = 0;
    auto flush = [&] {
        if (pending) { ggml_backend_tensor_set(target, encoded.data(), first * target->nb[1], pending * bytes); }
        pending = 0;
    };
    for (int64_t row = 0; row < source->ne[1]; ++row) {
        int64_t index;
        std::memcpy(&index, static_cast<const char *>(indices->data) + row * indices->nb[0], sizeof index);
        if (index < 0 || index >= target->ne[1]) { throw std::runtime_error("ARGUS KV row index out of range"); }
        if (pending && (pending == capacity || index != first + static_cast<int64_t>(pending))) { flush(); }
        if (!pending) { first = index; }
        const auto * values = reinterpret_cast<const float *>(static_cast<const char *>(source->data) + row * source->nb[1]);
        auto * destination = static_cast<char *>(encoded.data()) + pending * bytes;
        if (target->type == GGML_TYPE_F32) { std::memcpy(destination, values, bytes); }
        else { ggml_get_type_traits_cpu(target->type)->from_float(values, destination, target->ne[0]); }
        ++pending;
    }
    flush();
    *static_cast<float *>(dst->data) = 0;
} catch (const std::exception & error) {
    GGML_ABORT("ARGUS disk append failed: %s", error.what());
}
} // namespace

bool argus_ggml_disk_enabled() { return std::getenv("ARGUS_KV_STAGING_BYTES") != nullptr; }
size_t argus_disk_staging_limit() { return limit("ARGUS_KV_STAGING_BYTES"); }

ArgusStagingReservation::ArgusStagingReservation(size_t bytes) : bytes_(rounded(bytes)) {
    const size_t maximum = argus_disk_staging_limit();
    std::lock_guard<std::mutex> guard(budget_mutex);
    if (bytes_ > maximum || staging_live > maximum - bytes_) {
        throw std::runtime_error("ARGUS staging budget exceeded: requested=" + std::to_string(bytes_) +
                                 " live=" + std::to_string(staging_live) + " limit=" + std::to_string(maximum));
    }
    staging_live += bytes_;
    peak_staging = std::max(peak_staging, staging_live);
}
ArgusStagingReservation::~ArgusStagingReservation() {
    std::lock_guard<std::mutex> guard(budget_mutex);
    staging_live -= bytes_;
}

ArgusStagingBuffer::ArgusStagingBuffer(size_t bytes) : reservation_(bytes), bytes_(rounded(bytes)),
    data_(mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) {
    if (data_ == MAP_FAILED) { throw std::bad_alloc(); }
    name_mapping(data_, bytes_, "argus-disk-staging");
}
ArgusStagingBuffer::~ArgusStagingBuffer() { munmap(data_, bytes_); }

ArgusDiskRevision argus_disk_revision(const ggml_tensor * tensor) {
    const auto * storage = tensor->view_src ? tensor->view_src : tensor;
    if (!argus_ggml_is_disk_tensor(storage)) { throw std::invalid_argument("ARGUS revision requires disk storage"); }
    auto & store = store_for(storage->buffer);
    std::lock_guard<std::mutex> guard(store.mutex);
    return {store.id, store.content_revision};
}

ArgusDiskPageRevision argus_disk_page_revision(const ggml_tensor * tensor, size_t start) {
    if (!argus_ggml_is_disk_tensor(tensor)) { throw std::invalid_argument("ARGUS revision requires disk storage"); }
    const auto * storage = tensor->view_src ? tensor->view_src : tensor;
    auto & store = store_for(storage->buffer);
    const size_t index = migration_page(store, tensor, start);
    std::lock_guard<std::mutex> guard(store.mutex);
    return {store.id, index, store.pages[index].content_revision};
}

static ArgusDiskPageDescriptor page_descriptor(const Store & store, size_t index) {
    const auto & page = store.pages[index];
    auto placement = ArgusTier::disk;
#ifdef ARGUS_CUDA
    if (page.resident) { placement = page.resident->tier(); }
#endif
    return {{store.id, index, page.content_revision}, page.placement_revision, placement,
            page.codec, page.last_access_step, page.access_count, page.content_revision != 0 && page.active != 2};
}

ArgusDiskPageDescriptor argus_disk_page_descriptor(const ggml_tensor * tensor, size_t start) {
    argus_profile::Scope timer(argus_profile::descriptor_scan);
    if (!argus_ggml_is_disk_tensor(tensor)) { throw std::invalid_argument("ARGUS descriptor requires disk storage"); }
    if (start >= ggml_nbytes(tensor)) { throw std::out_of_range("ARGUS descriptor tensor range"); }
    const auto * storage = tensor->view_src ? tensor->view_src : tensor;
    auto & store = store_for(storage->buffer);
    const size_t index = checked_offset(store, tensor, start, 1) / page_size;
    std::lock_guard<std::mutex> guard(store.mutex);
    return page_descriptor(store, index);
}

ArgusDiskPrefetch::ArgusDiskPrefetch() {
    if (mprotect(stack_.data(), 4096, PROT_NONE)) { throw std::runtime_error("ARGUS prefetch stack guard failed"); }
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes)) { throw std::runtime_error("ARGUS prefetch attributes failed"); }
    const int configured = pthread_attr_setstack(&attributes, static_cast<char *>(stack_.data()) + 4096, stack_.size() - 4096);
    const int created = configured ? configured : pthread_create(&thread_, &attributes, run, this);
    pthread_attr_destroy(&attributes);
    if (created) { throw std::runtime_error(std::string("ARGUS prefetch worker creation failed: ") + std::strerror(created)); }
}
ArgusDiskPrefetch::~ArgusDiskPrefetch() {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        stop_ = true;
        condition_.notify_all();
    }
    // Join before the caller's block buffers or the charged stack can be reused.
    if (pthread_join(thread_, nullptr)) { GGML_ABORT("ARGUS could not join prefetch worker"); }
}
void ArgusDiskPrefetch::submit(const ggml_tensor * k, const ggml_tensor * v, int64_t first, int64_t count,
                              void * keys, void * values) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (busy_) { throw std::runtime_error("ARGUS prefetch queue is full"); }
    if (!k || !v || !keys || !values) { throw std::invalid_argument("ARGUS prefetch requires tensors and destinations"); }
    if (first < 0 || count <= 0 || first > k->ne[2] || count > k->ne[2] - first ||
        first > v->ne[2] || count > v->ne[2] - first) { throw std::out_of_range("ARGUS prefetch range"); }
    request_ = {k, v, first, count, keys, values, argus_disk_revision(k), argus_disk_revision(v)};
    error_ = nullptr;
    busy_ = pending_ = true;
    ready_ = false;
    condition_.notify_all();
}
void * ArgusDiskPrefetch::run(void * context) {
    auto & self = *static_cast<ArgusDiskPrefetch *>(context);
    std::unique_lock<std::mutex> guard(self.mutex_);
    for (;;) {
        self.condition_.wait(guard, [&] { return self.stop_ || self.pending_; });
        if (self.stop_) { return nullptr; }
        const auto request = self.request_;
        self.pending_ = false;
        guard.unlock();
        std::exception_ptr error;
        try {
            ggml_backend_tensor_get(request.k, request.keys, request.first * request.k->nb[2], request.count * request.k->nb[2]);
            ggml_backend_tensor_get(request.v, request.values, request.first * request.v->nb[2], request.count * request.v->nb[2]);
        } catch (...) { error = std::current_exception(); }
        guard.lock();
        self.error_ = error;
        self.ready_ = true;
        self.condition_.notify_all();
    }
}
void ArgusDiskPrefetch::take() {
    std::unique_lock<std::mutex> guard(mutex_);
    if (!busy_) { throw std::runtime_error("ARGUS prefetch queue is empty"); }
    condition_.wait(guard, [&] { return ready_; });
    busy_ = ready_ = false;
    if (error_) { std::rethrow_exception(error_); }
    if (!(request_.k_revision == argus_disk_revision(request_.k)) ||
        !(request_.v_revision == argus_disk_revision(request_.v))) {
        throw std::runtime_error("ARGUS stale prefetch generation");
    }
}

ggml_backend_buffer_type_t argus_ggml_disk_buffer_type() {
    static DiskSupport support;
    static ggml_backend_buffer_type type = {{name, allocate, alignment, nullptr, nullptr, is_host}, nullptr, &support};
    static const bool registered = [&] {
        ggml_backend_cpu_get_extra_buffer_types().push_back(&type);
        return true;
    }();
    (void) registered;
    return &type;
}
bool argus_ggml_is_disk_tensor(const ggml_tensor * tensor) {
    const auto * storage = tensor->view_src ? tensor->view_src : tensor;
    return storage->buffer && ggml_backend_buffer_get_type(storage->buffer) == argus_ggml_disk_buffer_type();
}

ggml_tensor * argus_ggml_disk_set_rows(ggml_context * ctx, ggml_tensor * target,
                                    ggml_tensor * source, ggml_tensor * indices) {
    if (!argus_ggml_is_disk_tensor(target)) { return nullptr; }
    if (source->type != GGML_TYPE_F32 || indices->type != GGML_TYPE_I64 || source->ne[0] != target->ne[0] ||
        source->ne[2] != 1 || source->ne[3] != 1 || target->ne[2] != 1 || target->ne[3] != 1 ||
        source->nb[0] != sizeof(float) || indices->ne[0] != source->ne[1] ||
        (target->type != GGML_TYPE_F32 && !ggml_get_type_traits_cpu(target->type)->from_float)) {
        throw std::runtime_error("ARGUS disk append contract is unsupported");
    }
    if (rounded(ggml_row_size(target->type, target->ne[0])) + page_size > argus_disk_staging_limit()) {
        throw std::runtime_error("ARGUS staging budget cannot hold an encoded KV row");
    }
    ggml_tensor * args[] = {source, indices, target};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 3, set_rows, 1, nullptr);
}

void argus_disk_publish_stats() {
    const char * path = std::getenv("ARGUS_KV_STATS_PATH");
    if (!path || !*path) { return; }
    std::string tier_stats;
#ifdef ARGUS_CUDA
    const auto usage = argus_tier_usage();
    const auto policy = argus_kv_policy_stats();
    tier_stats = ",\"gpu_bytes\":" + std::to_string(usage.gpu) +
        ",\"pinned_bytes\":" + std::to_string(usage.pinned) + ",\"ram_bytes\":" + std::to_string(usage.ram) +
        ",\"peak_gpu_bytes\":" + std::to_string(usage.peak_gpu) +
        ",\"peak_pinned_bytes\":" + std::to_string(usage.peak_pinned) +
        ",\"peak_ram_bytes\":" + std::to_string(usage.peak_ram) +
        ",\"cuda_attention_calls\":" + std::to_string(usage.attention_calls) +
        ",\"policy_promotions\":" + std::to_string(policy.promotions) +
        ",\"policy_demotions\":" + std::to_string(policy.demotions) +
        ",\"policy_rejected\":" + std::to_string(policy.rejected) +
        ",\"policy_nanoseconds\":" + std::to_string(policy.nanoseconds);
#endif
    tier_stats += argus_profile::json_fields() + argus_profile::residency_json();
    std::lock_guard<std::mutex> guard(budget_mutex);
    const std::string scratch = std::string(path) + ".tmp";
    FILE * stream = std::fopen(scratch.c_str(), "w");
    if (!stream) { throw std::runtime_error("ARGUS could not open stats file"); }
    const int wrote = std::fprintf(stream,
        "{\"mode\":\"direct\",\"disk_bytes\":%zu,\"resident_bytes\":%zu,\"staging_bytes\":%zu,"
        "\"peak_resident_bytes\":%zu,\"peak_staging_bytes\":%zu,\"read_bytes\":%zu,"
        "\"written_bytes\":%zu,\"committed_pages\":%zu,\"stores\":%zu%s}\n",
        disk_live, metadata_live, staging_live, peak_metadata, peak_staging,
        read_bytes.load(), written_bytes.load(), committed_pages.load(), stores, tier_stats.c_str());
    const int closed = std::fclose(stream);
    if (wrote < 0 || closed || std::rename(scratch.c_str(), path)) {
        throw std::runtime_error("ARGUS could not publish stats");
    }
}

#ifdef ARGUS_CUDA
static void move_page(Store & store, size_t index, ArgusTier tier, ArgusDiskPageRevision expected) {
    std::lock_guard<std::mutex> guard(store.mutex);
    auto & page = store.pages[index];
    if (expected.allocation != store.id || expected.page != index ||
        expected.content_revision != page.content_revision) { throw std::runtime_error("ARGUS stale migration"); }
    if (!page.content_revision || page.active == 2) { throw std::invalid_argument("ARGUS cannot promote an unwritten page"); }
    if ((page.resident ? page.resident->tier() : ArgusTier::disk) == tier) { return; }
    if (store.gpu_control) { throw std::runtime_error("ARGUS GPU control cannot migrate out of GPU"); }
    if (page.placement_revision == UINT64_MAX) { throw std::runtime_error("ARGUS generation exhausted"); }
    std::unique_ptr<ArgusTierBuffer> target;
    Bounce data;
    if (tier != ArgusTier::disk) {
        read_page(store, index, data.data);
        const auto digest = checksum(data.data);
        target = std::make_unique<ArgusTierBuffer>(tier, page_size);
        target->write(data.data, page_size);
        target->read(data.data, 0, page_size);
        if (checksum(data.data) != digest) { throw std::runtime_error("ARGUS migration verification failed"); }
    } else {
        // A retained backing copy can still have suffered corruption since its write.
        read_disk_page(store, index, data.data);
    }
    delete page.resident;
    page.resident = target.release();
    ++page.placement_revision;
}

void argus_disk_move_page(const ggml_tensor * tensor, size_t start, ArgusTier tier, ArgusDiskPageRevision expected) {
    if (!argus_ggml_is_disk_tensor(tensor)) { throw std::invalid_argument("ARGUS move requires disk-backed tensor"); }
    const auto * storage = tensor->view_src ? tensor->view_src : tensor;
    auto & store = store_for(storage->buffer);
    move_page(store, migration_page(store, tensor, start), tier, expected);
}

void argus_disk_move_page(ArgusDiskPageRevision expected, ArgusTier tier) {
    std::lock_guard<std::mutex> guard(registry_mutex);
    for (auto * store = registry; store; store = store->next) {
        if (store->id != expected.allocation) { continue; }
        if (expected.page >= store->bytes / page_size) { throw std::out_of_range("ARGUS migration page range"); }
        move_page(*store, expected.page, tier, expected);
        return;
    }
    throw std::runtime_error("ARGUS stale migration allocation");
}

void argus_disk_visit_resident_pages(void (*visit)(const ArgusDiskPageDescriptor &, void *), void * context) {
    argus_profile::Scope timer(argus_profile::descriptor_scan);
    std::lock_guard<std::mutex> guard(registry_mutex);
    for (auto * store = registry; store; store = store->next) {
        std::lock_guard<std::mutex> page_guard(store->mutex);
        for (size_t i = 0; i < rounded(store->bytes) / page_size; ++i) {
            if (store->pages[i].resident) { visit(page_descriptor(*store, i), context); }
        }
    }
}

void argus_disk_stage_cuda(const ggml_tensor * tensor, void * host, void * device,
                          size_t start, size_t bytes, void * stream) {
    argus_profile::Scope timer(argus_profile::staging);
    if (!argus_ggml_is_disk_tensor(tensor)) { throw std::invalid_argument("ARGUS staging requires disk tensor"); }
    if (!host || !device || start > ggml_nbytes(tensor) || bytes > ggml_nbytes(tensor) - start) {
        throw std::out_of_range("ARGUS CUDA staging range");
    }
    const auto * storage = tensor->view_src ? tensor->view_src : tensor;
    auto & store = store_for(storage->buffer);
    size_t position = checked_offset(store, tensor, start, bytes);
    Bounce bounce;
    std::lock_guard<std::mutex> guard(store.mutex);
    try {
        size_t copied = 0;
        while (copied < bytes) {
            const size_t page_index = position / page_size, within = position % page_size;
            const size_t count = std::min(bytes - copied, page_size - within);
            const auto & page = page_at(store, page_index);
            if (page.resident && page.resident->tier() == ArgusTier::gpu && page.active != 2) {
                argus_cuda_copy(static_cast<char *>(device) + copied,
                    static_cast<char *>(page.resident->data()) + within, count, true, stream);
            } else {
                read_page(store, page_index, bounce.data);
                std::memcpy(static_cast<char *>(host) + copied, static_cast<char *>(bounce.data) + within, count);
                argus_cuda_copy(static_cast<char *>(device) + copied, static_cast<char *>(host) + copied,
                                count, false, stream);
            }
            position += count;
            copied += count;
        }
        argus_cuda_wait(stream);
        record_access(store, position - bytes, bytes);
    } catch (...) {
        // Drain queued copies before unlocking resident pages or releasing caller buffers.
        argus_cuda_wait(stream);
        throw;
    }
}

// Diagnostic snapshot of the pages an attention invocation would read, taken before
// record_access so access counts are those the eligibility check saw.
static void residency_census(Store * const stores[2], const size_t starts[2], const size_t counts[2]) {
    using namespace argus_profile;
    Scope timer(descriptor_scan);
    auto & c = census[phase];
    uint64_t gpu = 0, written = 0, cold = 0;
    for (int t = 0; t < 2; ++t) {
        const Page * pages = stores[t]->pages + starts[t] / page_size;
        const auto is_written = [&](size_t i) { return pages[i].content_revision && pages[i].active != 2; };
        const auto tier = [&](size_t i) { return pages[i].resident ? pages[i].resident->tier() : ArgusTier::disk; };
        // Distance behind the store's most recent page write; pages ahead of it (or a
        // frontier outside this view) are binned as 64+.
        const size_t first = starts[t] / page_size, last = stores[t]->last_written_page;
        uint64_t tensor_cold = 0, run = 0;
        bool previous_cold = false;
        for (size_t i = 0; i < counts[t]; ++i) {
            const bool page_written = is_written(i);
            const auto placement = tier(i);
            const bool is_cold = page_written && placement != ArgusTier::gpu;
            ++c.pages[!page_written ? page_unwritten : placement == ArgusTier::gpu ? page_gpu :
                      placement == ArgusTier::pinned ? page_pinned : placement == ArgusTier::ram ? page_ram : page_disk];
            written += page_written;
            gpu += page_written && !is_cold;
            if (i && is_cold != previous_cold) { ++c.transitions; }
            if (is_cold) {
                ++tensor_cold; ++run;
                ++c.frontier_distance[last >= first + i && last != SIZE_MAX ? bin(last - first - i) : bins - 1];
                ++c.cold_access[bin(pages[i].access_count)];
            } else if (run) { ++c.cold_runs; ++c.run_length[bin(run)]; run = 0; }
            previous_cold = is_cold;
        }
        if (run) { ++c.cold_runs; ++c.run_length[bin(run)]; }
        if (tensor_cold) { ++(t ? c.cold_value_invocations : c.cold_key_invocations); }
        cold += tensor_cold;
    }
    ++c.invocations;
    ++c.resident_percent[percent_bin(gpu, written)];
    for (auto seen = c.max_cold_pages.load(); cold > seen && !c.max_cold_pages.compare_exchange_weak(seen, cold);) {}
}

bool argus_disk_read_resident(const ggml_tensor * k, const ggml_tensor * v,
        const void ** pages, size_t capacity,
        void (*consume)(size_t, size_t, void *), void * context) {
    if (!pages || !consume || !argus_ggml_is_disk_tensor(k) || !argus_ggml_is_disk_tensor(v) ||
        k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || k->ne[2] <= 0 || k->ne[2] != v->ne[2] ||
        ggml_nbytes(k) != size_t(k->ne[2]) * k->nb[2] || ggml_nbytes(v) != size_t(v->ne[2]) * v->nb[2]) {
        throw std::invalid_argument("ARGUS resident read requires F16 disk tensors and a callback");
    }
    // ponytail: registry lock spans the read; use pinned page leases if concurrent requests need teardown independence.
    std::lock_guard<std::mutex> registry_guard(registry_mutex);
    auto & ks = store_for((k->view_src ? k->view_src : k)->buffer);
    auto & vs = store_for((v->view_src ? v->view_src : v)->buffer);
    std::unique_lock<std::mutex> kl(ks.mutex, std::defer_lock), vl(vs.mutex, std::defer_lock);
    if (&ks == &vs) { kl.lock(); } else { std::lock(kl, vl); }
    const size_t starts[] = {checked_offset(ks, k, 0, ggml_nbytes(k)), checked_offset(vs, v, 0, ggml_nbytes(v))};
    const ggml_tensor * tensors[] = {k, v};
    Store * stores[] = {&ks, &vs};
    size_t counts[2]{}, used = 0;
    for (int t = 0; t < 2; ++t) {
        counts[t] = (starts[t] % page_size + ggml_nbytes(tensors[t]) + page_size - 1) / page_size;
        if (counts[t] > capacity - used) { throw std::out_of_range("ARGUS resident pointer table capacity"); }
        used += counts[t];
    }
    if (argus_profile::enabled()) { residency_census(stores, starts, counts); }
    used = 0;
    {
        argus_profile::Scope timer(argus_profile::page_lookup);
        for (int t = 0; t < 2; ++t) {
            for (size_t i = 0; i < counts[t]; ++i) {
                const auto & page = stores[t]->pages[starts[t] / page_size + i];
                if (page.codec != GGML_TYPE_F16) { argus_profile::reject(argus_profile::reject_codec); return false; }
                if (!page.content_revision || page.active == 2) { pages[used++] = nullptr; }
                else if (page.resident && page.resident->tier() == ArgusTier::gpu) { pages[used++] = page.resident->data(); }
                else { argus_profile::reject(t ? argus_profile::reject_value_page : argus_profile::reject_key_page); return false; }
            }
        }
    }
    consume(counts[0], counts[1], context);
    // Preserve the staged path's successful 32-cell read history for policy.
    for (int64_t first = 0; first < k->ne[2]; first += 32) {
        const size_t count = std::min<int64_t>(32, k->ne[2] - first);
        record_access(ks, starts[0] + first * k->nb[2], count * k->nb[2]);
        record_access(vs, starts[1] + first * v->nb[2], count * v->nb[2]);
    }
    return true;
}
#endif
