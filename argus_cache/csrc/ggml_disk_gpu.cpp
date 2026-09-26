// ARGUS disk store, GPU side: page migration, borrowed resident reads, GPU appends and
// the background write-back of dirty pages. Store lifetime and host I/O live in
// ggml_disk_buffer.cpp; the shared internals in ggml_disk_store.h.
#include "ggml_disk_store.h"
#ifdef ARGUS_CUDA
#include "ggml_kv_policy.h"
#include "ggml_profile.h"
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

namespace argus_disk {
// Writes a page image to one disk slot and reads it back, verified like write_page.
// Caller holds io_mutex. Throws, publishing nothing, on any failure.
void write_slot(Store & store, size_t index, uint32_t slot, void * data, uint32_t digest) {
    ssize_t wrote;
    {
        argus_profile::Scope io(argus_profile::disk_write);
        do { wrote = pwrite(store.fd, data, page_size, offset(index, slot)); } while (wrote < 0 && errno == EINTR);
    }
    if (wrote != static_cast<ssize_t>(page_size)) { throw std::runtime_error("ARGUS short or failed direct page write"); }
    written_bytes += page_size;
    ssize_t got;
    {
        argus_profile::Scope io(argus_profile::disk_read);
        do { got = pread(store.fd, data, page_size, offset(index, slot)); } while (got < 0 && errno == EINTR);
    }
    if (got != static_cast<ssize_t>(page_size) || checksum(data) != digest) {
        throw std::runtime_error("ARGUS disk page write verification failed");
    }
    read_bytes += page_size;
}

// Copies a dirty page's resident bytes and returns their digest. Caller holds `mutex`.
uint32_t snapshot(Store & store, size_t index, void * data) {
    Page & descriptor = page_at(store, index);
    descriptor.resident->read(data, 0, page_size);
    const auto digest = checksum(data);
    if (!descriptor.digest_pending && digest != descriptor.checksum) {
        throw std::runtime_error("ARGUS resident checksum mismatch");
    }
    return digest;
}

void publish_slot(Page & descriptor, uint32_t slot, uint32_t digest) {
    descriptor.active = slot;
    descriptor.checksum = digest;
    descriptor.digest_pending = false;
    descriptor.dirty = false;
}

// Synchronous flush, for demotion and explicit sync points. Caller holds `mutex`. On
// failure the page stays dirty and the previously published slot is untouched.
void flush_page(Store & store, size_t index) {
    Page & descriptor = page_at(store, index);
    if (!descriptor.dirty) { return; }
    Bounce data;
    const auto digest = snapshot(store, index, data.data);
    std::lock_guard<std::mutex> io_guard(store.io_mutex);
    const uint32_t slot = 1 - (descriptor.active & 1);
    write_slot(store, index, slot, data.data, digest);
    publish_slot(descriptor, slot, digest);
}

// Background publisher of dirty pages, one per disk-backed store. Its stack is charged to
// staging like the prefetch worker's. A failure stops it until the next wake; the page
// stays dirty and the error is rethrown by the next GPU append.
struct Flusher {
    explicit Flusher(Store & store) : store_(store) {
        if (mprotect(stack_.data(), 4096, PROT_NONE)) { throw std::runtime_error("ARGUS flusher stack guard failed"); }
        pthread_attr_t attributes;
        if (pthread_attr_init(&attributes)) { throw std::runtime_error("ARGUS flusher attributes failed"); }
        const int configured = pthread_attr_setstack(&attributes, static_cast<char *>(stack_.data()) + 4096, stack_.size() - 4096);
        const int created = configured ? configured : pthread_create(&thread_, &attributes, run, this);
        pthread_attr_destroy(&attributes);
        if (created) { throw std::runtime_error(std::string("ARGUS flusher creation failed: ") + std::strerror(created)); }
    }
    ~Flusher() {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stop_ = true;
            condition_.notify_all();
        }
        if (pthread_join(thread_, nullptr)) { GGML_ABORT("ARGUS could not join the flusher"); }
    }
    void wake() {
        std::lock_guard<std::mutex> guard(mutex_);
        pending_ = true;
        condition_.notify_all();
    }
    std::exception_ptr take_error() {
        std::lock_guard<std::mutex> guard(mutex_);
        return std::exchange(error_, nullptr);
    }
private:
    static void * run(void * context) {
        auto & self = *static_cast<Flusher *>(context);
        std::unique_lock<std::mutex> guard(self.mutex_);
        for (;;) {
            self.condition_.wait(guard, [&] { return self.stop_ || self.pending_; });
            if (self.stop_) { return nullptr; }
            self.pending_ = false;
            guard.unlock();
            std::exception_ptr error;
            try {
                while (self.flush_one()) {}
            } catch (...) { error = std::current_exception(); }
            guard.lock();
            if (error) { self.error_ = error; }
        }
    }
    // One page: snapshot under the store lock, slot I/O under io_mutex alone, then publish
    // only if neither the content nor the published slot moved meanwhile.
    bool flush_one() {
        auto & store = store_;
        size_t index;
        uint32_t digest, seen;
        uint64_t revision;
        {
            std::lock_guard<std::mutex> store_guard(store.mutex);
            while (!store.dirty_pages.empty() && !page_at(store, store.dirty_pages.back()).dirty) {
                store.dirty_pages.pop_back(); // flushed synchronously since
            }
            if (store.dirty_pages.empty()) { return false; }
            index = store.dirty_pages.back();
            store.dirty_pages.pop_back();
            digest = snapshot(store, index, bounce_.data());
            revision = page_at(store, index).content_revision;
            seen = page_at(store, index).active;
        }
        const uint32_t slot = 1 - (seen & 1);
        bool written = false;
        try {
            std::lock_guard<std::mutex> io_guard(store.io_mutex);
            if (page_at(store, index).active == seen) { // else someone published it meanwhile
                write_slot(store, index, slot, bounce_.data(), digest);
                written = true;
            }
        } catch (...) {
            std::lock_guard<std::mutex> store_guard(store.mutex);
            if (page_at(store, index).dirty) { store.dirty_pages.push_back(index); }
            throw;
        }
        std::lock_guard<std::mutex> store_guard(store.mutex);
        std::lock_guard<std::mutex> io_guard(store.io_mutex);
        Page & descriptor = page_at(store, index);
        if (written && descriptor.dirty && descriptor.content_revision == revision && descriptor.active == seen) {
            publish_slot(descriptor, slot, digest);
        } else if (descriptor.dirty) {
            store.dirty_pages.push_back(index); // rewritten meanwhile: flush the new content later
        }
        return true;
    }
    Store & store_;
    ArgusStagingBuffer bounce_{page_size};
    ArgusStagingBuffer stack_{ArgusDiskPrefetch::stack_bytes};
    pthread_t thread_{};
    std::mutex mutex_;
    std::condition_variable condition_;
    std::exception_ptr error_;
    bool stop_ = false, pending_ = false;
};

// One flush of whole pages under GPU control. Per page the contract is write_page's:
// a separately budgeted destination is written, read back and checked before the old
// page is released, and nothing is published until every page of the run passes.
// Only the copies are shared: one queued upload and read-back per page, one wait.
// Returns false without touching the store when the run cannot be committed this way;
// the caller then writes the pages one at a time.
bool write_run(Store & store, size_t first, size_t count, const char * source) {
    argus_profile::Scope timer(argus_profile::write_page);
    constexpr auto exhausted = std::numeric_limits<uint64_t>::max();
    if (store.content_revision == exhausted) { throw std::runtime_error("ARGUS page generation exhausted"); }
    for (size_t i = 0; i < count; ++i) {
        const Page & descriptor = page_at(store, first + i);
        if (descriptor.content_revision == exhausted || descriptor.placement_revision == exhausted) {
            throw std::runtime_error("ARGUS page generation exhausted");
        }
    }
    std::vector<uint32_t> digests(count);
    for (size_t i = 0; i < count; ++i) { digests[i] = checksum(source + i * page_size); }
    std::vector<std::unique_ptr<ArgusTierBuffer>> targets;
    std::vector<void *> raw(count);
    targets.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        targets.push_back(std::make_unique<ArgusTierBuffer>(ArgusTier::gpu, page_size));
        raw[i] = targets[i]->data();
    }
    const size_t staging_bytes = 2 * count * page_size;
    if (!store.commit_host || store.commit_host->size() < staging_bytes) {
        try {
            store.commit_host.reset();
            store.commit_host = std::make_unique<ArgusTierBuffer>(ArgusTier::pinned, staging_bytes);
        } catch (const std::exception &) { return false; } // no pinned budget for the flush
    }
    const auto * verified = static_cast<const char *>(argus_cuda_commit_pages(
        store.commit_host->data(), raw.data(), source, count, page_size));
    if (!verified) { return false; } // the destinations are released with `targets`
    for (size_t i = 0; i < count; ++i) {
        if (checksum(verified + i * page_size) != digests[i]) {
            throw std::runtime_error("ARGUS GPU control verification failed");
        }
    }
    for (size_t i = 0; i < count; ++i) {
        Page & descriptor = page_at(store, first + i);
        delete descriptor.resident;
        descriptor.resident = targets[i].release();
        store.page_table.reset();
        ++descriptor.content_revision;
        ++descriptor.placement_revision;
        descriptor.checksum = digests[i];
        descriptor.digest_pending = false;
        descriptor.active = 3;
        ++store.content_revision;
    }
    store.last_written_page = first + count - 1;
    committed_pages += count;
    return true;
}

void destroy_flusher(Flusher * flusher) { delete flusher; }
} // namespace argus_disk
using namespace argus_disk;

static void move_page(Store & store, size_t index, ArgusTier tier, ArgusDiskPageRevision expected) {
    std::lock_guard<std::mutex> guard(store.mutex);
    auto & page = store.pages[index];
    if (expected.allocation != store.id || expected.page != index ||
        expected.content_revision != page.content_revision) { throw std::runtime_error("ARGUS stale migration"); }
    if (!has_content(page)) { throw std::invalid_argument("ARGUS cannot promote an unwritten page"); }
    if ((page.resident ? page.resident->tier() : ArgusTier::disk) == tier) { return; }
    if (store.gpu_control) { throw std::runtime_error("ARGUS GPU control cannot migrate out of GPU"); }
    if (page.placement_revision == UINT64_MAX) { throw std::runtime_error("ARGUS generation exhausted"); }
    // Dropping the last copy of dirty bytes would lose them: publish them first.
    if (tier == ArgusTier::disk) { flush_page(store, index); }
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
    if (store.write_event) { argus_cuda_event_wait(store.write_event); } // this copy runs on another stream
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
        void (*consume)(size_t, size_t, void *), void * context, const ArgusColdStaging * cold) {
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
    const auto page = [&](int t, size_t i) -> const Page & { return stores[t]->pages[starts[t] / page_size + i]; };
    const auto written = [](const Page & p) { return has_content(p); };
    const auto on_gpu = [](const Page & p) { return p.resident && p.resident->tier() == ArgusTier::gpu; };
    size_t cold_pages = 0;
    {
        argus_profile::Scope timer(argus_profile::page_lookup);
        for (int t = 0; t < 2; ++t) {
            for (size_t i = 0; i < counts[t]; ++i) {
                if (page(t, i).codec != GGML_TYPE_F16) { argus_profile::reject(argus_profile::reject_codec); return false; }
                if (!written(page(t, i)) || on_gpu(page(t, i))) { continue; }
                if (!cold) { argus_profile::reject(t ? argus_profile::reject_value_page : argus_profile::reject_key_page); return false; }
                ++cold_pages;
            }
        }
    }
    char * host = nullptr;
    if (cold_pages && !(host = static_cast<char *>(cold->reserve(cold_pages, context)))) {
        argus_profile::reject(argus_profile::reject_cold_budget);
        return false;
    }
    // Verification failures throw here, before anything is queued on the GPU.
    std::vector<size_t> cold_entries;
    cold_entries.reserve(cold_pages);
    used = 0;
    for (int t = 0; t < 2; ++t) {
        for (size_t i = 0; i < counts[t]; ++i, ++used) {
            const auto & descriptor = page(t, i);
            if (!written(descriptor)) { pages[used] = nullptr; }
            else if (on_gpu(descriptor)) { pages[used] = descriptor.resident->data(); }
            else {
                read_page(*stores[t], starts[t] / page_size + i, host + cold_entries.size() * page_size);
                cold_entries.push_back(used);
            }
        }
    }
    if (cold_pages) {
        const char * device = cold->upload(cold_pages, context);
        for (size_t slot = 0; slot < cold_pages; ++slot) { pages[cold_entries[slot]] = device + slot * page_size; }
    }
    consume(counts[0], counts[1], context);
    argus_profile::resident_cold_pages[argus_profile::phase] += cold_pages;
    // Preserve the staged path's successful 32-cell read history for policy.
    for (int64_t first = 0; first < k->ne[2]; first += 32) {
        const size_t count = std::min<int64_t>(32, k->ne[2] - first);
        record_access(ks, starts[0] + first * k->nb[2], count * k->nb[2]);
        record_access(vs, starts[1] + first * v->nb[2], count * v->nb[2]);
    }
    return true;
}

bool argus_disk_gpu_appendable(const ggml_tensor * target) {
    if (!argus_ggml_is_disk_tensor(target)) { return false; }
    const size_t row = target->nb[1];
    auto & store = store_for((target->view_src ? target->view_src : target)->buffer);
    // GPU control, or a policy-managed store with a GPU tier. Policy off stays the
    // reference disk path: a GPU append would make placement decisions it must not make.
    const bool gpu_backed = store.gpu_control || (argus_kv_policy_enabled() && argus_tier_budget(ArgusTier::gpu).limit);
    return gpu_backed && row && page_size % row == 0 && row == ggml_row_size(target->type, target->ne[0]) &&
           checked_offset(store, target, 0, ggml_nbytes(target)) % row == 0;
}

bool argus_disk_gpu_rows(const ggml_tensor * target, const int64_t * rows, size_t count, void * stream,
                         void (*encode)(void * const * destinations, void * context), void * context) {
    if (!argus_disk_gpu_appendable(target)) { throw std::invalid_argument("ARGUS GPU append needs a GPU-backed store"); }
    auto & store = store_for((target->view_src ? target->view_src : target)->buffer);
    if (!store.gpu_control) {
        if (!store.flusher) { store.flusher = new Flusher(store); }
        if (auto error = store.flusher->take_error()) { std::rethrow_exception(error); }
    }
    const size_t row = target->nb[1];
    std::lock_guard<std::mutex> guard(store.mutex);
    const size_t base = checked_offset(store, target, 0, ggml_nbytes(target));
    // Pages this call touches, and a GPU page for each one that has none yet.
    std::vector<size_t> pages;
    std::vector<std::unique_ptr<ArgusTierBuffer>> fresh;
    size_t needed = 0;
    for (size_t i = 0; i < count; ++i) {
        if (rows[i] < 0 || rows[i] >= target->ne[1]) { throw std::out_of_range("ARGUS KV row index out of range"); }
        const size_t page = (base + size_t(rows[i]) * row) / page_size;
        if (std::find(pages.begin(), pages.end(), page) != pages.end()) { continue; }
        const Page & descriptor = page_at(store, page);
        if (descriptor.content_revision == std::numeric_limits<uint64_t>::max() ||
            descriptor.placement_revision == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("ARGUS page generation exhausted");
        }
        const bool on_gpu = descriptor.resident && descriptor.resident->tier() == ArgusTier::gpu;
        if (store.gpu_control && has_content(descriptor) && !on_gpu) { throw std::logic_error("ARGUS GPU control page has no GPU copy"); }
        pages.push_back(page);
        needed += on_gpu ? 0 : 1;
    }
    if (!store.gpu_control) {
        // The new pages and the row table must fit beside the scratch headroom the policy
        // keeps free; appending into it would make the next prepare() evict dirty pages.
        const auto budget = argus_tier_budget(ArgusTier::gpu);
        const size_t wanted = (needed + (count * sizeof(void *) + page_size - 1) / page_size) * page_size +
                              argus_kv_policy_headroom(ArgusTier::gpu);
        if (budget.live > budget.limit || budget.limit - budget.live < wanted) { return false; } // host append
    }
    for (size_t page : pages) {
        const Page & descriptor = page_at(store, page);
        fresh.emplace_back();
        if (descriptor.resident && descriptor.resident->tier() == ArgusTier::gpu) { continue; }
        fresh.back() = std::make_unique<ArgusTierBuffer>(ArgusTier::gpu, page_size);
        if (has_content(descriptor)) { // keep the rows this call does not write: promotion by write
            Bounce data;
            read_page(store, page, data.data);
            fresh.back()->write(data.data, page_size);
        } else {
            argus_cuda_zero(fresh.back()->data(), page_size, stream); // an unwritten page reads as zeros
        }
    }
    if (store.content_revision == std::numeric_limits<uint64_t>::max()) { throw std::runtime_error("ARGUS store generation exhausted"); }
    std::vector<void *> destinations(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t position = base + size_t(rows[i]) * row, page = position / page_size;
        const size_t slot = std::find(pages.begin(), pages.end(), page) - pages.begin();
        auto * buffer = fresh[slot] ? fresh[slot].get() : page_at(store, page).resident;
        destinations[i] = static_cast<char *>(buffer->data()) + position % page_size;
    }
    // Encoded and finished under the lock: no reader or flusher sees a half-written page.
    encode(destinations.data(), context);
    for (size_t i = 0; i < pages.size(); ++i) {
        Page & descriptor = page_at(store, pages[i]);
        if (fresh[i]) {
            delete descriptor.resident;
            descriptor.resident = fresh[i].release();
            store.page_table.reset();
            ++descriptor.placement_revision;
        }
        ++descriptor.content_revision;
        descriptor.digest_pending = true;
        if (store.gpu_control) {
            descriptor.active = 3;
        } else if (!descriptor.dirty) {
            descriptor.dirty = true;
            store.dirty_pages.push_back(pages[i]);
        }
        ++store.content_revision;
        store.last_written_page = pages[i];
    }
    committed_pages += pages.size();
    if (store.flusher) { store.flusher->wake(); }
    return true;
}

bool argus_disk_gpu_rows_on_device(const ggml_tensor * target, void * stream,
        void (*launch)(void * const * pages, size_t base, int * error, void * context), void * context) {
    if (!argus_disk_gpu_appendable(target)) { throw std::invalid_argument("ARGUS GPU append needs a GPU-backed store"); }
    auto & store = store_for((target->view_src ? target->view_src : target)->buffer);
    if (!store.gpu_control) { return false; }
    std::lock_guard<std::mutex> guard(store.mutex);
    if (store.append_error) { // a row outside its tensor is refused here, one append late
        argus_cuda_event_wait(store.write_event);
        if (*static_cast<volatile int *>(store.append_error->data())) { throw std::out_of_range("ARGUS KV row index out of range"); }
    }
    const size_t count = rounded(store.bytes) / page_size;
    if (!store.page_table) {
        // Put every page on the GPU (unwritten ones zeroed), once: appends then need no host round trip.
        size_t missing = 0;
        for (size_t i = 0; i < count; ++i) { missing += !store.pages[i].resident; }
        const auto budget = argus_tier_budget(ArgusTier::gpu);
        const size_t wanted = (missing + (count * sizeof(void *) + page_size - 1) / page_size) * page_size;
        if (budget.live > budget.limit || budget.limit - budget.live < wanted) { return false; }
        std::vector<void *> addresses(count);
        for (size_t i = 0; i < count; ++i) {
            Page & page = store.pages[i];
            if (!page.resident) {
                auto fresh = std::make_unique<ArgusTierBuffer>(ArgusTier::gpu, page_size);
                argus_cuda_zero(fresh->data(), page_size, stream);
                page.resident = fresh.release();
                ++page.placement_revision;
            }
            addresses[i] = page.resident->data();
        }
        auto table = std::make_unique<ArgusTierBuffer>(ArgusTier::gpu, count * sizeof(void *));
        table->write(addresses.data(), count * sizeof(void *));
        store.page_table = std::move(table);
        if (!store.append_error) {
            store.append_error = std::make_unique<ArgusTierBuffer>(ArgusTier::pinned, page_size);
            *static_cast<int *>(store.append_error->data()) = 0;
        }
        if (!store.write_event) { store.write_event = argus_cuda_event_create(); }
    }
    const size_t base = checked_offset(store, target, 0, ggml_nbytes(target));
    launch(static_cast<void * const *>(store.page_table->data()), base, static_cast<int *>(store.append_error->data()), context);
    argus_cuda_event_record(store.write_event, stream);
    // The rows are known only on the device: every page the tensor spans may have changed.
    // Host readers wait for write_event; attention runs later on the same stream.
    const size_t first = base / page_size, last = (base + ggml_nbytes(target) - 1) / page_size;
    for (size_t i = first; i <= last; ++i) {
        Page & descriptor = page_at(store, i);
        if (descriptor.content_revision == std::numeric_limits<uint64_t>::max()) { throw std::runtime_error("ARGUS page generation exhausted"); }
        ++descriptor.content_revision;
        descriptor.digest_pending = true;
        descriptor.active = 3;
    }
    if (store.content_revision == std::numeric_limits<uint64_t>::max()) { throw std::runtime_error("ARGUS store generation exhausted"); }
    ++store.content_revision;
    store.last_written_page = last;
    return true;
}

void argus_disk_flush(const ggml_tensor * tensor) {
    if (!argus_ggml_is_disk_tensor(tensor)) { throw std::invalid_argument("ARGUS flush requires disk storage"); }
    auto & store = store_for((tensor->view_src ? tensor->view_src : tensor)->buffer);
    std::lock_guard<std::mutex> guard(store.mutex);
    for (size_t index : store.dirty_pages) { flush_page(store, index); }
    store.dirty_pages.clear();
    // Everything is published, so an earlier background failure has been recovered from.
    if (store.flusher) { store.flusher->take_error(); }
}
#endif
