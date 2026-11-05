// v0.8 KV tiers over llama.cpp's own KV cache: registry of device mirrors, their writes and
// refreshes, and the split decode graph (see ggml_kv_tiers.h).
#include "ggml_kv_tiers.h"
#include "ggml_decode_attention.h"
#include "argus_config.h"
#ifdef ARGUS_CUDA
#include "ggml_split_attention.h"
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
// Device memory of the mirrors allocated together; freed with the last of them.
struct Allocation {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    Allocation() = default;
    Allocation(const Allocation &) = delete;
    Allocation & operator=(const Allocation &) = delete;
    ~Allocation() {
        if (buffer) { ggml_backend_buffer_free(buffer); }
        if (ctx) { ggml_free(ctx); }
    }
};

struct Entry {
    ggml_backend_dev_t dev;
    ggml_tensor * device = nullptr; // null until argus_kv_allocate_mirrors gives it a mirror
    int64_t cells = 0;
    std::shared_ptr<Allocation> allocation;
};

struct Mirror {
    ggml_tensor * device;
    int64_t cells;
};

std::mutex registry_lock;
bool detached = false;                            // under registry_lock: mirrors lent out (P3)
std::weak_ptr<Allocation> arena;                  // under registry_lock: the device memory mirrors live in
std::unordered_map<const ggml_tensor *, Entry> registry; // host tensor -> entry, under registry_lock

// Copy `cells` rows from `host` (F16, Q8_0, or Q4_0) into `device` (always F16 on GPU).
void sync_mirror_from_host(const ggml_tensor * host, ggml_tensor * device, int64_t cells) {
    if (cells <= 0 || !host || !device) { return; }
    if (host->type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(device, host->data, 0, static_cast<size_t>(cells) * host->nb[1]);
        return;
    }
#ifdef ARGUS_CUDA
    if (argus_cuda_sync_mirror_from_host(host, device, cells)) { return; }
#endif
    const int64_t ne0 = host->ne[0], total = cells * ne0;
    std::vector<float> tmp_f32(total);
    std::vector<ggml_fp16_t> tmp_f16(total);
    const auto to_float = ggml_get_type_traits(host->type)->to_float;
    if (!to_float) { throw std::runtime_error("ARGUS KV mirror: host type has no to_float conversion"); }
    for (int64_t c = 0; c < cells; ++c) {
        const char * src = static_cast<const char *>(host->data) + c * host->nb[1];
        to_float(src, tmp_f32.data() + c * ne0, ne0);
    }
    ggml_fp32_to_fp16_row(tmp_f32.data(), tmp_f16.data(), total);
    ggml_backend_tensor_set(device, tmp_f16.data(), 0, total * sizeof(ggml_fp16_t));
}

// Entries keep their mirror while the cache that registered them lives, so the copy is enough.
bool find(const ggml_tensor * host, Mirror & mirror) {
    std::lock_guard<std::mutex> guard(registry_lock);
    const auto it = registry.find(host);
    if (detached || it == registry.end() || !it->second.device) { return false; }
    mirror = {it->second.device, it->second.cells};
    return true;
}

// Rows outside the mirror land on its scratch row (row `cells`).
void clamp_rows(ggml_tensor * dst, int ith, int, void * userdata) {
    if (ith != 0) { return; }
    const auto cells = static_cast<int64_t>(reinterpret_cast<intptr_t>(userdata));
    const ggml_tensor * indices = dst->src[0];
    auto * out = static_cast<int64_t *>(dst->data);
    for (int64_t i = 0; i < indices->ne[0]; ++i) {
        const int64_t row = indices->type == GGML_TYPE_I64 ? static_cast<const int64_t *>(indices->data)[i]
                                                           : static_cast<const int32_t *>(indices->data)[i];
        out[i] = std::min(row, cells);
    }
}

// Owned by the cache that registered the tensor; not copyable, so only it erases the entry.
struct Registration {
    explicit Registration(const ggml_tensor * tensor) : host(tensor) {}
    Registration(const Registration &) = delete;
    Registration & operator=(const Registration &) = delete;
    const ggml_tensor * host;
    ~Registration() {
        std::lock_guard<std::mutex> guard(registry_lock);
        registry.erase(host);
    }
};

// Mirrors for the entries of one device that have none yet, `cells` rows each (fewer for a
// shorter tensor), filled from the host, in one arena of at least `shared` bytes (for what
// borrows the memory while the mirrors are detached). False if the device cannot allocate it.
bool allocate(ggml_backend_dev_t dev, std::vector<std::pair<const ggml_tensor *, Entry *>> & pending, int64_t cells,
              size_t shared) {
    auto allocation = std::make_shared<Allocation>();
    allocation->ctx = ggml_init({ggml_tensor_overhead() * pending.size(), nullptr, true});
    if (!allocation->ctx) { return false; }
    ggml_backend_buffer_type_t type = ggml_backend_dev_buffer_type(dev);
    const size_t alignment = ggml_backend_buft_get_alignment(type);
    std::vector<ggml_tensor *> mirrors;
    std::vector<size_t> offsets;
    size_t size = 0;
    for (const auto & [host, entry] : pending) {
        ggml_tensor * mirror = ggml_new_tensor_2d(allocation->ctx, GGML_TYPE_F16, host->ne[0], std::min(cells, host->ne[1]) + 1);
        ggml_format_name(mirror, "argus_mirror_%s", host->name);
        mirrors.push_back(mirror);
        offsets.push_back(size);
        size += (ggml_nbytes(mirror) + alignment - 1) / alignment * alignment;
    }
    allocation->buffer = ggml_backend_buft_alloc_buffer(type, std::max(size, shared));
    if (!allocation->buffer) { return false; }
    char * base = static_cast<char *>(ggml_backend_buffer_get_base(allocation->buffer));
    for (size_t i = 0; i < mirrors.size(); ++i) {
        if (ggml_backend_tensor_alloc(allocation->buffer, mirrors[i], base + offsets[i]) != GGML_STATUS_SUCCESS) { return false; }
    }
    ggml_backend_buffer_clear(allocation->buffer, 0);
    arena = allocation;
    for (size_t i = 0; i < pending.size(); ++i) {
        const ggml_tensor * host = pending[i].first;
        Entry & entry = *pending[i].second;
        entry.device = mirrors[i];
        entry.cells = mirrors[i]->ne[1] - 1;
        entry.allocation = allocation;
        sync_mirror_from_host(host, entry.device, entry.cells);
    }
    return true;
}
} // namespace

bool argus_kv_tiers_requested() {
    const ArgusConfig cfg = argus_get_config();
    return cfg.vram_auto || cfg.vram_bytes > 0;
}

std::shared_ptr<void> argus_kv_register(const ggml_tensor * host, ggml_backend_dev_t dev) {
    if (!host || !dev || host->ne[2] != 1 || host->nb[1] != ggml_row_size(host->type, host->ne[0])) {
        throw std::invalid_argument("ARGUS KV tiers need a single-stream KV tensor with packed rows");
    }
    {
        std::lock_guard<std::mutex> guard(registry_lock);
        if (!registry.emplace(host, Entry{dev, nullptr, 0, nullptr}).second) { throw std::logic_error("ARGUS KV tensor registered twice"); }
    }
    return std::make_shared<Registration>(host);
}

void argus_kv_allocate_mirrors(size_t shared_bytes) {
    std::lock_guard<std::mutex> guard(registry_lock);
    const ArgusConfig cfg = argus_get_config();
    std::unordered_map<ggml_backend_dev_t, std::vector<std::pair<const ggml_tensor *, Entry *>>> pending;
    std::unordered_map<ggml_backend_dev_t, size_t> mirrored;
    for (auto & [host, entry] : registry) {
        if (entry.device) { mirrored[entry.dev] += ggml_nbytes(entry.device); }
        else { pending[entry.dev].emplace_back(host, &entry); }
    }
    for (auto & [dev, entries] : pending) {
        size_t bytes_per_cell = 0;
        int64_t longest = 0;
        for (const auto & [host, entry] : entries) {
            bytes_per_cell += ggml_row_size(GGML_TYPE_F16, host->ne[0]);
            longest = std::max(longest, host->ne[1]);
        }
        const size_t full_kv_bytes = static_cast<size_t>(longest + 1) * bytes_per_cell;
        size_t budget = 0;
        if (cfg.vram_auto) {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            budget = argus_plan_vram_budget(free, shared_bytes, cfg, full_kv_bytes);
        } else {
            const size_t limit = cfg.vram_bytes;
            budget = limit > mirrored[dev] ? limit - mirrored[dev] : 0;
        }
        // One scratch row per mirror is part of the cost of a cell range.
        const int64_t fit = static_cast<int64_t>(budget / bytes_per_cell) - 1;
        const int64_t cells = fit >= longest ? longest : std::max<int64_t>(0, fit / 256 * 256);
        const size_t shared = shared_bytes <= budget ? shared_bytes : 0;
        // On stderr whatever the log level: runs can prove which tiers exist.
        if (cells <= 0 || !allocate(dev, entries, cells, shared)) {
            std::fprintf(stderr, "ARGUS KV tiers: no device mirror (budget %.1f MiB): host KV and K1 only\n",
                         budget / 1048576.0);
            continue;
        }
        std::fprintf(stderr, "ARGUS KV tiers: cells [0, %lld) of %lld mirrored in device memory (%.1f MiB, budget %.1f MiB%s)\n",
                     (long long) cells, (long long) longest, cells * double(bytes_per_cell) / 1048576.0,
                     budget / 1048576.0, cfg.vram_auto ? ", auto" : "");
    }
}

ggml_backend_buffer_t argus_kv_arena(size_t bytes) {
    std::lock_guard<std::mutex> guard(registry_lock);
    const auto allocation = arena.lock();
    return allocation && ggml_backend_buffer_get_size(allocation->buffer) >= bytes ? allocation->buffer : nullptr;
}

void argus_kv_mirrors_detach() {
    std::lock_guard<std::mutex> guard(registry_lock);
    detached = true;
}

void argus_kv_mirrors_attach() {
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> mirrors;
    {
        std::lock_guard<std::mutex> guard(registry_lock);
        if (!detached) { return; }
        detached = false;
        for (const auto & [host, entry] : registry) {
            if (entry.device) { mirrors.emplace_back(host, entry.device); }
        }
    }
    // Whatever used the arena overwrote the mirrors: fill them from the host again.
    for (const auto & [host, device] : mirrors) {
        sync_mirror_from_host(host, device, device->ne[1] - 1);
    }
}

ggml_backend_buffer_type_t argus_kv_host_buffer_type() {
#ifdef ARGUS_CUDA
    return ggml_backend_cuda_host_buffer_type();
#else
    return nullptr;
#endif
}

ggml_tensor * argus_kv_append(ggml_context * ctx, ggml_tensor * host, ggml_tensor * rows, ggml_tensor * indices,
                              int64_t n_kv, int64_t min_idx) {
    Mirror mirror;
    if (!find(host, mirror)) { return nullptr; }
#ifdef ARGUS_CUDA
    const bool decode_batch = rows->ne[1] <= argus_k1_max_tokens();
    const bool mirror_prefill = argus_get_config().prefill_on_gpu && n_kv > 0 && n_kv <= mirror.cells;
    if (host->buffer && ggml_backend_buffer_get_type(host->buffer) == ggml_backend_cuda_host_buffer_type() &&
        (decode_batch || mirror_prefill)) {
        if (ggml_tensor * appended = argus_ggml_cuda_tier_append(ctx, mirror.device, rows, indices, host)) { return appended; }
    }
#else
    (void) n_kv;
#endif
    ggml_tensor * stored = ggml_set_rows(ctx, host, rows, indices);
    if (min_idx >= mirror.cells) { return stored; }
    if (indices->type != GGML_TYPE_I64 && indices->type != GGML_TYPE_I32) {
        throw std::runtime_error("ARGUS KV mirror: unsupported row index type");
    }
    // Depending on `stored` orders the mirror write after the host write.
    ggml_tensor * args[] = {indices, stored};
    ggml_tensor * clamped = ggml_custom_4d(ctx, GGML_TYPE_I64, indices->ne[0], 1, 1, 1, args, 2, clamp_rows, 1,
                                           reinterpret_cast<void *>(static_cast<intptr_t>(mirror.cells)));
    return ggml_set_rows(ctx, mirror.device, rows, clamped);
}

bool argus_kv_prefill_mirror_views(ggml_context * ctx, const ggml_tensor * q, ggml_tensor * & k, ggml_tensor * & v,
                                   bool unsupported_features) {
#ifdef ARGUS_CUDA
    Mirror keys, values;
    if (unsupported_features || !argus_get_config().prefill_on_gpu || q->ne[2] <= argus_k1_max_tokens() ||
        !k || !v || !k->view_src || !v->view_src || k->view_offs != 0 || v->view_offs != 0 ||
        k->ne[3] != 1 || v->ne[3] != 1 || !find(k->view_src, keys) || !find(v->view_src, values) ||
        k->ne[2] > keys.cells || v->ne[2] > values.cells) {
        return false;
    }
    const size_t k_nb1 = static_cast<size_t>(k->ne[0]) * sizeof(ggml_fp16_t), k_nb2 = keys.device->nb[1];
    const size_t v_nb1 = static_cast<size_t>(v->ne[0]) * sizeof(ggml_fp16_t), v_nb2 = values.device->nb[1];
    k = ggml_view_4d(ctx, keys.device, k->ne[0], k->ne[1], k->ne[2], 1, k_nb1, k_nb2, k_nb2 * k->ne[2], 0);
    v = ggml_view_4d(ctx, values.device, v->ne[0], v->ne[1], v->ne[2], 1, v_nb1, v_nb2, v_nb2 * v->ne[2], 0);
    return true;
#else
    (void) ctx; (void) q; (void) k; (void) v; (void) unsupported_features;
    return false;
#endif
}

void argus_kv_mirror_refresh(const ggml_tensor * host) {
    const ggml_tensor * base = (host && host->view_src) ? host->view_src : host;
    Mirror mirror;
    if (!find(base, mirror)) { return; }
    sync_mirror_from_host(base, mirror.device, mirror.cells);
}

#ifdef ARGUS_CUDA
namespace {
ggml_backend_t cuda_backend(ggml_backend_sched_t sched) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_is_cuda(backend)) { return backend; }
    }
    throw std::runtime_error("ARGUS KV tiers need a CUDA backend in the scheduler");
}

// GPU nodes of a decode graph built by decode_graph: the reduce and the partials under it.
void assign_gpu(ggml_backend_sched_t sched, ggml_tensor * reduce) {
    ggml_backend_t gpu = cuda_backend(sched);
    ggml_backend_sched_set_tensor_backend(sched, reduce->src[0], gpu);
    ggml_backend_sched_set_tensor_backend(sched, reduce, gpu);
}
} // namespace
#endif

ggml_tensor * argus_kv_tiered_attention(ggml_context * ctx, ggml_backend_sched_t sched, ggml_backend_t cpu,
                                        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
                                        bool unsupported_features, float scale) {
#ifdef ARGUS_CUDA
    Mirror keys, values;
    if (unsupported_features || !mask || !k->view_src || !v->view_src || q->ne[2] > argus_k1_max_tokens() ||
        k->ne[3] != 1 || v->ne[3] != 1 || !find(k->view_src, keys) || !find(v->view_src, values)) {
        return nullptr;
    }
    const int64_t n_kv = k->ne[2], cells = std::min({keys.cells, values.cells, n_kv});
    // The mirror is always F16 on the GPU even when host KV is quantized (Q8_0 / Q4_0).
    const size_t k_f16_nb1 = static_cast<size_t>(k->ne[0]) * sizeof(ggml_fp16_t), k_f16_nb2 = keys.device->nb[1];
    const size_t v_f16_nb1 = static_cast<size_t>(v->ne[0]) * sizeof(ggml_fp16_t), v_f16_nb2 = values.device->nb[1];
    ggml_tensor * k_gpu = ggml_view_3d(ctx, keys.device, k->ne[0], k->ne[1], cells, k_f16_nb1, k_f16_nb2, 0);
    ggml_tensor * v_gpu = ggml_view_3d(ctx, values.device, v->ne[0], v->ne[1], cells, v_f16_nb1, v_f16_nb2, 0);
    ggml_tensor * gpu_part = cells == n_kv ? argus_ggml_cuda_decode_attention(ctx, q, k_gpu, v_gpu, mask, scale)
                                           : argus_ggml_cuda_decode_partials(ctx, q, k_gpu, v_gpu, mask, 0, scale);
    if (!gpu_part) {
        // Shapes the GPU kernels do not cover: K1 over the whole host view, which (unlike a
        // GPU read of the host view) runs after the decode batch's GPU appends.
        ggml_tensor * out = argus_ggml_decode_attention_after(ctx, q, k, v, mask, 0, nullptr, scale);
        if (out) {
            ggml_backend_sched_set_tensor_backend(sched, out->src[4], cpu);
            ggml_backend_sched_set_tensor_backend(sched, out, cpu);
        }
        return out;
    }
    if (cells == n_kv) {
        static std::once_flag announced;
        std::call_once(announced, [] { std::fprintf(stderr, "ARGUS S1: decode attention on the GPU mirror\n"); });
        assign_gpu(sched, gpu_part);
        return gpu_part;
    }
    ggml_tensor * prior = gpu_part;
    ggml_tensor * k_host = ggml_view_3d(ctx, k->view_src, k->ne[0], k->ne[1], n_kv - cells, k->nb[1], k->nb[2],
                                        k->view_offs + cells * k->nb[2]);
    ggml_tensor * v_host = ggml_view_3d(ctx, v->view_src, v->ne[0], v->ne[1], n_kv - cells, v->nb[1], v->nb[2],
                                        v->view_offs + cells * v->nb[2]);
    ggml_tensor * out = argus_ggml_decode_attention_after(ctx, q, k_host, v_host, mask, cells, prior, scale);
    if (!out) { return nullptr; }
    static std::once_flag announced;
    std::call_once(announced, [] { std::fprintf(stderr, "ARGUS S1: split decode attention (GPU mirror + K1 on host)\n"); });
    assign_gpu(sched, prior);
    ggml_backend_sched_set_tensor_backend(sched, out->src[4], cpu);
    ggml_backend_sched_set_tensor_backend(sched, out, cpu);
    return out;
#else
    (void) ctx; (void) sched; (void) cpu; (void) q; (void) k; (void) v; (void) mask; (void) unsupported_features; (void) scale;
    return nullptr;
#endif
}

bool argus_kv_managed() { return argus_k1_enabled() || argus_kv_tiers_requested(); }

bool argus_kv_prefill_on_gpu(const ggml_tensor * k, int64_t n_tokens) {
    if (!argus_get_config().prefill_on_gpu) { return false; }
    const ggml_tensor * storage = k->view_src ? k->view_src : k;
    return n_tokens > argus_k1_max_tokens() && storage->buffer && ggml_backend_buffer_is_host(storage->buffer);
}
