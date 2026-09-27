// v0.8 KV tiers over llama.cpp's own KV cache: registry of device mirrors, their writes and
// refreshes, and the split decode graph (see ggml_kv_tiers.h).
#include "ggml_kv_tiers.h"
#include "ggml_decode_attention.h"
#ifdef ARGUS_CUDA
#include "ggml_split_attention.h"
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
struct Mirror {
    ggml_tensor * device;
    int64_t cells;
};

std::mutex registry_lock;
std::unordered_map<const ggml_tensor *, Mirror> registry; // host tensor -> mirror, under registry_lock

size_t byte_setting(const char * name) {
    const char * text = std::getenv(name);
    if (!text || !*text) { return 0; }
    size_t value = 0;
    const auto parsed = std::from_chars(text, text + std::strlen(text), value);
    if (parsed.ec != std::errc{} || *parsed.ptr) { throw std::invalid_argument(std::string(name) + " must be a byte count"); }
    return value;
}

// Mirrors are never erased while a cache that registered them lives, so the copy is enough.
bool find(const ggml_tensor * host, Mirror & mirror) {
    std::lock_guard<std::mutex> guard(registry_lock);
    const auto it = registry.find(host);
    if (it == registry.end()) { return false; }
    mirror = it->second;
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

// Owned by the cache that registered the mirror; not copyable, so only it erases the entry.
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
} // namespace

int64_t argus_kv_mirror_cells(int64_t kv_size, size_t bytes_per_cell) {
    const size_t budget = byte_setting("ARGUS_KV_VRAM_BYTES");
    if (!budget || !bytes_per_cell || kv_size <= 0) { return 0; }
    size_t used = 0;
    {
        std::lock_guard<std::mutex> guard(registry_lock);
        for (const auto & [host, mirror] : registry) { used += ggml_nbytes(mirror.device); }
    }
    if (used >= budget) { return 0; }
    // One scratch row per mirror is part of the cost of a cell range.
    const int64_t fit = static_cast<int64_t>((budget - used) / bytes_per_cell) - 1;
    const int64_t cells = fit >= kv_size ? kv_size : std::max<int64_t>(0, fit / 256 * 256);
    if (cells > 0) {  // on stderr whatever the log level: runs can prove the tiers exist
        std::fprintf(stderr, "ARGUS KV tiers: cells [0, %lld) of %lld mirrored in device memory (%.1f MiB)\n",
                     (long long) cells, (long long) kv_size, cells * double(bytes_per_cell) / (1 << 20));
    }
    return cells;
}

std::shared_ptr<void> argus_kv_mirror_register(const ggml_tensor * host, ggml_tensor * device, int64_t cells) {
    if (!host || !device || cells <= 0 || device->type != host->type || device->ne[0] != host->ne[0] ||
        device->ne[1] != cells + 1 || cells > host->ne[1] || host->ne[2] != 1) {
        throw std::invalid_argument("ARGUS KV mirror must match its host tensor, with one scratch row");
    }
    {
        std::lock_guard<std::mutex> guard(registry_lock);
        if (!registry.emplace(host, Mirror{device, cells}).second) {
            throw std::logic_error("ARGUS KV mirror registered twice");
        }
    }
    return std::make_shared<Registration>(host);
}

ggml_backend_buffer_type_t argus_kv_host_buffer_type() {
#ifdef ARGUS_CUDA
    return ggml_backend_cuda_host_buffer_type();
#else
    return nullptr;
#endif
}

ggml_tensor * argus_kv_append(ggml_context * ctx, ggml_tensor * host, ggml_tensor * rows, ggml_tensor * indices) {
    Mirror mirror;
    if (!find(host, mirror)) { return nullptr; }
#ifdef ARGUS_CUDA
    // Decode batches only. Their attention is always built here: the mirror is read later on
    // the same stream, the host cells by K1 after GGML synchronizes the GPU for its inputs.
    // A prefill batch's attention may run on the GPU from a copy of the host view that GGML
    // takes when the split starts, before an append queued in the same split has run.
    if (host->buffer && ggml_backend_buffer_get_type(host->buffer) == ggml_backend_cuda_host_buffer_type() &&
        rows->ne[1] <= argus_k1_max_tokens()) {
        if (ggml_tensor * appended = argus_ggml_cuda_tier_append(ctx, mirror.device, rows, indices, host)) { return appended; }
    }
#endif
    ggml_tensor * stored = ggml_set_rows(ctx, host, rows, indices);
    if (indices->type != GGML_TYPE_I64 && indices->type != GGML_TYPE_I32) {
        throw std::runtime_error("ARGUS KV mirror: unsupported row index type");
    }
    // Depending on `stored` orders the mirror write after the host write.
    ggml_tensor * args[] = {indices, stored};
    ggml_tensor * clamped = ggml_custom_4d(ctx, GGML_TYPE_I64, indices->ne[0], 1, 1, 1, args, 2, clamp_rows, 1,
                                           reinterpret_cast<void *>(static_cast<intptr_t>(mirror.cells)));
    return ggml_set_rows(ctx, mirror.device, rows, clamped);
}

void argus_kv_mirror_refresh(const ggml_tensor * host) {
    Mirror mirror;
    if (!find(host, mirror)) { return; }
    ggml_backend_tensor_set(mirror.device, host->data, 0, static_cast<size_t>(mirror.cells) * host->nb[1]);
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
    // The mirror has the host rows' layout, so the host view's strides apply from its first row.
    ggml_tensor * k_gpu = ggml_view_3d(ctx, keys.device, k->ne[0], k->ne[1], cells, k->nb[1], k->nb[2], 0);
    ggml_tensor * v_gpu = ggml_view_3d(ctx, values.device, v->ne[0], v->ne[1], cells, v->nb[1], v->nb[2], 0);
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

bool argus_kv_managed() { return argus_k1_enabled() || byte_setting("ARGUS_KV_VRAM_BYTES") > 0; }

bool argus_kv_prefill_on_gpu(const ggml_tensor * k, int64_t n_tokens) {
    const char * text = std::getenv("ARGUS_KV_PREFILL");
    if (!text || !*text || std::string(text) == "cpu") { return false; }
    if (std::string(text) != "gpu") { throw std::runtime_error("ARGUS_KV_PREFILL must be cpu or gpu"); }
    const ggml_tensor * storage = k->view_src ? k->view_src : k;
    return n_tokens > argus_k1_max_tokens() && storage->buffer && ggml_backend_buffer_is_host(storage->buffer);
}
