#pragma once

#include <torch/extension.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "zero_copy_pool.h"
#include "tier_codec.h"

// Saf C++ Sayfa Metadata Yapısı
struct Page {
    int page_id;
    int pool_slot;
    int page_size;
    float importance_score;
    float attention_sum;
    int last_step_accessed;
    std::string tier_name; // "active", "fp8", "int8", "int4", "int2", "one_bit", "jl"
    // Original dtype of key/value as pushed by the caller (fp16/fp32/bf16...).
    // Dequantized/resurrected tensors are cast back to this so mixed-dtype
    // callers stay internally consistent through the demote/resurrect cascade.
    at::ScalarType orig_dtype = at::kHalf;

    // PyTorch ATen Tensor görünümleri (Veri kopyalamasını engellemek için)
    at::Tensor key_tensor;
    at::Tensor value_tensor;

    // Sıkıştırılmış veri alanları (CUDA veya Pinned Host bellek pointer'ları)
    at::Tensor compressed_key;
    at::Tensor compressed_value;
    float key_scale;
    float value_scale;
    float key_min;
    float value_min;

    // Catch-all for pluggable Python-side metadata (eviction policy reference
    // bits/heat registers, outlier isolation indices, etc.) that don't have a
    // fixed C++ field. Keeps Page usable as a free-form dict from Python.
    std::unordered_map<std::string, pybind11::object> extra_fields;
};

class ArgusCppManager {
public:
    int max_active_pages_;
    int generation_step_;
    // Saf C++ Sayfa Havuzları
    std::vector<std::shared_ptr<Page>> active_pages_;
    std::unordered_map<std::string, std::vector<std::shared_ptr<Page>>> pages_by_tier_;

private:
    int page_size_;
    int device_id_;
    int page_counter_;
    
    // Tiers Spec limitleri
    std::unordered_map<std::string, int> tier_max_pages_;

    // Zero-Copy Host Memory Pool
    std::unique_ptr<ZeroCopyHostPool> host_pool_;

    // Asynchronous Prefetcher
    cudaStream_t prefetch_stream_;
    std::thread prefetch_worker_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    std::queue<std::shared_ptr<Page>> prefetch_queue_;
    bool stop_prefetch_worker_;
    // True from the moment the worker pops an item until it's fully stored
    // (or skipped) — lets wait_for_prefetch_idle() distinguish "queue empty"
    // from "queue empty AND nothing in-flight".
    bool prefetch_worker_busy_ = false;

    // Prefetch Cache: page_id -> (key_tensor, value_tensor)
    std::unordered_map<int, std::pair<at::Tensor, at::Tensor>> prefetch_cache_;
    int prefetch_hit_count_ = 0;

    void prefetch_worker_loop();

public:
    ArgusCppManager(int page_size, int max_active_pages, int device_id);
    ~ArgusCppManager();

    // Token ekleme ve attention işlemleri
    void push_new_tokens(torch::Tensor k, torch::Tensor v);
    torch::Tensor inplace_paged_attention(
        torch::Tensor q,
        float scale,
        c10::optional<torch::Tensor> sink_k = c10::nullopt,
        c10::optional<torch::Tensor> sink_v = c10::nullopt,
        c10::optional<torch::Tensor> anchor_k = c10::nullopt,
        c10::optional<torch::Tensor> anchor_v = c10::nullopt,
        c10::optional<torch::Tensor> k_buffer = c10::nullopt,
        c10::optional<torch::Tensor> v_buffer = c10::nullopt,
        float resurrection_threshold = 0.15f
    );

    // Bellek tahliye ve katman yönetimi (Eviction & Promotion/Demotion Cascade)
    void manage_memory_lifecycle();
    void demote_to_next_tier(std::shared_ptr<Page> page);
    void resurrect_page(std::shared_ptr<Page> page, std::string current_tier);

    // ── Codec-driven compression core ───────────────────────────────────────
    // Every tier transition in this class routes through these two methods.
    // They read the storage format from the registered TierCodec instead of
    // branching on the tier's name, which is what makes a plugin tier a
    // first-class citizen rather than an uncompressed-passthrough special case.

    // Compresses page->key_tensor/value_tensor per `codec`, parks the result in
    // pinned host memory, and clears the fp16 tensors. Writes the per-page
    // scale/min metadata the matching decompress needs.
    void compress_page(const std::shared_ptr<Page> &page,
                       const argus::TierCodec &codec);

    // Inverse of compress_page. Returns fresh tensors and never mutates the
    // page, so resurrect (which moves the page) and peek (which does not) can
    // share it. `stream` selects the CUDA stream; `allow_python_callbacks`
    // must be false when called off the main thread, since the JL projection
    // providers reach back into Python and the GIL is not held there.
    // `synchronize` is required when `stream` is not the caller's ambient
    // stream (the prefetch worker), because the trailing dtype cast is issued
    // on the ambient stream and would otherwise race the dequant kernel.
    // Returns undefined tensors when the codec cannot run — currently only a
    // Projection tier whose operator is neither cached nor buildable without
    // the GIL. Callers must check .defined().
    std::pair<at::Tensor, at::Tensor>
    decompress_page(const std::shared_ptr<Page> &page,
                    const argus::TierCodec &codec, cudaStream_t stream,
                    bool allow_python_callbacks, bool synchronize);

    // Codec registry access. Registering a codec for a tier name is how a
    // Python plugin gets a real native storage format; tiers with no
    // registered codec fall back to an uncompressed host spill.
    argus::CodecRegistry codec_registry_;
    const argus::TierCodec &codec_for(const std::string &tier_name) const {
        return codec_registry_.get(tier_name);
    }

    // Hot-path logging. Off by default — the previous unconditional std::cout
    // per attention call dominated the profile at long context. Enable with
    // ARGUS_VERBOSE=1 or set_verbose(True).
    bool verbose_ = false;
    void set_verbose(bool v) { verbose_ = v; }

    // Dequantizes a page's compressed data using the same native CUDA
    // kernels / JL matmul as resurrect_page, but returns the result instead
    // of mutating the page or moving it between active_pages_/pages_by_tier_.
    // This is the only correct way for Python to read back tier-compressed
    // data: the pluggable Python QuantizationBackend classes pack along a
    // different axis than these kernels do, so they silently misdecode
    // anything actually compressed here.
    std::pair<at::Tensor, at::Tensor> peek_decompress_page(std::shared_ptr<Page> page, std::string current_tier);

    // Prefetching interface
    void speculate_and_prefetch(const std::vector<int>& page_ids);

    // Python-visible introspection so speculate_and_prefetch's effect on the
    // live hot path (resurrect_page / inplace_paged_attention's prefetch-cache
    // checks) can actually be observed and tested, instead of relying on a
    // separate Python-side simulation that the C++ hot path never consults.
    int get_prefetch_hit_count() const { return prefetch_hit_count_; }
    std::vector<int> get_prefetched_page_ids() {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        std::vector<int> ids;
        ids.reserve(prefetch_cache_.size());
        for (const auto &kv : prefetch_cache_) ids.push_back(kv.first);
        return ids;
    }
    // Non-consuming peek — used by callers (e.g. get_all_keys_values's batched
    // decompress path) that just want to reuse an already-prefetched tensor
    // pair without going through resurrect_page's consuming fast path.
    pybind11::object get_prefetched_tensors(int page_id) {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        auto it = prefetch_cache_.find(page_id);
        if (it == prefetch_cache_.end()) return pybind11::none();
        prefetch_hit_count_++;
        return pybind11::make_tuple(it->second.first, it->second.second);
    }
    // Blocks until the background prefetch worker has drained its queue, so
    // callers (tests included) can synchronously observe the result of a
    // speculate_and_prefetch() call instead of racing the worker thread.
    void wait_for_prefetch_idle() {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_cv_.wait(lock, [this]() { return prefetch_queue_.empty() && !prefetch_worker_busy_; });
    }

    // Yardımcı metotlar
    int get_page_count() const;
    int get_active_page_count() const;
    int get_tier_page_count(const std::string& tier_name) const;

    // Single source of truth for page IDs — Python's split_page/merge_pages
    // must draw from this too, or their own counter collides with page IDs
    // already assigned here (both would otherwise start from 0).
    int next_page_id() { return page_counter_++; }

    // ZeroCopyHostPool access
    ZeroCopyHostPool* get_host_pool() { return host_pool_.get(); }

    void set_tier_max_pages(const std::string& tier_name, int max_pages) {
        tier_max_pages_[tier_name] = max_pages;
    }

    void add_tier_cpp(const std::string& tier_name) {
        if (pages_by_tier_.find(tier_name) == pages_by_tier_.end()) {
            pages_by_tier_[tier_name] = std::vector<std::shared_ptr<Page>>();
        }
    }
    void remove_tier_cpp(const std::string& tier_name) {
        pages_by_tier_.erase(tier_name);
    }

    std::vector<std::string> tier_pipeline_ = {"fp8", "int8", "int4", "int2", "one_bit", "jl"};
    void set_tier_pipeline_cpp(const std::vector<std::string>& pipeline) {
        tier_pipeline_ = pipeline;
    }

    // Keyed by sequence length (N for w_proj's [M,N] shape, N for
    // recon_operator's [N,M] shape) — variable-granularity micro-pages need a
    // differently-shaped JL matrix than full-size pages, since JL projects
    // along the sequence axis. Falls back to page_size_ when unspecified.
    std::unordered_map<int, at::Tensor> jl_w_proj_by_len_;
    std::unordered_map<int, at::Tensor> jl_recon_operator_by_len_;

    void set_jl_projection_matrix(torch::Tensor w_proj) {
        if (w_proj.defined() && w_proj.dim() >= 2) {
            jl_w_proj_by_len_[w_proj.size(-1)] = w_proj;
        }
    }
    void set_jl_recon_operator(torch::Tensor recon_operator) {
        if (recon_operator.defined() && recon_operator.dim() >= 2) {
            jl_recon_operator_by_len_[recon_operator.size(-2)] = recon_operator;
        }
    }

    // Returns the cached matrix for this seq_len, invoking the lazy provider
    // (with `sample` for device/dtype context) if not yet built.
    at::Tensor get_jl_w_proj(int seq_len, const at::Tensor &sample) {
        at::Tensor &cached = jl_w_proj_by_len_[seq_len];
        if ((!cached.defined() || cached.numel() == 0) && !jl_projection_provider_.is_none()) {
            cached = jl_projection_provider_(sample).cast<at::Tensor>();
        }
        TORCH_CHECK(cached.defined() && cached.numel() > 0,
                    "[ARGUS C++] JL projection matrix not set for seq_len=", seq_len,
                    ". Call set_jl_projection_matrix() first.");
        return cached;
    }
    at::Tensor get_jl_recon_operator(int seq_len, const at::Tensor &sample, int sample_seq_len) {
        at::Tensor &cached = jl_recon_operator_by_len_[seq_len];
        if ((!cached.defined() || cached.numel() == 0) && !jl_recon_provider_.is_none()) {
            cached = jl_recon_provider_(sample, sample_seq_len).cast<at::Tensor>();
        }
        TORCH_CHECK(cached.defined() && cached.numel() > 0,
                    "[ARGUS C++] JL reconstruction operator not set for seq_len=", seq_len);
        return cached;
    }

    // ── Pluggable policy hooks ──────────────────────────────────────────────
    // These let the Python-side management layer (eviction policies, JL
    // projection lazy-init) stay authoritative over *decisions* while C++
    // keeps owning the *mechanics* (memory pool, quantization kernels,
    // attention). All are invoked synchronously from calls that originate in
    // Python (push_new_tokens / inplace_paged_attention), so the GIL is
    // already held — never call these from prefetch_worker_loop (background
    // thread, no GIL).
    pybind11::object active_pool_victim_selector_ = pybind11::none();
    pybind11::object on_page_access_callback_ = pybind11::none();
    pybind11::object jl_projection_provider_ = pybind11::none();
    pybind11::object jl_recon_provider_ = pybind11::none();

    void set_active_pool_victim_selector(pybind11::object fn) { active_pool_victim_selector_ = fn; }
    void set_on_page_access_callback(pybind11::object fn) { on_page_access_callback_ = fn; }
    void set_jl_projection_provider(pybind11::object fn) { jl_projection_provider_ = fn; }
    void set_jl_recon_provider(pybind11::object fn) { jl_recon_provider_ = fn; }

    // Mirrors the pre-C++-port "Eager Bypass": when there's no memory
    // pressure and nothing compressed yet, inplace_paged_attention skips the
    // QoS/importance/resurrection bookkeeping entirely and just returns SDPA
    // output — full native speed on the common case. force_qos overrides it.
    bool force_qos_ = false;
    void set_force_qos(bool v) { force_qos_ = v; }

    // Picks the active-pool victim: defers to the pluggable eviction policy
    // when registered, otherwise falls back to the lowest-importance scan.
    std::shared_ptr<Page> select_active_victim();
};

