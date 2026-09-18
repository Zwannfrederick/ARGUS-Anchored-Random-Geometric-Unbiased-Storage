// Standalone CUDA/storage contract. The scheduler hook is captured here; native
// llama tests separately exercise the actual GGML CUDA backend dispatch.
#include "ggml_cuda_attention.h"
#include "ggml_kv_policy.h"
#include "ggml_profile.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

static bool short_write = false;
static bool corrupt_read = false;
static size_t io_reads = 0, io_writes = 0;
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" ssize_t __wrap_pwrite(int fd, const void * data, size_t count, off_t offset) {
    ++io_writes;
    return __real_pwrite(fd, data, short_write ? count / 2 : count, offset);
}
extern "C" ssize_t __wrap_pread(int fd, void * data, size_t count, off_t offset) {
    ++io_reads;
    const auto result = __real_pread(fd, data, count, offset);
    if (corrupt_read && result > 0) { static_cast<unsigned char *>(data)[0] ^= 1; }
    return result;
}

static ggml_cuda_custom_compute_t compute = nullptr;
extern "C" void ggml_backend_cuda_register_custom_op(ggml_custom_op_t,
        ggml_cuda_custom_compute_t callback, ggml_backend_buffer_type_t) { compute = callback; }

static void require(bool value) { if (!value) { throw std::runtime_error("CUDA mechanism assertion failed"); } }

template<class F> void refuses(F f) {
    bool refused = false;
    try { f(); } catch (const std::exception &) { refused = true; }
    require(refused);
}

static void check_gpu_control() {
    setenv("ARGUS_KV_GPU_CONTROL", "1", 1);
    setenv("ARGUS_KV_GPU_BYTES", "16384", 1);
    auto * ctx = ggml_init({1 << 20, nullptr, true});
    auto * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, argus_ggml_disk_buffer_type());
    require(buffer);
    const auto reads = io_reads, writes = io_writes;
    std::vector<float> payload(2048, 1.25f), copied(2048);
    ggml_backend_tensor_set(tensor, payload.data(), 0, 8192);
    require(argus_disk_page_descriptor(tensor, 0).placement == ArgusTier::gpu);
    const auto revision = argus_disk_page_revision(tensor, 0);
    refuses([&] { argus_disk_move_page(tensor, 0, ArgusTier::disk, revision); });
    setenv("ARGUS_KV_GPU_BYTES", "8192", 1);
    const float updated = 7.0f;
    refuses([&] { ggml_backend_tensor_set(tensor, &updated, 0, sizeof(updated)); });
    require(argus_disk_page_revision(tensor, 0).content_revision == revision.content_revision);
    ggml_backend_tensor_get(tensor, copied.data(), 0, 8192);
    require(copied == payload);
    setenv("ARGUS_KV_GPU_BYTES", "16384", 1);
    ggml_backend_tensor_set(tensor, &updated, 0, sizeof(updated));
    payload[0] = updated;
    ggml_backend_tensor_get(tensor, copied.data(), 0, 8192);
    require(copied == payload && io_reads == reads && io_writes == writes);
    ggml_backend_buffer_clear(buffer, 0);
    require(argus_tier_usage().gpu == 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    unsetenv("ARGUS_KV_GPU_CONTROL");
}

static void check_policy() {
    setenv("ARGUS_KV_GPU_BYTES", "4096", 1);
    setenv("ARGUS_KV_PINNED_BYTES", "4096", 1);
    setenv("ARGUS_KV_RAM_BYTES", "4096", 1);
    auto * ctx = ggml_init({1 << 20, nullptr, true});
    auto * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, argus_ggml_disk_buffer_type());
    require(buffer);
    std::vector<unsigned char> payload(16384, 73), copied(payload.size());
    ggml_backend_tensor_set(tensor, payload.data(), 0, payload.size());
    const auto revision = argus_disk_revision(tensor);
    unsetenv("ARGUS_KV_POLICY");
    require(!argus_kv_policy_enabled());
    setenv("ARGUS_KV_POLICY", "typo", 1);
    refuses([] { argus_kv_policy_enabled(); });
    setenv("ARGUS_KV_POLICY", "off", 1);
    argus_kv_policy_observe(tensor);
    require(argus_kv_policy_stats().promotions == 0);
    setenv("ARGUS_KV_POLICY", "on", 1);
    argus_kv_policy_observe(tensor); // Written but never read: no promotion.
    ggml_backend_tensor_get(tensor, copied.data(), 0, copied.size());
    argus_kv_policy_observe(tensor); // One-shot scan: no promotion.
    require(argus_kv_policy_stats().promotions == 0);
    ggml_backend_tensor_get(tensor, copied.data(), 0, copied.size());
    argus_kv_policy_observe(tensor);
    require(argus_disk_page_descriptor(tensor, 0).placement == ArgusTier::gpu);
    require(argus_disk_page_descriptor(tensor, 4096).placement == ArgusTier::pinned);
    require(argus_disk_page_descriptor(tensor, 8192).placement == ArgusTier::ram);
    require(argus_disk_page_descriptor(tensor, 12288).placement == ArgusTier::disk);
    for (int i = 0; i < 3; ++i) { ggml_backend_tensor_get(tensor, copied.data(), 12288, 4096); }
    argus_kv_policy_observe(tensor); // Hotter page replaces the colder GPU page.
    require(argus_disk_page_descriptor(tensor, 12288).placement == ArgusTier::gpu);
    require(argus_disk_page_descriptor(tensor, 0).placement == ArgusTier::disk);
    require(argus_disk_revision(tensor) == revision);
    const auto stats = argus_kv_policy_stats();
    corrupt_read = true;
    argus_kv_policy_prepare(4096, 0);
    corrupt_read = false;
    require(argus_kv_policy_stats().rejected == stats.rejected + 1);
    require(argus_disk_page_descriptor(tensor, 12288).placement == ArgusTier::gpu);
    argus_kv_policy_prepare(4096, 4096);
    require(argus_tier_usage().gpu == 0 && argus_tier_usage().pinned == 0);
    require(argus_kv_policy_stats().demotions >= 3);
    ggml_backend_tensor_get(tensor, copied.data(), 0, copied.size());
    require(copied == payload);
    require(argus_disk_page_descriptor(tensor, 12288).codec == GGML_TYPE_F32);
    const auto stale = argus_disk_page_revision(tensor, 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    refuses([&] { argus_disk_move_page(stale, ArgusTier::gpu); });
    require(argus_tier_usage().gpu == 0 && argus_tier_usage().pinned == 0 && argus_tier_usage().ram == 0);
    setenv("ARGUS_KV_POLICY", "off", 1);
    std::puts("policy: off/on, admission, tier budgets, eviction, failure counters and teardown passed");
}

// Qwen2.5-0.5B geometry: 14 Q / 2 KV heads, D=64, a 64-token ubatch over 600 cells of
// which the first 500 are written (the rest are unwritten, logical-zero pages).
// Every resident kernel (direct, batched, lane-per-cell) must equal staged bit for bit.
static void check_qwen_geometry(std::mt19937 & rng) {
    using namespace argus_profile;
    std::normal_distribution<float> normal;
    std::uniform_real_distribution<float> uniform;
    setenv("ARGUS_KV_GPU_BYTES", "4194304", 1);
    const int heads = 14, tokens = 64, cells = 600, written = 500;
    auto * kv_ctx = ggml_init({1 << 20, nullptr, true});
    auto * k = ggml_new_tensor_3d(kv_ctx, GGML_TYPE_F16, 64, 2, cells);
    auto * v = ggml_new_tensor_3d(kv_ctx, GGML_TYPE_F16, 64, 2, cells);
    setenv("ARGUS_KV_GPU_CONTROL", "1", 1);
    auto * store = ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx, argus_ggml_disk_buffer_type());
    unsetenv("ARGUS_KV_GPU_CONTROL");
    require(store);
    auto * ctx = ggml_init({1 << 20, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, heads, tokens);
    auto * half_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cells, tokens);
    auto * float_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cells, tokens);
    ArgusTierBuffer q_gpu(ArgusTier::gpu, ggml_nbytes(q)), half_gpu(ArgusTier::gpu, ggml_nbytes(half_mask));
    ArgusTierBuffer float_gpu(ArgusTier::gpu, ggml_nbytes(float_mask)), out_gpu(ArgusTier::gpu, ggml_nbytes(q));
    q->data = q_gpu.data(); half_mask->data = half_gpu.data(); float_mask->data = float_gpu.data();
    std::vector<float> query(64 * heads * tokens), bias(cells * tokens), expected(query.size()), actual(query.size());
    std::vector<ggml_fp16_t> kv(64 * 2 * written), half_bias(bias.size());
    const auto run = [&](const char * label, bool float_bias) {
        q_gpu.write(query.data(), query.size() * sizeof(float));
        for (size_t i = 0; i < bias.size(); ++i) { half_bias[i] = ggml_fp32_to_fp16(bias[i]); }
        half_gpu.write(half_bias.data(), half_bias.size() * 2);
        float_gpu.write(bias.data(), bias.size() * sizeof(float));
        auto * out = argus_ggml_cuda_attention(ctx, q, k, v, float_bias ? float_mask : half_mask, 0.125f);
        out->data = out_gpu.data();
        setenv("ARGUS_KV_ATTENTION_PATH", "staged", 1);
        compute(out, 0, nullptr);
        out_gpu.read(expected.data(), 0, expected.size() * sizeof(float));
        for (const auto * path : {"direct", "batched", "cells", "cells-mlp"}) {
            setenv("ARGUS_KV_ATTENTION_PATH", path, 1);
            const auto accepted = resident_accepted[prefill].load(), lanes = resident_cell_kernel[prefill].load();
            compute(out, 0, nullptr);
            out_gpu.read(actual.data(), 0, actual.size() * sizeof(float));
            for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i] && !(std::isnan(actual[i]) && std::isnan(expected[i]))) {
                std::fprintf(stderr, "qwen %s path=%s index=%zu actual=%a expected=%a\n", label, path, i, actual[i], expected[i]);
                require(false);
            }
            require(resident_accepted[prefill] == accepted + 1);
            require(resident_cell_kernel[prefill] == lanes + (std::strncmp(path, "cells", 5) == 0));
        }
    };
    const auto fill_kv = [&](bool constant_keys) {
        for (auto & x : kv) { x = ggml_fp32_to_fp16(normal(rng)); }
        if (constant_keys) { for (size_t i = 0; i < kv.size(); ++i) { kv[i] = kv[i % 128]; } } // equal scores (ties)
        ggml_backend_tensor_set(k, kv.data(), 0, kv.size() * 2);
        for (auto & x : kv) { x = ggml_fp32_to_fp16(normal(rng)); }
        ggml_backend_tensor_set(v, kv.data(), 0, kv.size() * 2);
    };
    fill_kv(false);
    for (auto & x : query) { x = normal(rng); }
    for (int t = 0; t < tokens; ++t) for (int c = 0; c < cells; ++c) { bias[t * cells + c] = c <= 436 + t ? 0.0f : -INFINITY; }
    run("causal", false);
    // Scattered masks, a fully masked 32-cell tile, fully masked rows, live unwritten cells.
    for (int t = 0; t < tokens; ++t) for (int c = 0; c < cells; ++c) {
        const bool masked = (c >= 64 && c < 96) || t % 17 == 3 || uniform(rng) < 0.3f;
        bias[t * cells + c] = masked ? -INFINITY : 0.0f;
    }
    run("scattered", false);
    for (int t = 0; t < tokens; ++t) for (int c = 0; c < cells; ++c) {
        if (std::isfinite(bias[t * cells + c])) { bias[t * cells + c] = 4.0f * normal(rng); }
    }
    run("float bias", true);
    // Wide score range: many alpha/beta underflow to zero.
    for (auto & x : query) { x *= 24.0f; }
    run("wide scores", true);
    // Signed-zero and tied scores: zero queries (+0 and -0) and identical key rows.
    fill_kv(true);
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) for (int i = 0; i < 64; ++i) {
        float & x = query[(t * heads + h) * 64 + i];
        x = h % 3 == 0 ? 0.0f : h % 3 == 1 ? -0.0f : normal(rng);
    }
    for (int t = 0; t < tokens; ++t) for (int c = 0; c < cells; ++c) { bias[t * cells + c] = c <= 500 + t ? 0.0f : -INFINITY; }
    run("zeros and ties", false);
    unsetenv("ARGUS_KV_ATTENTION_PATH");
    ggml_backend_buffer_free(store);
    ggml_free(kv_ctx);
    ggml_free(ctx);
    setenv("ARGUS_KV_GPU_BYTES", "1048576", 1);
    std::puts("qwen geometry: staged == direct == batched == cells across masks, zeros, ties and wide scores");
}

static std::vector<ArgusDiskPageDescriptor> descriptors(const ggml_tensor * k, const ggml_tensor * v) {
    std::vector<ArgusDiskPageDescriptor> all;
    for (const auto * tensor : {k, v}) {
        for (size_t offset = 0; offset < ggml_nbytes(tensor); offset += 4096) { all.push_back(argus_disk_page_descriptor(tensor, offset)); }
    }
    return all;
}
static void move_pages(const ggml_tensor * tensor, ArgusTier tier) {
    for (size_t offset = 0; offset + 4096 <= ggml_nbytes(tensor); offset += 4096) {
        argus_disk_move_page(tensor, offset, tier, argus_disk_page_revision(tensor, offset));
    }
}

// GPU pages are borrowed and written non-GPU pages are staged for one invocation only:
// exact against the staged path, with placement, revisions and access history unchanged.
static void check_mixed_residency(ggml_tensor * k, ggml_tensor * v, ggml_tensor * out,
                                  const ArgusTierBuffer & output, size_t elements) {
    using namespace argus_profile;
    std::vector<float> expected(elements), actual(elements);
    const auto run = [&](const char * label, uint64_t cold_pages) {
        setenv("ARGUS_KV_ATTENTION_PATH", "staged", 1);
        const auto before = descriptors(k, v);
        compute(out, 0, nullptr);
        const auto staged = descriptors(k, v);
        output.read(expected.data(), 0, elements * sizeof(float));
        for (const auto * path : {"direct", "batched", "cells", "cells-mlp"}) {
            setenv("ARGUS_KV_ATTENTION_PATH", path, 1);
            const auto accepted = resident_accepted[prefill].load(), cold = resident_cold_pages[prefill].load();
            const auto start = descriptors(k, v);
            compute(out, 0, nullptr);
            const auto end = descriptors(k, v);
            output.read(actual.data(), 0, elements * sizeof(float));
            if (actual != expected || resident_cold_pages[prefill] - cold != cold_pages) {
                std::fprintf(stderr, "mixed %s path=%s cold=%llu expected_cold=%llu\n", label, path,
                             (unsigned long long) (resident_cold_pages[prefill] - cold), (unsigned long long) cold_pages);
            }
            require(actual == expected && resident_accepted[prefill] == accepted + 1);
            require(resident_cold_pages[prefill] - cold == cold_pages);
            for (size_t i = 0; i < end.size(); ++i) {
                require(end[i].placement == start[i].placement && end[i].placement_revision == start[i].placement_revision &&
                        end[i].revision.content_revision == start[i].revision.content_revision);
                require(end[i].access_count - start[i].access_count == staged[i].access_count - before[i].access_count);
            }
        }
    };
    // 97 cells never fill whole pages: each tensor has a partial tail page, which
    // cannot migrate, so it stays on disk and is always cold.
    require(ggml_nbytes(k) % 4096 && ggml_nbytes(v) % 4096);
    move_pages(k, ArgusTier::gpu);
    move_pages(v, ArgusTier::gpu);
    run("tails", 2);
    argus_disk_move_page(k, 8192, ArgusTier::disk, argus_disk_page_revision(k, 8192));
    run("cold key", 3);
    argus_disk_move_page(k, 8192, ArgusTier::gpu, argus_disk_page_revision(k, 8192));
    argus_disk_move_page(v, 4096, ArgusTier::pinned, argus_disk_page_revision(v, 4096));
    run("cold value pinned", 3);
    argus_disk_move_page(v, 0, ArgusTier::ram, argus_disk_page_revision(v, 0));
    run("gpu pinned ram disk", 4);
    // A rewrite drops residency: the write frontier is cold.
    std::vector<unsigned char> page(4096);
    const size_t frontier = (ggml_nbytes(k) / 4096 - 1) * 4096;
    ggml_backend_tensor_get(k, page.data(), frontier, 4096);
    ggml_backend_tensor_set(k, page.data(), frontier, 4096);
    run("write frontier", 5);

    move_pages(k, ArgusTier::disk);
    move_pages(v, ArgusTier::disk);
    const size_t cold = (ggml_nbytes(k) + 4095) / 4096 + (ggml_nbytes(v) + 4095) / 4096;
    run("all cold", cold);

    // Batched scratch at the exact limit: host copy + device charge + resident table,
    // on top of the live pointer table. One page less declines to the staged path.
    setenv("ARGUS_KV_ATTENTION_PATH", "staged", 1);
    compute(out, 0, nullptr);
    output.read(expected.data(), 0, elements * sizeof(float));
    setenv("ARGUS_KV_ATTENTION_PATH", "batched", 1);
    const size_t live = argus_disk_staging_limit() - argus_disk_staging_free();
    const size_t exact = live + 4096 + 2 * cold * 4096 + 4096;
    const std::string saved = std::getenv("ARGUS_KV_STAGING_BYTES");
    for (size_t limit : {exact, exact - 4096}) {
        setenv("ARGUS_KV_STAGING_BYTES", std::to_string(limit).c_str(), 1);
        const auto declined = resident_rejected[prefill][reject_cold_budget].load(), accepted = resident_accepted[prefill].load();
        compute(out, 0, nullptr);
        output.read(actual.data(), 0, elements * sizeof(float));
        require(actual == expected);
        require(limit == exact ? resident_accepted[prefill] == accepted + 1 && resident_rejected[prefill][reject_cold_budget] == declined
                               : resident_accepted[prefill] == accepted && resident_rejected[prefill][reject_cold_budget] == declined + 1);
    }
    setenv("ARGUS_KV_STAGING_BYTES", saved.c_str(), 1);

    // A failed cold read changes nothing and queues nothing; the next read succeeds.
    const auto before = descriptors(k, v);
    const auto usage = argus_tier_usage();
    const auto free_staging = argus_disk_staging_free();
    corrupt_read = true;
    refuses([&] { compute(out, 0, nullptr); });
    corrupt_read = false;
    const auto after = descriptors(k, v);
    for (size_t i = 0; i < after.size(); ++i) {
        require(after[i].placement == before[i].placement && after[i].access_count == before[i].access_count);
    }
    require(argus_tier_usage().gpu == usage.gpu && argus_disk_staging_free() == free_staging);
    compute(out, 0, nullptr);
    output.read(actual.data(), 0, elements * sizeof(float));
    require(actual == expected);

    // A concurrent writer waits for the borrowed and staged pages to be released.
    struct Race {
        std::vector<char> host;
        std::unique_ptr<ArgusTierBuffer> device;
        std::promise<void> entered;
        std::future<void> attempting;
        std::future<void> * writer;
    } race;
    std::promise<void> attempted;
    race.attempting = attempted.get_future();
    auto started = race.entered.get_future();
    auto writer = std::async(std::launch::async, [&] {
        started.wait(); attempted.set_value();
        ggml_backend_tensor_set(k, page.data(), 0, 2);
    });
    race.writer = &writer;
    static constexpr ArgusColdStaging staging{
        [](size_t pages, void * context) -> void * {
            auto & r = *static_cast<Race *>(context);
            r.host.resize(pages * 4096);
            r.device = std::make_unique<ArgusTierBuffer>(ArgusTier::gpu, pages * 4096);
            return r.host.data();
        },
        [](size_t pages, void * context) -> const char * {
            auto & r = *static_cast<Race *>(context);
            r.device->write(r.host.data(), pages * 4096);
            return static_cast<const char *>(r.device->data());
        }};
    std::vector<const void *> pointers(64);
    require(argus_disk_read_resident(k, v, pointers.data(), pointers.size(), [](size_t, size_t, void * context) {
        auto & r = *static_cast<Race *>(context);
        r.entered.set_value(); r.attempting.wait();
        require(r.writer->wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    }, &race, &staging));
    writer.get();
    unsetenv("ARGUS_KV_ATTENTION_PATH");
    std::puts("mixed residency: exact parity, no placement change, scratch limit, failure and writer passed");
}

int main(int argc, char ** argv) try {
    require(argc == 2 || argc == 3);
    setenv("ARGUS_KV_POLICY", "off", 1);
    setenv("ARGUS_KV_DIR", argv[1], 1);
    setenv("ARGUS_KV_MAX_BYTES", "1048576", 1);
    setenv("ARGUS_KV_RESIDENT_BYTES", "1048576", 1);
    setenv("ARGUS_KV_STAGING_BYTES", "1048576", 1);
    setenv("ARGUS_KV_GPU_BYTES", "1048576", 1);
    setenv("ARGUS_KV_PINNED_BYTES", "1048576", 1);
    setenv("ARGUS_KV_RAM_BYTES", "1048576", 1);
    const int d = argc == 3 ? std::stoi(argv[2]) : 64;
    require(d == 48 || d == 64 || d == 256);
    const int heads = d == 64 ? 6 : 4, kv_heads = 2, cells = 97, tokens = 3;
    auto * ctx = ggml_init({1 << 20, nullptr, true});
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, kv_heads, cells);
    auto * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, kv_heads, cells);
    auto * q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 256, 32);
    auto * q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 32);
    auto * store = ggml_backend_alloc_ctx_tensors_from_buft(ctx, argus_ggml_disk_buffer_type());
    require(store);
    // Placement preserves encoded bytes even when the attention kernel lacks that codec.
    for (auto * encoded : {q8, q4}) {
        std::vector<unsigned char> payload(ggml_nbytes(encoded)), copied(payload.size());
        for (size_t i = 0; i < payload.size(); ++i) { payload[i] = i % 251; }
        ggml_backend_tensor_set(encoded, payload.data(), 0, payload.size());
        for (auto tier : {ArgusTier::gpu, ArgusTier::pinned, ArgusTier::ram, ArgusTier::disk}) {
            const auto before_move = argus_disk_page_descriptor(encoded, 0);
            argus_disk_move_page(encoded, 0, tier, argus_disk_page_revision(encoded, 0));
            const auto after_move = argus_disk_page_descriptor(encoded, 0);
            require(after_move.codec == encoded->type && after_move.placement == tier && after_move.written);
            require(after_move.placement_revision == before_move.placement_revision + 1);
            require(after_move.revision.content_revision == before_move.revision.content_revision);
            require(after_move.access_count == before_move.access_count &&
                    after_move.last_access_step == before_move.last_access_step);
            ggml_backend_tensor_get(encoded, copied.data(), 0, copied.size());
            require(copied == payload);
            require(argus_disk_page_descriptor(encoded, 0).access_count == before_move.access_count + 1);
        }
    }
    std::mt19937 rng(42);
    std::normal_distribution<float> normal;
    std::vector<ggml_fp16_t> keys(d * kv_heads * cells), values(keys.size());
    for (auto & x : keys) { x = ggml_fp32_to_fp16(normal(rng)); }
    for (auto & x : values) { x = ggml_fp32_to_fp16(normal(rng)); }
    ggml_backend_tensor_set(k, keys.data(), 0, keys.size() * 2);
    ggml_backend_tensor_set(v, values.data(), 0, values.size() * 2);
    const auto initial = argus_disk_page_revision(k, 0);
    const auto content = argus_disk_revision(k);
    // Another page's content write must not invalidate this migration token.
    ggml_backend_tensor_set(k, reinterpret_cast<const char *>(keys.data()) + 4096, 4096, 4096);
    refuses([&] { argus_disk_move_page(k, 4096, ArgusTier::ram, initial); });
    refuses([&] { argus_disk_move_page(v, 0, ArgusTier::ram, initial); });
    const auto after_write = argus_disk_revision(k);
    require(!(content == after_write));
    // Same FP16 bytes through every placement. Policy is explicit in this test.
    for (auto tier : {ArgusTier::ram, ArgusTier::pinned, ArgusTier::gpu, ArgusTier::disk}) {
        std::vector<ggml_fp16_t> prefetched_keys(keys.size()), prefetched_values(values.size());
        ArgusDiskPrefetch prefetch;
        prefetch.submit(k, v, 0, cells, prefetched_keys.data(), prefetched_values.data());
        argus_disk_move_page(k, 0, tier, initial);
        prefetch.take();
        require(prefetched_keys == keys && prefetched_values == values);
        std::vector<ggml_fp16_t> roundtrip(keys.size());
        ggml_backend_tensor_get(k, roundtrip.data(), 0, roundtrip.size() * 2);
        require(roundtrip == keys);
    }
    require(after_write == argus_disk_revision(k));
    refuses([&] { argus_disk_move_page(k, 1, ArgusTier::gpu, argus_disk_page_revision(k, 0)); });
    argus_disk_move_page(k, 0, ArgusTier::gpu, argus_disk_page_revision(k, 0));
    argus_disk_move_page(k, 4096, ArgusTier::pinned, argus_disk_page_revision(k, 4096));
    argus_disk_move_page(v, 0, ArgusTier::ram, argus_disk_page_revision(v, 0));
    const auto before = argus_disk_revision(k);
    setenv("ARGUS_KV_GPU_BYTES", "4096", 1);
    refuses([&] { argus_disk_move_page(k, 8192, ArgusTier::gpu, argus_disk_page_revision(k, 8192)); });
    require(before == argus_disk_revision(k));
    require(argus_tier_usage().gpu == 4096);
    setenv("ARGUS_KV_GPU_BYTES", "1048576", 1);
    short_write = true;
    refuses([&] { ggml_backend_tensor_set(k, keys.data(), 0, 4096); });
    short_write = false;
    require(before == argus_disk_revision(k));
    require(argus_tier_usage().gpu == 4096);
    std::vector<ggml_fp16_t> preserved(keys.size());
    ggml_backend_tensor_get(k, preserved.data(), 0, preserved.size() * 2);
    require(preserved == keys);
    corrupt_read = true;
    refuses([&] { argus_disk_move_page(k, 0, ArgusTier::disk, initial); });
    corrupt_read = false;
    require(before == argus_disk_revision(k) && argus_tier_usage().gpu == 4096);
    {
        auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, tokens);
        auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cells, tokens);
        std::vector<float> query(d * heads * tokens);
        for (auto & x : query) { x = normal(rng); }
        std::vector<ggml_fp16_t> bias(cells * tokens);
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < cells; ++c) {
                // Include an entirely masked row and a partial final tile.
                bias[t * cells + c] = ggml_fp32_to_fp16(t && c < 90 + t ? 0.0f : -INFINITY);
            }
        }
        ArgusTierBuffer q_gpu(ArgusTier::gpu, query.size() * sizeof(float));
        ArgusTierBuffer mask_gpu(ArgusTier::gpu, bias.size() * 2);
        ArgusTierBuffer output_gpu(ArgusTier::gpu, query.size() * sizeof(float));
        q_gpu.write(query.data(), query.size() * sizeof(float)); mask_gpu.write(bias.data(), bias.size() * 2);
        q->data = q_gpu.data(); mask->data = mask_gpu.data();
        auto * out = argus_ggml_cuda_attention(ctx, q, k, v, mask, 1.0f / std::sqrt(float(d)));
        out->data = output_gpu.data();
        require(compute && argus_ggml_is_cuda_attention(out));
        std::vector<float> first(query.size()), second(query.size());
        const auto before_compute = argus_disk_page_descriptor(k, 0);
        using argus_profile::resident_rejected;
        using argus_profile::prefill;
        const auto mixed_accepts = argus_profile::resident_accepted[prefill].load();
        const auto censused = argus_profile::census[prefill].invocations.load();
        compute(out, 0, nullptr);
        // GPU, pinned, RAM and disk pages together: the resident path stages the cold ones.
        require(argus_profile::resident_accepted[prefill] == mixed_accepts + 1);
        if (argus_profile::enabled()) {
            const auto & c = argus_profile::census[prefill];
            require(c.invocations == censused + 1 && c.cold_key_invocations > 0 && c.cold_value_invocations > 0);
            require(c.pages[argus_profile::page_gpu] > 0 && c.pages[argus_profile::page_pinned] > 0 &&
                    c.pages[argus_profile::page_ram] > 0 && c.pages[argus_profile::page_disk] > 0);
        }
        const auto after_compute = argus_disk_page_descriptor(k, 0);
        require(after_compute.placement == ArgusTier::gpu && after_compute.codec == GGML_TYPE_F16);
        require(after_compute.access_count > before_compute.access_count &&
                after_compute.last_access_step > before_compute.last_access_step);
        output_gpu.read(first.data(), 0, first.size() * sizeof(float));
        // The staged checks below keep exercising the tiled path and its pinned budget.
        setenv("ARGUS_KV_ATTENTION_PATH", "staged", 1);
        setenv("ARGUS_KV_NO_OVERLAP", "1", 1);
        compute(out, 0, nullptr);
        output_gpu.read(second.data(), 0, second.size() * sizeof(float));
        require(first == second); // Mixed resident == staged, exactly.
        double error = 0;
        for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
            std::vector<double> scores(cells);
            double maximum = -INFINITY, sum = 0;
            for (int c = 0; c < cells; ++c) {
                double dot = 0;
                for (int i = 0; i < d; ++i) { dot += query[(t * heads + h) * d + i] * double(ggml_fp16_to_fp32(keys[(c * kv_heads + h / (heads / kv_heads)) * d + i])); }
                scores[c] = dot / std::sqrt(double(d)) + ggml_fp16_to_fp32(bias[t * cells + c]);
                maximum = std::max(maximum, scores[c]);
            }
            if (t) { for (auto score : scores) { sum += std::exp(score - maximum); } }
            for (int i = 0; i < d; ++i) {
                double expected = 0;
                if (t) { for (int c = 0; c < cells; ++c) { expected += std::exp(scores[c] - maximum) / sum * ggml_fp16_to_fp32(values[(c * kv_heads + h / (heads / kv_heads)) * d + i]); } }
                const float actual = first[(t * heads + h) * d + i];
                require(std::isfinite(actual));
                error = std::max(error, std::abs(actual - expected));
            }
        }
        require(error < 1e-4);
        const auto usage = argus_tier_usage();
        require(usage.peak_gpu <= 1048576 && usage.peak_pinned <= 1048576 && usage.peak_ram <= 1048576);
        setenv("ARGUS_KV_PINNED_BYTES", "4096", 1);
        refuses([&] { compute(out, 0, nullptr); });
        require(argus_tier_usage().gpu == usage.gpu && argus_tier_usage().pinned == usage.pinned);
        setenv("ARGUS_KV_PINNED_BYTES", "1048576", 1);
        // A successful write invalidates its resident copy; unrelated pages survive.
        ArgusDiskPrefetch pending;
        std::vector<ggml_fp16_t> pending_keys(keys.size()), pending_values(values.size());
        pending.submit(k, v, 0, cells, pending_keys.data(), pending_values.data());
        ggml_backend_tensor_set(k, keys.data(), 0, 4096);
        refuses([&] { pending.take(); });
        refuses([&] { argus_disk_move_page(k, 0, ArgusTier::gpu, initial); });
        compute(out, 0, nullptr);
        output_gpu.read(second.data(), 0, second.size() * sizeof(float));
        require(first == second);
        std::printf("cuda_attention_calls=%zu max_error=%g overlap_parity=true migration_failure_preserved=true\n",
                    argus_tier_usage().attention_calls, error);
        check_mixed_residency(k, v, out, output_gpu, first.size());

        // Fully resident, partial physical pages, unaligned views, F16/F32 masks.
        setenv("ARGUS_KV_GPU_CONTROL", "1", 1);
        auto * resident_ctx = ggml_init({1 << 20, nullptr, true});
        auto * rk = ggml_new_tensor_3d(resident_ctx, GGML_TYPE_F16, d, kv_heads, cells);
        auto * rv = ggml_new_tensor_3d(resident_ctx, GGML_TYPE_F16, d, kv_heads, cells);
        auto * resident_store = ggml_backend_alloc_ctx_tensors_from_buft(resident_ctx, argus_ggml_disk_buffer_type());
        require(resident_store);
        unsetenv("ARGUS_KV_GPU_CONTROL");
        ggml_backend_tensor_set(rk, keys.data(), 0, keys.size() * 2);
        ggml_backend_tensor_set(rv, values.data(), 0, values.size() * 2);
        auto * shifted_k = ggml_view_3d(resident_ctx, rk, d, kv_heads, cells - 1, rk->nb[1], rk->nb[2], 2);
        auto * shifted_v = ggml_view_3d(resident_ctx, rv, d, kv_heads, cells - 1, rv->nb[1], rv->nb[2], 2);
        auto * float_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cells, tokens);
        std::vector<float> float_bias(bias.size());
        for (size_t i = 0; i < bias.size(); ++i) {
            float_bias[i] = ggml_fp16_to_fp32(bias[i]);
            if (std::isfinite(float_bias[i])) { float_bias[i] += float(i % 7) * 0.125f; }
        }
        ArgusTierBuffer float_mask_gpu(ArgusTier::gpu, float_bias.size() * sizeof(float));
        float_mask_gpu.write(float_bias.data(), float_bias.size() * sizeof(float));
        float_mask->data = float_mask_gpu.data();
        for (bool shifted : {false, true}) for (auto * test_mask : {mask, float_mask}) {
            auto * test_k = shifted ? shifted_k : rk;
            auto * test_v = shifted ? shifted_v : rv;
            auto * result = argus_ggml_cuda_attention(ctx, q, test_k, test_v, test_mask, 1.0f / std::sqrt(float(d)));
            result->data = output_gpu.data();
            setenv("ARGUS_KV_ATTENTION_PATH", "staged", 1);
            const auto before_staged = argus_disk_page_descriptor(test_k, 0).access_count;
            const auto forced = resident_rejected[prefill][argus_profile::reject_forced_staged].load();
            compute(result, 0, nullptr);
            require(resident_rejected[prefill][argus_profile::reject_forced_staged] == forced + 1);
            output_gpu.read(second.data(), 0, second.size() * sizeof(float));
            const auto after_staged = argus_disk_page_descriptor(test_k, 0).access_count;
            for (const auto * path : {"direct", "batched", "cells", "cells-mlp"}) {
                setenv("ARGUS_KV_ATTENTION_PATH", path, 1);
                const auto before_direct = argus_disk_page_descriptor(test_k, 0).access_count;
                const auto copies = argus_profile::d2d_bytes[argus_profile::prefill].load();
                const auto accepted = argus_profile::resident_accepted[prefill].load();
                compute(result, 0, nullptr);
                require(argus_profile::resident_accepted[prefill] == accepted + 1);
                std::vector<float> direct(second.size());
                output_gpu.read(direct.data(), 0, direct.size() * sizeof(float));
                if (direct != second) {
                    for (size_t i = 0; i < direct.size(); ++i) if (direct[i] != second[i]) {
                        std::fprintf(stderr, "resident mismatch D=%d path=%s shifted=%d mask=%d index=%zu actual=%a expected=%a\n",
                            d, path, shifted, int(test_mask->type), i, direct[i], second[i]);
                        break;
                    }
                }
                require(direct == second); // No relaxation of the original stored-value reference gate.
                require(argus_profile::d2d_bytes[argus_profile::prefill] == copies);
                require(argus_disk_page_descriptor(test_k, 0).access_count - before_direct == after_staged - before_staged);
            }
        }
        if (d == 64) {
            // Distinct K/V stores and Dk != Dv exercise both lock and address paths.
            auto * value_ctx = ggml_init({1 << 20, nullptr, true});
            auto * narrow_v = ggml_new_tensor_3d(value_ctx, GGML_TYPE_F16, 48, kv_heads, cells);
            setenv("ARGUS_KV_GPU_CONTROL", "1", 1);
            auto * value_store = ggml_backend_alloc_ctx_tensors_from_buft(value_ctx, argus_ggml_disk_buffer_type());
            unsetenv("ARGUS_KV_GPU_CONTROL");
            require(value_store);
            ggml_backend_tensor_set(narrow_v, values.data(), 0, ggml_nbytes(narrow_v));
            auto * result = argus_ggml_cuda_attention(ctx, q, rk, narrow_v, float_mask, 0.125f);
            result->data = output_gpu.data();
            std::vector<float> expected(48 * heads * tokens), actual(expected.size());
            setenv("ARGUS_KV_ATTENTION_PATH", "staged", 1);
            compute(result, 0, nullptr);
            output_gpu.read(expected.data(), 0, expected.size() * sizeof(float));
            setenv("ARGUS_KV_ATTENTION_PATH", "batched", 1);
            compute(result, 0, nullptr);
            output_gpu.read(actual.data(), 0, actual.size() * sizeof(float));
            require(actual == expected);
            ggml_backend_buffer_free(value_store);
            ggml_free(value_ctx);

            check_qwen_geometry(rng);
        }
        std::vector<const void *> pointers(2 * ((keys.size() * 2 + 4095) / 4096));
        const auto accesses = argus_disk_page_descriptor(rk, 0).access_count;
        refuses([&] { argus_disk_read_resident(rk, rv, pointers.data(), pointers.size(),
            [](size_t, size_t, void *) { throw std::runtime_error("consumer failed"); }, nullptr); });
        require(argus_disk_page_descriptor(rk, 0).access_count == accesses);
        // A concurrent writer cannot replace/free borrowed GPU pages.
        std::promise<void> entered, attempted;
        auto started = entered.get_future();
        auto attempting = attempted.get_future();
        auto writer = std::async(std::launch::async, [&] {
            started.wait(); attempted.set_value();
            ggml_backend_tensor_set(rk, keys.data(), 0, sizeof(ggml_fp16_t));
        });
        struct Race { std::promise<void> & entered; std::future<void> & attempting, & writer; } race{entered, attempting, writer};
        require(argus_disk_read_resident(rk, rv, pointers.data(), pointers.size(), [](size_t, size_t, void * context) {
            auto & race = *static_cast<Race *>(context);
            race.entered.set_value(); race.attempting.wait();
            require(race.writer.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
        }, &race));
        writer.get();
        auto * one_query = ggml_view_3d(ctx, q, d, heads, 1, q->nb[1], q->nb[2], q->nb[2]);
        auto * decode = argus_ggml_cuda_attention(ctx, one_query, rk, rv, float_mask, 0.125f);
        decode->data = output_gpu.data();
        const auto resident_decodes = argus_profile::resident_calls[argus_profile::decode].load();
        const auto q1 = resident_rejected[argus_profile::decode][argus_profile::reject_q1].load();
        compute(decode, 0, nullptr);
        require(argus_profile::resident_calls[argus_profile::decode] == resident_decodes);
        require(resident_rejected[argus_profile::decode][argus_profile::reject_q1] == q1 + 1);
        ggml_backend_buffer_clear(resident_store, 0);
        auto * zeros = argus_ggml_cuda_attention(ctx, q, rk, rv, mask, 1.0f);
        zeros->data = output_gpu.data();
        compute(zeros, 0, nullptr);
        output_gpu.read(second.data(), 0, second.size() * sizeof(float));
        require(std::all_of(second.begin(), second.end(), [](float x) { return x == 0.0f; }));
        unsetenv("ARGUS_KV_ATTENTION_PATH");
        ggml_backend_buffer_free(resident_store);
        ggml_free(resident_ctx);
    }
    const auto pre_reset = argus_disk_page_revision(k, 0);
    ggml_backend_buffer_clear(store, 0);
    const auto cleared = argus_disk_page_descriptor(k, 0);
    require(cleared.codec == GGML_TYPE_F16 && cleared.placement == ArgusTier::disk && !cleared.written);
    require(cleared.access_count == 0 && cleared.last_access_step == 0);
    refuses([&] { argus_disk_move_page(k, 0, ArgusTier::gpu, pre_reset); });
    require(argus_tier_usage().gpu == 0 && argus_tier_usage().pinned == 0 && argus_tier_usage().ram == 0);
    ggml_backend_buffer_free(store);
    ggml_free(ctx);
    check_policy();
    check_gpu_control();
    if (argus_profile::enabled()) {
        using namespace argus_profile;
        require(nanoseconds[prefill][attention] > 0);
        require((nanoseconds[prefill][kernel_gpu] > 0) == cuda_events());
        require((nanoseconds[prefill][h2d_gpu] > 0) == cuda_events());
        require((nanoseconds[prefill][d2d_gpu] > 0) == cuda_events());
        require(h2d_bytes[prefill] > 0 && d2d_bytes[prefill] > 0);
        require(nanoseconds[prefill][synchronization] > 0 && nanoseconds[other][disk_read] > 0);
        uint64_t exclusive = 0;
        for (int m = 0; m < metric_count; ++m) { exclusive += exclusive_nanoseconds[prefill][m].load(); }
        require(exclusive == nanoseconds[prefill][attention].load());
    }
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAILED: %s\n", error.what());
    return 1;
}
