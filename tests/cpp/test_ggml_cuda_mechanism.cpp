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
        const auto key_rejects = resident_rejected[prefill][argus_profile::reject_key_page].load();
        const auto censused = argus_profile::census[prefill].invocations.load();
        compute(out, 0, nullptr);
        // K page 0 is GPU, page 1 pinned: the first failing check is a nonresident K page.
        require(resident_rejected[prefill][argus_profile::reject_key_page] == key_rejects + 1);
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
        setenv("ARGUS_KV_NO_OVERLAP", "1", 1);
        compute(out, 0, nullptr);
        output_gpu.read(second.data(), 0, second.size() * sizeof(float));
        require(first == second);
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
            for (const auto * path : {"direct", "batched"}) {
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

            // Qwen2.5-0.5B geometry: 14 Q / 2 KV heads, D=64, one causal 64-token ubatch.
            setenv("ARGUS_KV_GPU_BYTES", "4194304", 1);
            const int qwen_heads = 14, qwen_tokens = 64, qwen_cells = 500;
            auto * qwen_ctx = ggml_init({1 << 20, nullptr, true});
            auto * qwen_k = ggml_new_tensor_3d(qwen_ctx, GGML_TYPE_F16, 64, 2, qwen_cells);
            auto * qwen_v = ggml_new_tensor_3d(qwen_ctx, GGML_TYPE_F16, 64, 2, qwen_cells);
            setenv("ARGUS_KV_GPU_CONTROL", "1", 1);
            auto * qwen_store = ggml_backend_alloc_ctx_tensors_from_buft(qwen_ctx, argus_ggml_disk_buffer_type());
            unsetenv("ARGUS_KV_GPU_CONTROL");
            require(qwen_store);
            std::vector<ggml_fp16_t> qwen_kv(64 * 2 * qwen_cells);
            for (auto * tensor : {qwen_k, qwen_v}) {
                for (auto & x : qwen_kv) { x = ggml_fp32_to_fp16(normal(rng)); }
                ggml_backend_tensor_set(tensor, qwen_kv.data(), 0, qwen_kv.size() * 2);
            }
            auto * qwen_q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, qwen_heads, qwen_tokens);
            auto * qwen_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, qwen_cells, qwen_tokens);
            std::vector<float> qwen_query(64 * qwen_heads * qwen_tokens);
            for (auto & x : qwen_query) { x = normal(rng); }
            std::vector<ggml_fp16_t> qwen_bias(qwen_cells * qwen_tokens);
            for (int t = 0; t < qwen_tokens; ++t) {
                for (int c = 0; c < qwen_cells; ++c) { qwen_bias[t * qwen_cells + c] = ggml_fp32_to_fp16(c <= 436 + t ? 0.0f : -INFINITY); }
            }
            ArgusTierBuffer qwen_q_gpu(ArgusTier::gpu, qwen_query.size() * sizeof(float));
            ArgusTierBuffer qwen_mask_gpu(ArgusTier::gpu, qwen_bias.size() * 2);
            ArgusTierBuffer qwen_out_gpu(ArgusTier::gpu, qwen_query.size() * sizeof(float));
            qwen_q_gpu.write(qwen_query.data(), qwen_query.size() * sizeof(float));
            qwen_mask_gpu.write(qwen_bias.data(), qwen_bias.size() * 2);
            qwen_q->data = qwen_q_gpu.data(); qwen_mask->data = qwen_mask_gpu.data();
            auto * qwen = argus_ggml_cuda_attention(ctx, qwen_q, qwen_k, qwen_v, qwen_mask, 0.125f);
            qwen->data = qwen_out_gpu.data();
            std::vector<float> qwen_expected(qwen_query.size()), qwen_actual(qwen_query.size());
            setenv("ARGUS_KV_ATTENTION_PATH", "staged", 1);
            compute(qwen, 0, nullptr);
            qwen_out_gpu.read(qwen_expected.data(), 0, qwen_expected.size() * sizeof(float));
            for (const auto * path : {"direct", "batched"}) {
                setenv("ARGUS_KV_ATTENTION_PATH", path, 1);
                const auto accepted = argus_profile::resident_accepted[prefill].load();
                compute(qwen, 0, nullptr);
                require(argus_profile::resident_accepted[prefill] == accepted + 1);
                qwen_out_gpu.read(qwen_actual.data(), 0, qwen_actual.size() * sizeof(float));
                require(qwen_actual == qwen_expected);
            }
            ggml_backend_buffer_free(qwen_store);
            ggml_free(qwen_ctx);
            setenv("ARGUS_KV_GPU_BYTES", "1048576", 1);
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
