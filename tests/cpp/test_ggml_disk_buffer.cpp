// Include storage internals only in this standalone failure-injection check.
#include "ggml_disk_buffer.cpp"
#include "ggml-alloc.h"
#include <cassert>

static bool short_write = false;
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" ssize_t __wrap_pwrite(int fd, const void * data, size_t count, off_t offset) {
    return __real_pwrite(fd, data, short_write ? count / 2 : count, offset);
}

int main(int argc, char ** argv) {
    assert(argc == 2);
    setenv("ARGUS_KV_DIR", argv[1], 1);
    setenv("ARGUS_KV_MAX_BYTES", "65536", 1);
    setenv("ARGUS_KV_RESIDENT_BYTES", "8192", 1);
    setenv("ARGUS_KV_STAGING_BYTES", "16384", 1);
    ggml_init_params init{1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(init);
    auto * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, argus_ggml_disk_buffer_type());
    assert(buffer && !ggml_backend_buffer_is_host(buffer));
    auto descriptor = argus_disk_page_descriptor(tensor, 0);
    assert(descriptor.codec == GGML_TYPE_F32 && descriptor.placement == ArgusTier::disk);
    assert(!descriptor.written && descriptor.access_count == 0 && descriptor.last_access_step == 0);
    auto * view = ggml_view_1d(ctx, tensor, 1, 8188);
    assert(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
    const auto tail = argus_disk_page_descriptor(view, 0);
    assert(tail.codec == GGML_TYPE_F32 && tail.revision.page == 1);
    ggml_tensor reinterpretation = *tensor;
    reinterpretation.type = GGML_TYPE_F16;
    assert(init_tensor(buffer, &reinterpretation) == GGML_STATUS_FAILED);
    reinterpretation.view_src = tensor;
    assert(init_tensor(buffer, &reinterpretation) == GGML_STATUS_SUCCESS);
    assert(argus_disk_page_descriptor(&reinterpretation, 0).codec == GGML_TYPE_F32);
    bool refused = false;
    try { argus_disk_page_descriptor(view, 4); }
    catch (const std::out_of_range &) { refused = true; }
    assert(refused);
    std::vector<unsigned char> expected(6000), actual(expected.size());
    for (size_t i = 0; i < expected.size(); ++i) { expected[i] = static_cast<unsigned char>(i * 37); }
    ggml_backend_tensor_set(tensor, expected.data(), 123, expected.size());
    assert(argus_disk_page_descriptor(tensor, 0).access_count == 0); // RMW is not a demand read.
    ggml_backend_tensor_get(tensor, actual.data(), 123, actual.size());
    assert(expected == actual);
    descriptor = argus_disk_page_descriptor(tensor, 0);
    assert(descriptor.written && descriptor.codec == GGML_TYPE_F32);
    assert(descriptor.access_count == 1 && descriptor.last_access_step == 1);
    assert(argus_disk_page_descriptor(view, 0).last_access_step == descriptor.last_access_step);

    auto & store = store_for(buffer);
    const auto generation = store.pages[0].content_revision;
    const auto placement = store.pages[0].placement_revision;
    const auto content = argus_disk_revision(tensor);
    const auto page = argus_disk_page_revision(tensor, 0);
    assert(page.allocation == content.allocation && page.page == 0 && page.content_revision == generation);
    short_write = true;
    refused = false;
    try { ggml_backend_tensor_set(tensor, actual.data(), 200, actual.size()); }
    catch (const std::runtime_error &) { refused = true; }
    short_write = false;
    assert(refused && store.pages[0].content_revision == generation);
    assert(store.pages[0].placement_revision == placement && content == argus_disk_revision(tensor));
    assert(argus_disk_page_descriptor(tensor, 0).access_count == descriptor.access_count);
    ggml_backend_tensor_get(tensor, actual.data(), 123, actual.size());
    assert(expected == actual); // A partial write never changed the published slot.
    const auto before_write = argus_disk_page_descriptor(tensor, 0);
    ggml_backend_tensor_set(tensor, expected.data(), 123, expected.size());
    descriptor = argus_disk_page_descriptor(tensor, 0);
    assert(descriptor.codec == before_write.codec && descriptor.access_count == before_write.access_count);
    assert(descriptor.last_access_step == before_write.last_access_step);

    // Corruption must be detected before bytes are returned to attention.
    {
        Bounce damage;
        std::memset(damage.data, 0xff, page_size);
        assert(__real_pwrite(store.fd, damage.data, page_size, offset(0, store.pages[0].active)) == page_size);
    }
    refused = false;
    const auto before_corruption = argus_disk_page_descriptor(tensor, 0);
    try { ggml_backend_tensor_get(tensor, actual.data(), 123, actual.size()); }
    catch (const std::runtime_error &) { refused = true; }
    assert(refused);
    assert(argus_disk_page_descriptor(tensor, 0).access_count == before_corruption.access_count);
    ggml_backend_buffer_clear(buffer, 0);
    descriptor = argus_disk_page_descriptor(tensor, 0);
    assert(descriptor.codec == GGML_TYPE_F32 && !descriptor.written);
    assert(descriptor.access_count == 0 && descriptor.last_access_step == 0);
    assert(store.pages[0].content_revision > generation);
    assert(store.pages[0].placement_revision > placement && !(content == argus_disk_revision(tensor)));
    ggml_backend_tensor_get(tensor, actual.data(), 123, actual.size());
    assert(std::all_of(actual.begin(), actual.end(), [](auto value) { return value == 0; }));

    // No CPU address can accidentally reconstruct or fault in the entire KV.
    unsigned char residency[2]{};
    assert(mincore(store.address, 8192, residency) == 0);
    assert(!(residency[0] & 1) && !(residency[1] & 1));
    {
        ArgusStagingBuffer held(12288);
        refused = false;
        try { ArgusStagingBuffer overflow(8192); }
        catch (const std::runtime_error &) { refused = true; }
        assert(refused && staging_live == 12288);
    }
    assert(staging_live == 0 && peak_staging <= 16384);
    const auto prefetch_budget = ArgusDiskPrefetch::stack_bytes + 65536;
    setenv("ARGUS_KV_STAGING_BYTES", std::to_string(prefetch_budget).c_str(), 1);
    {
        ggml_tensor cells = *tensor;
        cells.ne[0] = 16; cells.ne[1] = 1; cells.ne[2] = 128;
        cells.nb[1] = cells.nb[2] = 64;
        ArgusStagingBuffer keys(4096), values(4096);
        ArgusDiskPrefetch prefetch;
        prefetch.submit(&cells, &cells, 0, 64, keys.data(), values.data());
        refused = false;
        try { prefetch.submit(&cells, &cells, 0, 64, keys.data(), values.data()); }
        catch (const std::runtime_error &) { refused = true; }
        assert(refused);
        prefetch.take();
        assert(argus_disk_page_descriptor(tensor, 0).access_count == 3); // prior get plus K and V
        assert(std::all_of(static_cast<unsigned char *>(keys.data()),
                           static_cast<unsigned char *>(keys.data()) + 4096, [](auto value) { return value == 0; }));
        prefetch.submit(&cells, &cells, 0, 64, keys.data(), values.data());
        ggml_backend_buffer_clear(buffer, 0);
        refused = false;
        try { prefetch.take(); }
        catch (const std::runtime_error &) { refused = true; }
        assert(refused);
        // Destruction joins pending work before either destination is released.
        prefetch.submit(&cells, &cells, 0, 64, keys.data(), values.data());
    }
    assert(staging_live == 0 && peak_staging <= prefetch_budget);
    {
        // Adjacent, crossing-page and repeated/scattered row indices retain set_rows order.
        ggml_tensor rows = *tensor;
        rows.ne[0] = 16; rows.ne[1] = 128;
        rows.nb[1] = 64;
        auto * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 12);
        auto * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 12);
        std::vector<float> input(16 * 12), reference(2048), output(2048);
        for (size_t i = 0; i < input.size(); ++i) { input[i] = float(i + 1); }
        source->data = input.data();
        std::vector<std::vector<int64_t>> cases = {{0,1,2,3,4,5,6,7,8,9,10,11},
            {60,61,62,63,64,65,66,67,68,69,70,71}, {0,2,1,2,127,126,125,124,123,122,121,120}};
        for (size_t c = 0; c < cases.size(); ++c) {
            ggml_backend_buffer_clear(buffer, 0);
            std::fill(reference.begin(), reference.end(), 0);
            indices->data = cases[c].data();
            float result = 1;
            ggml_tensor operation{};
            operation.src[0] = source; operation.src[1] = indices; operation.src[2] = &rows;
            operation.data = &result;
            const auto writes = committed_pages.load();
            set_rows(&operation, 0, 1, nullptr);
            for (size_t row = 0; row < cases[c].size(); ++row) {
                std::copy_n(input.data() + row * 16, 16, reference.data() + cases[c][row] * 16);
            }
            ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
            assert(output == reference && result == 0);
            if (c < 2) { assert(committed_pages.load() - writes == c + 1); }
        }
    }
    {
        auto * quant_ctx = ggml_init(init);
        auto * q8 = ggml_new_tensor_2d(quant_ctx, GGML_TYPE_Q8_0, 256, 8);
        auto * q4 = ggml_new_tensor_2d(quant_ctx, GGML_TYPE_Q4_0, 256, 8);
        auto * quant_buffer = ggml_backend_alloc_ctx_tensors_from_buft(quant_ctx, argus_ggml_disk_buffer_type());
        assert(quant_buffer);
        std::vector<float> input(256 * 8);
        for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 71) - 35) / 17; }
        int64_t row_indices[] = {0,1,2,3,4,5,6,7};
        auto * source = ggml_new_tensor_2d(quant_ctx, GGML_TYPE_F32, 256, 8);
        auto * indices = ggml_new_tensor_1d(quant_ctx, GGML_TYPE_I64, 8);
        source->data = input.data(); indices->data = row_indices;
        for (auto * target : {q8, q4}) {
            const size_t row_bytes = ggml_row_size(target->type, 256);
            std::vector<unsigned char> reference(ggml_nbytes(target)), actual_quant(reference.size());
            for (size_t row = 0; row < 8; ++row) {
                ggml_get_type_traits_cpu(target->type)->from_float(input.data() + row * 256,
                    reference.data() + row * row_bytes, 256);
            }
            float result = 1;
            ggml_tensor operation{};
            operation.src[0] = source; operation.src[1] = indices; operation.src[2] = target;
            operation.data = &result;
            const auto writes = committed_pages.load();
            set_rows(&operation, 0, 1, nullptr);
            ggml_backend_tensor_get(target, actual_quant.data(), 0, actual_quant.size());
            assert(actual_quant == reference && result == 0 && committed_pages.load() - writes == 1);
        }
        ggml_backend_buffer_free(quant_buffer);
        ggml_free(quant_ctx);
    }
    store.access_step = store.pages[0].access_count = UINT64_MAX;
    ggml_backend_tensor_get(tensor, actual.data(), 0, 4);
    descriptor = argus_disk_page_descriptor(tensor, 0);
    assert(descriptor.access_count == UINT64_MAX && descriptor.last_access_step == UINT64_MAX);
    setenv("ARGUS_KV_RESIDENT_BYTES", "4096", 1);
    assert(!ggml_backend_buft_alloc_buffer(argus_ggml_disk_buffer_type(), 8192));
    setenv("ARGUS_KV_RESIDENT_BYTES", "8192", 1);
    setenv("ARGUS_KV_MAX_BYTES", "16384", 1);
    assert(!ggml_backend_buft_alloc_buffer(argus_ggml_disk_buffer_type(), 8192));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    assert(disk_live == 0 && metadata_live == 0 && staging_live == 0 && stores == 0);
    std::puts("direct storage: round-trip, short-write rollback, checksum, budgets, clear and teardown passed");
}
