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
    std::vector<unsigned char> expected(6000), actual(expected.size());
    for (size_t i = 0; i < expected.size(); ++i) { expected[i] = static_cast<unsigned char>(i * 37); }
    ggml_backend_tensor_set(tensor, expected.data(), 123, expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 123, actual.size());
    assert(expected == actual);

    auto & store = store_for(buffer);
    const auto generation = store.pages[0].generation;
    short_write = true;
    bool refused = false;
    try { ggml_backend_tensor_set(tensor, actual.data(), 200, actual.size()); }
    catch (const std::runtime_error &) { refused = true; }
    short_write = false;
    assert(refused && store.pages[0].generation == generation);
    ggml_backend_tensor_get(tensor, actual.data(), 123, actual.size());
    assert(expected == actual); // A partial write never changed the published slot.

    // Corruption must be detected before bytes are returned to attention.
    {
        Bounce damage;
        std::memset(damage.data, 0xff, page_size);
        assert(__real_pwrite(store.fd, damage.data, page_size, offset(0, store.pages[0].active)) == page_size);
    }
    refused = false;
    try { ggml_backend_tensor_get(tensor, actual.data(), 123, actual.size()); }
    catch (const std::runtime_error &) { refused = true; }
    assert(refused);
    ggml_backend_buffer_clear(buffer, 0);
    assert(store.pages[0].generation > generation);
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
    setenv("ARGUS_KV_STAGING_BYTES", "131072", 1);
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
    assert(staging_live == 0 && peak_staging <= 131072);
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
