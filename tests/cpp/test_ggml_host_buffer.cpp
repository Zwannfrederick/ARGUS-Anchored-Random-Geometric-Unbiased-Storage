// Standalone check: link with ggml-base and argus_cache/csrc/ggml_host_buffer.cpp.
#include "ggml_host_buffer.h"
#include "ggml-alloc.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

int main() {
    char directory[] = "/tmp/argus-host-test-XXXXXX";
    assert(mkdtemp(directory));
    assert(setenv("ARGUS_KV_DIR", directory, 1) == 0);
    assert(setenv("ARGUS_KV_MAX_BYTES", "1048576", 1) == 0);
    ggml_init_params parameters{1024 * 1024, nullptr, true};
    auto * context = ggml_init(parameters);
    assert(context);
    auto * keys = ggml_new_tensor_2d(context, GGML_TYPE_F32, 32, 16);
    auto * values = ggml_new_tensor_2d(context, GGML_TYPE_F32, 32, 16);
    auto type = argus_ggml_host_buffer_type();
    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(context, type);
    assert(buffer && ggml_backend_buffer_is_host(buffer));
    assert(std::strcmp(ggml_backend_buffer_name(buffer), "ARGUS_HOST") == 0);
    assert(keys->buffer == buffer && values->buffer == buffer);
    std::vector<float> input(512, 3.25f), output(512);
    ggml_backend_tensor_set(keys, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_copy(keys, values);
    ggml_backend_tensor_get(values, output.data(), 0, output.size() * sizeof(float));
    assert(input == output);
    auto * view = ggml_view_1d(context, keys, 32, 32 * sizeof(float));
    assert(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(view, output.data(), 0, 32 * sizeof(float));
    assert(output[0] == 3.25f);
    assert(setenv("ARGUS_KV_MAX_BYTES", "1", 1) == 0);
    assert(ggml_backend_buft_alloc_buffer(type, 4096) == nullptr);
    ggml_backend_tensor_get(keys, output.data(), 0, output.size() * sizeof(float));
    assert(input == output);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    // Closing the arena must return the complete allocation budget.
    assert(setenv("ARGUS_KV_MAX_BYTES", "4096", 1) == 0);
    buffer = ggml_backend_buft_alloc_buffer(type, 4096);
    assert(buffer);
    ggml_backend_buffer_free(buffer);
    assert(rmdir(directory) == 0);
}
