// Real model parity and state lifecycle through the patched llama.cpp library.
// Usage: test_llama_host_ownership /path/to/stories15M.gguf
#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>

int main(int argc, char ** argv) {
    assert(argc == 2);
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], model_params);
    assert(model);
    auto params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_threads = 2;
    params.n_threads_batch = 2;
    params.offload_kqv = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    unsetenv("ARGUS_KV_DIR");
    auto * stock = llama_init_from_model(model, params);
    assert(stock);
    char directory[] = "/tmp/argus-llama-test-XXXXXX";
    assert(mkdtemp(directory));
    assert(setenv("ARGUS_KV_DIR", directory, 1) == 0);
    assert(setenv("ARGUS_KV_MAX_BYTES", "16777216", 1) == 0);
    auto * argus = llama_init_from_model(model, params);
    assert(argus);
    const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_token prompt[] = {1, 2, 3};
    assert(llama_decode(stock, llama_batch_get_one(prompt, 3)) == 0);
    assert(llama_decode(argus, llama_batch_get_one(prompt, 3)) == 0);
    float max_error = 0.0f;
    auto compare = [&]() {
        const auto * a = llama_get_logits_ith(stock, -1);
        const auto * b = llama_get_logits_ith(argus, -1);
        for (int i = 0; i < vocabulary; ++i) {
            assert(std::isfinite(a[i]) && std::isfinite(b[i]));
            const float error = std::abs(a[i] - b[i]);
            if (error > max_error) { max_error = error; }
            assert(error <= 1e-5f);
        }
    };
    compare();
    std::vector<uint8_t> saved(llama_state_get_size(stock));
    const size_t written = llama_state_get_data(stock, saved.data(), saved.size());
    assert(written > 0);
    assert(llama_state_set_data(argus, saved.data(), written) == written);
    // Continue after restoration: this consumes the stored K/V, not just the
    // weights or newest token, and compares every vocabulary logit.
    for (int step = 0; step < 8; ++step) {
        llama_token token = 4 + step;
        assert(llama_decode(stock, llama_batch_get_one(&token, 1)) == 0);
        assert(llama_decode(argus, llama_batch_get_one(&token, 1)) == 0);
        compare();
    }
    llama_free(argus);
    assert(setenv("ARGUS_KV_MAX_BYTES", "1", 1) == 0);
    assert(llama_init_from_model(model, params) == nullptr);
    llama_free(stock);
    llama_model_free(model);
    llama_backend_free();
    assert(rmdir(directory) == 0);
    std::printf("{\"logit_parity_passed\":true,\"max_abs_logit_error\":%.9g,"
                "\"decode_steps_after_restore\":8,\"budget_refusal_passed\":true}\n", max_error);
}
