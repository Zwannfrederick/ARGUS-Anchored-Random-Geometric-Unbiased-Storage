// ARGUS paged attention through the patched llama.cpp library.
// Usage: test_llama_paged_attention MODEL.gguf STORAGE_DIR [KV_TYPE: f32|f16|q8_0|q4_0]
// Stock and ARGUS contexts share the model; both use flash attention settings so
// only KV ownership and the attention kernel differ.
#include "llama.h"
#include "ggml-backend.h"
#ifdef ARGUS_TEST_CUDA_TIER
#include "ggml_cuda_attention.h"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

size_t argus_rss_bytes() {
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    bool argus = false;
    size_t total = 0;
    while (std::getline(smaps, line)) {
        if (line.find('-') != std::string::npos && line.find(' ') != std::string::npos &&
            std::isxdigit(static_cast<unsigned char>(line[0])) && line.find(':') > line.find(' ')) {
            argus = line.find("/argus-ggml-") != std::string::npos || line.find("[anon:argus-disk-") != std::string::npos;
        } else if (argus && line.rfind("Rss:", 0) == 0) {
            total += std::stoull(line.substr(4)) * 1024;
        }
    }
    return total;
}

struct FirstAttention {
    std::vector<float> values;
    bool move_pages = false;
    int moved_pages = 0;
};

// Captures layer 0's attention output; later layers amplify rounding differently per model.
bool capture_first_attention(ggml_tensor * tensor, bool ask, void * user_data) {
    const bool wanted = std::strcmp(tensor->name, "kqv_out-0") == 0 && tensor->type == GGML_TYPE_F32;
    if (ask || !wanted) { return wanted || !ask; }
    auto & captured = *static_cast<FirstAttention *>(user_data);
    auto & values = captured.values;
    values.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
#ifdef ARGUS_TEST_CUDA_TIER
    if (captured.move_pages && !captured.moved_pages) {
        auto * operation = tensor;
        for (int depth = 0; operation && depth < 8 && !argus_ggml_is_cuda_attention(operation); ++depth) {
            operation = operation->src[0];
        }
        require(operation && argus_ggml_is_cuda_attention(operation), "CUDA attention node not found");
        // Explicit test placement, not an eviction or hotness policy in the runtime.
        for (int input : {1, 2}) {
            auto * kv = operation->src[input];
            argus_disk_move_page(kv, 0, input == 1 ? ArgusTier::gpu : ArgusTier::pinned, argus_disk_page_revision(kv, 0));
            ++captured.moved_pages;
        }
    }
#endif
    return true;
}

ggml_type parse_type(const char * name) {
    const std::string text = name ? name : "f16";
    if (text == "f32") { return GGML_TYPE_F32; }
    if (text == "f16") { return GGML_TYPE_F16; }
    if (text == "q8_0") { return GGML_TYPE_Q8_0; }
    if (text == "q4_0") { return GGML_TYPE_Q4_0; }
    throw std::runtime_error("unsupported KV type");
}
} // namespace

int main(int argc, char ** argv) try {
    require(argc >= 3, "usage: MODEL STORAGE_DIR [KV_TYPE]");
    const ggml_type kv_type = parse_type(argc > 3 ? argv[3] : nullptr);
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], model_params);
    require(model, "model load");
    const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));

    auto params = llama_context_default_params();
    params.n_ctx = 1024;
    params.n_batch = 1024;
    params.n_ubatch = 256;
    params.n_seq_max = 1;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.offload_kqv = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.type_k = kv_type;
    params.type_v = kv_type;

    unsetenv("ARGUS_KV_DIR");
    unsetenv("ARGUS_KV_RESIDENT_BYTES");
    // Reference attention: GGML matmul + F32 softmax for float KV (stock CPU flash
    // attention accumulates F16 V in F16), stock flash attention for quantized KV.
    FirstAttention stock_attention, argus_attention;
    auto reference = params;
    reference.cb_eval = capture_first_attention;
    reference.cb_eval_user_data = &stock_attention;
    const bool float_kv = kv_type == GGML_TYPE_F16 || kv_type == GGML_TYPE_F32;
    if (float_kv) { reference.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; }
    auto * stock = llama_init_from_model(model, reference);
    require(stock, "stock context");

    char directory[4096];
    std::snprintf(directory, sizeof directory, "%s/argus-paged-test-XXXXXX", argv[2]);
    require(mkdtemp(directory), "mkdtemp");
    const std::string stats = std::string(directory) + "/stats.json";
    const char * budget_text = std::getenv("ARGUS_TEST_RESIDENT_BYTES");
    const size_t resident_budget = budget_text ? std::stoull(budget_text) : (1u << 20);
    setenv("ARGUS_KV_DIR", directory, 1);
    setenv("ARGUS_KV_MAX_BYTES", "268435456", 1);
    setenv("ARGUS_KV_RESIDENT_BYTES", std::to_string(resident_budget).c_str(), 1);
    setenv("ARGUS_KV_BLOCK_CELLS", "32", 0);
    setenv("ARGUS_KV_STATS_PATH", stats.c_str(), 1);

    // Fail closed: paged KV without flash-attention layout must not silently fall back.
    auto unsupported = params;
    unsupported.cb_eval = nullptr;
    unsupported.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    unsupported.type_k = unsupported.type_v = GGML_TYPE_F16;
    auto * refused = llama_init_from_model(model, unsupported);
    require(refused == nullptr, "unsupported paged contract must refuse context creation");

    params.cb_eval = capture_first_attention;
    params.cb_eval_user_data = &argus_attention;
#ifdef ARGUS_TEST_CUDA_TIER
    argus_attention.move_pages = true;
#endif
    auto * argus = llama_init_from_model(model, params);
    require(argus, "argus context");

    // Logit parity is required for numerically stable models; chaotic ones (e.g. Gemma 4
    // on random tokens) amplify float rounding ~10x per layer even between stock kernels.
    const bool strict_logits = !std::getenv("ARGUS_TEST_REPORT_LOGITS_ONLY");
    double max_error = 0.0, attention_error = 0.0;
    int compared_steps = 0, greedy_mismatches = 0;
    auto compare = [&]() {
        require(!stock_attention.values.empty() && stock_attention.values.size() == argus_attention.values.size(),
                "first attention output was not captured");
        for (size_t i = 0; i < stock_attention.values.size(); ++i) {
            attention_error = std::max(attention_error, double(std::abs(stock_attention.values[i] - argus_attention.values[i])));
        }
        const float * a = llama_get_logits_ith(stock, -1);
        const float * b = llama_get_logits_ith(argus, -1);
        int best_a = 0, best_b = 0;
        for (int i = 0; i < vocabulary; ++i) {
            require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite logit");
            max_error = std::max(max_error, double(std::abs(a[i] - b[i])));
            if (a[i] > a[best_a]) { best_a = i; }
            if (b[i] > b[best_b]) { best_b = i; }
        }
        greedy_mismatches += best_a != best_b;
        require(!strict_logits || best_a == best_b, "greedy token differs");
        ++compared_steps;
        return best_a;
    };
    auto decode_both = [&](std::vector<llama_token> tokens) {
        require(llama_decode(stock, llama_batch_get_one(tokens.data(), tokens.size())) == 0, "stock decode");
        require(llama_decode(argus, llama_batch_get_one(tokens.data(), tokens.size())) == 0, "argus decode");
        return compare();
    };

    // 600-token prompt in 256-token ubatches: most KV blocks become cold.
    const char * prompt_text = std::getenv("ARGUS_TEST_PROMPT_TOKENS");
    std::vector<llama_token> prompt(prompt_text ? std::stoul(prompt_text) : 600);
    for (size_t i = 0; i < prompt.size(); ++i) { prompt[i] = 1 + static_cast<llama_token>((i * 7919) % (vocabulary - 1)); }
    llama_token next = decode_both(prompt);
    for (int step = 0; step < 8; ++step) { next = decode_both({next}); }

    // Crop the last 100 positions and continue with different tokens.
    const llama_pos crop = static_cast<llama_pos>(prompt.size() * 5 / 6);
    require(llama_memory_seq_rm(llama_get_memory(stock), 0, crop, -1), "stock crop");
    require(llama_memory_seq_rm(llama_get_memory(argus), 0, crop, -1), "argus crop");
    next = decode_both({11, 12, 13, 14});
    for (int step = 0; step < 4; ++step) { next = decode_both({next}); }

    // Save the paged context, punch-clear it, restore, then continue from stored K/V.
    std::vector<uint8_t> saved(llama_state_get_size(argus));
    const size_t written = llama_state_get_data(argus, saved.data(), saved.size());
    require(written > 0, "state save");
    llama_memory_clear(llama_get_memory(argus), true);
    require(llama_state_set_data(argus, saved.data(), written) == written, "state restore");
    for (int step = 0; step < 4; ++step) { next = decode_both({next}); }

    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    next = decode_both({next});
    const size_t rss = argus_rss_bytes();
    std::ifstream stats_stream(stats);
    std::stringstream stats_json;
    stats_json << stats_stream.rdbuf();
    std::string published = stats_json.str();
    while (!published.empty() && std::isspace(static_cast<unsigned char>(published.back()))) { published.pop_back(); }
    require(!published.empty(), "no runtime statistics were published");
    require(std::getenv("ARGUS_KV_STAGING_BYTES") || budget_text ||
            published.find("\"paged_out_bytes\":0,") == std::string::npos, "no page-out was recorded");
    // Slack: the hot window is computed in cells, so each K/V tensor (at most two per
    // layer) may keep one partial boundary page.
    const size_t boundary_slack = static_cast<size_t>(llama_model_n_layer(model)) * 2 * 4096;
    if (!budget_text && rss > resident_budget + boundary_slack) {
        throw std::runtime_error("argus KV RSS " + std::to_string(rss) + " exceeds budget plus boundary-page slack");
    }
    if (attention_error > 1e-3) {
        throw std::runtime_error("first-layer attention error above tolerance: " + std::to_string(attention_error));
    }
    if (strict_logits && max_error > 5e-2) {
        throw std::runtime_error("logit error above tolerance: " + std::to_string(max_error));
    }

    llama_free(argus);
    llama_free(stock);
#ifdef ARGUS_TEST_CUDA_TIER
    require(argus_attention.moved_pages == 2, "native pages were not migrated");
    const auto remaining = argus_tier_usage();
    require(!remaining.gpu && !remaining.pinned && !remaining.ram, "native tier allocations leaked");
#endif
    setenv("ARGUS_KV_MAX_BYTES", "1", 1);
    require(llama_init_from_model(model, params) == nullptr, "1-byte allocation budget must refuse");
    llama_model_free(model);
    llama_backend_free();
    std::remove(stats.c_str());
    rmdir(directory);
    std::printf("{\"kv_type\":\"%s\",\"compared_steps\":%d,\"max_abs_first_attention_error\":%.9g,"
                "\"greedy_mismatches\":%d,\"strict_logits\":%s,\"max_abs_logit_error\":%.9g,"
                "\"argus_kv_rss_bytes\":%zu,\"resident_budget_bytes\":%zu,\"migrated_pages\":%d,\"stats\":%s}\n",
                argc > 3 ? argv[3] : "f16", compared_steps, attention_error, greedy_mismatches,
                strict_logits ? "true" : "false", max_error, rss, resident_budget, argus_attention.moved_pages,
                published.empty() ? "null" : published.c_str());
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAILED: %s\n", error.what());
    return 1;
}
