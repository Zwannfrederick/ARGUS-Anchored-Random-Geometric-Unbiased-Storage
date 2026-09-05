#include "manager.h"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

using argus::CodecKind;
using argus::TierCodec;

// Single generic dequantization entry point from quantization_kernels.cu.
// See tier_codec.h for what the (kind, bits, pack_factor) triple means.
extern "C" {
void launch_dequantize_generic(const void *input, const float scale,
                               const float min_val, half *output,
                               int num_bytes, int kind, int bits,
                               int pack_factor, cudaStream_t stream);
cudaError_t launch_quantize_ggml_block(const float *input, uint8_t *output,
                                       int num_blocks, int kind,
                                       cudaStream_t stream);
cudaError_t launch_dequantize_ggml_block(const void *input, float *output,
                                         int num_blocks, int kind,
                                         cudaStream_t stream);
}

namespace {

constexpr float ggml_fp16_max = 65504.0f;
constexpr float ggml_q8_0_max_input = ggml_fp16_max * 127.0f;
constexpr float ggml_q4_0_max_input = ggml_fp16_max * 8.0f;

int checked_block_count(const int64_t element_count,
                        const TierCodec &codec) {
  const int64_t block_count = element_count / codec.block_size();
  TORCH_CHECK(block_count <= std::numeric_limits<int>::max(),
              "[ARGUS C++] tier '", codec.name,
              "' page contains too many blocks for the CUDA launch ABI: ",
              block_count, ".");
  return static_cast<int>(block_count);
}

// Logging is opt-in: the previous unconditional per-call std::cout showed up
// as a measurable fraction of decode time at long context.
bool verbose_from_env() {
  const char *v = std::getenv("ARGUS_VERBOSE");
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// Packs `pack_factor` sub-byte fields of `bits` bits each along the last axis
// into single bytes, little-end first. This is the inverse of the unpacking
// the generic CUDA kernel performs, expressed in ATen so it stays correct on
// any device the caller's tensors happen to live on.
at::Tensor pack_subbyte(const at::Tensor &q, int bits, int pack_factor) {
  namespace idx = torch::indexing;
  at::Tensor packed =
      q.index({"...", idx::Slice(0, idx::None, pack_factor)}).clone();
  for (int i = 1; i < pack_factor; ++i) {
    packed = packed +
             q.index({"...", idx::Slice(i, idx::None, pack_factor)}) *
                 (1 << (bits * i));
  }
  return packed;
}

} // namespace

ArgusCppManager::ArgusCppManager(int page_size, int max_active_pages,
                                 int device_id)
    : max_active_pages_(max_active_pages), generation_step_(0),
      page_size_(page_size), device_id_(device_id), page_counter_(0),
      stop_prefetch_worker_(false) {

  verbose_ = verbose_from_env();

  // Initialize Zero-Copy Host Memory Pool
  host_pool_ = std::make_unique<ZeroCopyHostPool>(device_id_);

  // Create async prefetch stream
  cudaError_t err = cudaStreamCreate(&prefetch_stream_);
  if (err != cudaSuccess) {
    std::cerr << "[ARGUS C++] Failed to create prefetch stream: "
              << cudaGetErrorString(err) << std::endl;
  }

  // Start background prefetch worker thread
  prefetch_worker_ = std::thread(&ArgusCppManager::prefetch_worker_loop, this);

  // Default per-tier capacities. Python overrides these from the TierSpec
  // list during construction; these only matter if the manager is driven
  // directly from C++.
  tier_max_pages_["fp8"] = 1;
  tier_max_pages_["int8"] = 1;
  tier_max_pages_["int4"] = 1;
  tier_max_pages_["int2"] = 1;
  tier_max_pages_["one_bit"] = 999;
  tier_max_pages_["jl"] = 999;
}

ArgusCppManager::~ArgusCppManager() {
  // Stop prefetch worker
  {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    stop_prefetch_worker_ = true;
  }
  prefetch_cv_.notify_all();
  if (prefetch_worker_.joinable()) {
    prefetch_worker_.join();
  }

  if (prefetch_stream_) {
    cudaStreamDestroy(prefetch_stream_);
  }

  active_pages_.clear();
  pages_by_tier_.clear();
  prefetch_cache_.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Codec-driven compression core
// ─────────────────────────────────────────────────────────────────────────────

void ArgusCppManager::compress_page(const std::shared_ptr<Page> &page,
                                    const TierCodec &codec) {
  const c10::cuda::CUDAGuard device_guard{
      static_cast<c10::DeviceIndex>(device_id_)};
  TORCH_CHECK(page->key_tensor.defined() && page->value_tensor.defined(),
              "[ARGUS C++] compress_page called on a page with no resident "
              "key/value tensors (tier=",
              page->tier_name, ", page_id=", page->page_id, ").");

  if (codec.is_block_quantized()) {
    const cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(device_id_).stream();
    for (int which = 0; which < 2; ++which) {
      const at::Tensor &src =
          which == 0 ? page->key_tensor : page->value_tensor;
      at::Tensor &dst =
          which == 0 ? page->compressed_key : page->compressed_value;
      TORCH_CHECK(src.size(-1) % codec.block_size() == 0,
                  "[ARGUS C++] tier '", codec.name, "' requires the last ",
                  "dimension to be a multiple of ", codec.block_size(),
                  ", got ", src.size(-1), ".");

      // GGML's reference quantizers consume fp32. Preserve that decision
      // surface for fp16, bf16, and fp32 callers instead of narrowing values
      // to half before choosing scales and integer codes.
      const at::Tensor float_src = src.to(torch::kFloat32).contiguous();
      const float input_amax =
          torch::max(torch::abs(float_src)).item<float>();
      const float max_input = codec.kind == CodecKind::GgmlQ8_0
                                  ? ggml_q8_0_max_input
                                  : ggml_q4_0_max_input;
      TORCH_CHECK(std::isfinite(input_amax) && input_amax <= max_input,
                  "[ARGUS C++] tier '", codec.name,
                  "' cannot represent this block range with an fp16 scale: ",
                  "amax=", input_amax, ", maximum=", max_input, ".");
      std::vector<int64_t> compressed_shape = src.sizes().vec();
      compressed_shape.back() =
          src.size(-1) / codec.block_size() * codec.block_bytes();
      at::Tensor packed = torch::empty(
          compressed_shape,
          torch::TensorOptions().device(src.device()).dtype(torch::kUInt8));
      const int num_blocks = checked_block_count(src.numel(), codec);
      const cudaError_t launch_error = launch_quantize_ggml_block(
          float_src.data_ptr<float>(),
          packed.data_ptr<uint8_t>(), num_blocks,
          static_cast<int>(codec.kind), stream);
      TORCH_CHECK(launch_error == cudaSuccess,
                  "[ARGUS C++] ", codec.name,
                  " quantization launch failed: ",
                  cudaGetErrorString(launch_error));
      const cudaError_t sync_error = cudaStreamSynchronize(stream);
      TORCH_CHECK(sync_error == cudaSuccess,
                  "[ARGUS C++] ", codec.name,
                  " quantization failed: ", cudaGetErrorString(sync_error));
      dst = host_pool_->tensor_to_pinned(packed);
    }
    page->key_scale = 0.0f;
    page->value_scale = 0.0f;
    page->key_min = 0.0f;
    page->value_min = 0.0f;
  } else if (codec.kind == CodecKind::Projection) {
    at::Tensor w_proj = get_jl_w_proj(page->page_size, page->key_tensor);
    auto comp_k = torch::matmul(w_proj, page->key_tensor.to(w_proj.scalar_type()));
    auto comp_v = torch::matmul(w_proj, page->value_tensor.to(w_proj.scalar_type()));
    page->compressed_key = host_pool_->tensor_to_pinned(comp_k);
    page->compressed_value = host_pool_->tensor_to_pinned(comp_v);
    // A projection carries no per-page scalar metadata; the reconstruction
    // operator holds everything needed to invert it.
    page->key_scale = 0.0f;
    page->value_scale = 0.0f;
    page->key_min = 0.0f;
    page->value_min = 0.0f;
  } else if (codec.kind == CodecKind::Passthrough) {
    // Unknown/plugin tier with no registered storage format: spill verbatim.
    // Costs no space but never corrupts data, which is the right default.
    page->compressed_key = host_pool_->tensor_to_pinned(page->key_tensor);
    page->compressed_value = host_pool_->tensor_to_pinned(page->value_tensor);
    page->key_scale = 1.0f;
    page->value_scale = 1.0f;
    page->key_min = 0.0f;
    page->value_min = 0.0f;
  } else {
    const float levels = codec.levels();
    const int pack_factor = codec.pack_factor();

    // key and value are quantized independently so a page's two halves can
    // have very different dynamic ranges without one clipping the other.
    for (int which = 0; which < 2; ++which) {
      const at::Tensor &src =
          (which == 0) ? page->key_tensor : page->value_tensor;
      float &scale_out = (which == 0) ? page->key_scale : page->value_scale;
      float &min_out = (which == 0) ? page->key_min : page->value_min;
      at::Tensor &dst =
          (which == 0) ? page->compressed_key : page->compressed_value;

      at::Tensor q;

      // Note the explicit round(): casting a float tensor to an integer dtype
      // truncates toward zero, which biases every value by up to half a
      // quantization step and roughly doubles reconstruction error on the
      // low-bit affine tiers. Rounding to nearest is what the error budgets
      // in the fidelity tests assume.
      if (codec.kind == CodecKind::SignedLinear) {
        const float amax =
            torch::max(torch::abs(src)).to(torch::kFloat32).item<float>();
        scale_out = (amax > 0.0f) ? (amax / levels) : 1.0f;
        min_out = 0.0f;
        q = (src / scale_out).round().clamp(-levels, levels).to(torch::kChar);
      } else if (codec.kind == CodecKind::UnsignedAffine) {
        const float vmin = torch::min(src).to(torch::kFloat32).item<float>();
        const float vmax = torch::max(src).to(torch::kFloat32).item<float>();
        scale_out = (vmax > vmin) ? ((vmax - vmin) / levels) : 1.0f;
        min_out = vmin;
        q = ((src - vmin) / scale_out).round().clamp(0, levels).to(torch::kByte);
      } else { // SignPacked
        // The reconstructed value is ±scale, so scale is the single magnitude
        // representing every element. The L2-optimal choice is the mean
        // absolute value, not the maximum: argmin_s E[(|x| - s)^2] = E[|x|].
        // Using the max inflates the reconstruction norm by |x|max / E[|x|]
        // (about 3.5x for Gaussian input), which dominated 1-bit error even
        // though the signs were correct.
        const float mean_abs =
            torch::mean(torch::abs(src.to(torch::kFloat32))).item<float>();
        scale_out = (mean_abs > 0.0f) ? mean_abs : 1.0f;
        min_out = 0.0f;
        q = (src >= 0.0f).to(torch::kByte);
      }

      if (pack_factor > 1) {
        TORCH_CHECK(src.size(-1) % pack_factor == 0,
                    "[ARGUS C++] tier '", codec.name, "' packs ", pack_factor,
                    " elements per byte, but the page's last dimension (",
                    src.size(-1), ") is not a multiple of it.");
        q = pack_subbyte(q, codec.bits, pack_factor);
      }

      dst = host_pool_->tensor_to_pinned(q.contiguous());
    }
  }

  page->key_tensor = torch::Tensor();
  page->value_tensor = torch::Tensor();
}

std::pair<at::Tensor, at::Tensor>
ArgusCppManager::decompress_page(const std::shared_ptr<Page> &page,
                                 const TierCodec &codec, cudaStream_t stream,
                                 bool allow_python_callbacks,
                                 bool synchronize) {
  const c10::cuda::CUDAGuard device_guard{
      static_cast<c10::DeviceIndex>(device_id_)};
  at::Tensor k_out, v_out;

  if (codec.kind == CodecKind::Projection) {
    auto gpu_k = page->compressed_key.to(torch::kCUDA);
    auto gpu_v = page->compressed_value.to(torch::kCUDA);

    at::Tensor recon_op;
    if (allow_python_callbacks) {
      recon_op = get_jl_recon_operator(page->page_size, gpu_k, page->page_size);
    } else {
      // Background thread: the lazy provider would need the GIL, so only an
      // already-cached operator is usable. Returning undefined tensors here
      // makes the caller skip this page rather than cache garbage.
      auto it = jl_recon_operator_by_len_.find(page->page_size);
      if (it == jl_recon_operator_by_len_.end() || !it->second.defined()) {
        return {at::Tensor(), at::Tensor()};
      }
      recon_op = it->second;
    }

    k_out = torch::matmul(recon_op, gpu_k.to(recon_op.scalar_type()));
    v_out = torch::matmul(recon_op, gpu_v.to(recon_op.scalar_type()));
  } else if (codec.kind == CodecKind::Passthrough) {
    k_out = page->compressed_key.to(torch::kCUDA);
    v_out = page->compressed_value.to(torch::kCUDA);
  } else if (codec.is_block_quantized()) {
    const int block_bytes = codec.block_bytes();
    TORCH_CHECK(page->compressed_key.size(-1) % block_bytes == 0 &&
                    page->compressed_value.size(-1) % block_bytes == 0,
                "[ARGUS C++] malformed ", codec.name,
                " page: compressed row does not contain whole blocks.");

    std::vector<int64_t> key_shape = page->compressed_key.sizes().vec();
    std::vector<int64_t> val_shape = page->compressed_value.sizes().vec();
    key_shape.back() = key_shape.back() / block_bytes * codec.block_size();
    val_shape.back() = val_shape.back() / block_bytes * codec.block_size();

    const auto options = torch::TensorOptions()
                             .device(torch::kCUDA, device_id_)
                             .dtype(torch::kFloat32);
    k_out = torch::empty(key_shape, options);
    v_out = torch::empty(val_shape, options);

    const void *k_ptr =
        host_pool_->get_device_pointer(page->compressed_key.data_ptr());
    const void *v_ptr =
        host_pool_->get_device_pointer(page->compressed_value.data_ptr());
    at::Tensor staged_k;
    at::Tensor staged_v;
    if (k_ptr == nullptr) {
      staged_k = torch::empty(
          page->compressed_key.sizes(),
          torch::TensorOptions()
              .device(torch::kCUDA, device_id_)
              .dtype(page->compressed_key.scalar_type()));
      const cudaError_t copy_error = cudaMemcpyAsync(
          staged_k.data_ptr(), page->compressed_key.data_ptr(),
          page->compressed_key.nbytes(), cudaMemcpyHostToDevice, stream);
      TORCH_CHECK(copy_error == cudaSuccess,
                  "[ARGUS C++] ", codec.name,
                  " K staging failed: ", cudaGetErrorString(copy_error));
      k_ptr = staged_k.data_ptr();
    }
    if (v_ptr == nullptr) {
      staged_v = torch::empty(
          page->compressed_value.sizes(),
          torch::TensorOptions()
              .device(torch::kCUDA, device_id_)
              .dtype(page->compressed_value.scalar_type()));
      const cudaError_t copy_error = cudaMemcpyAsync(
          staged_v.data_ptr(), page->compressed_value.data_ptr(),
          page->compressed_value.nbytes(), cudaMemcpyHostToDevice, stream);
      TORCH_CHECK(copy_error == cudaSuccess,
                  "[ARGUS C++] ", codec.name,
                  " V staging failed: ", cudaGetErrorString(copy_error));
      v_ptr = staged_v.data_ptr();
    }

    const int key_blocks = checked_block_count(k_out.numel(), codec);
    const int value_blocks = checked_block_count(v_out.numel(), codec);
    const cudaError_t key_launch_error = launch_dequantize_ggml_block(
        k_ptr, k_out.data_ptr<float>(),
        key_blocks,
        static_cast<int>(codec.kind), stream);
    TORCH_CHECK(key_launch_error == cudaSuccess,
                "[ARGUS C++] ", codec.name,
                " K dequantization launch failed: ",
                cudaGetErrorString(key_launch_error));
    const cudaError_t value_launch_error = launch_dequantize_ggml_block(
        v_ptr, v_out.data_ptr<float>(),
        value_blocks,
        static_cast<int>(codec.kind), stream);
    TORCH_CHECK(value_launch_error == cudaSuccess,
                "[ARGUS C++] ", codec.name,
                " V dequantization launch failed: ",
                cudaGetErrorString(value_launch_error));

    if (synchronize || staged_k.defined() || staged_v.defined()) {
      const cudaError_t sync_error = cudaStreamSynchronize(stream);
      TORCH_CHECK(sync_error == cudaSuccess,
                  "[ARGUS C++] ", codec.name,
                  " dequantization failed: ", cudaGetErrorString(sync_error));
    }
  } else {
    const int pack_factor = codec.pack_factor();

    std::vector<int64_t> key_shape = page->compressed_key.sizes().vec();
    std::vector<int64_t> val_shape = page->compressed_value.sizes().vec();
    key_shape.back() *= pack_factor;
    val_shape.back() *= pack_factor;

    auto options = torch::TensorOptions()
                       .device(torch::kCUDA, device_id_)
                       .dtype(torch::kFloat16);
    k_out = torch::empty(key_shape, options);
    v_out = torch::empty(val_shape, options);

    // Prefer the zero-copy device mapping of the pinned host buffer; fall
    // back to an explicit H2D copy when the page isn't pool-managed.
    const void *k_ptr =
        host_pool_->get_device_pointer(page->compressed_key.data_ptr());
    const void *v_ptr =
        host_pool_->get_device_pointer(page->compressed_value.data_ptr());

    at::Tensor staged_k, staged_v;
    if (!k_ptr) {
      staged_k = page->compressed_key.to(torch::kCUDA);
      staged_v = page->compressed_value.to(torch::kCUDA);
      k_ptr = staged_k.data_ptr();
      v_ptr = staged_v.data_ptr();
    }

    const int kind_id = static_cast<int>(codec.kind);
    launch_dequantize_generic(
        k_ptr, page->key_scale, page->key_min,
        reinterpret_cast<half *>(k_out.data_ptr<at::Half>()),
        page->compressed_key.numel(), kind_id, codec.bits, pack_factor, stream);
    launch_dequantize_generic(
        v_ptr, page->value_scale, page->value_min,
        reinterpret_cast<half *>(v_out.data_ptr<at::Half>()),
        page->compressed_value.numel(), kind_id, codec.bits, pack_factor,
        stream);

    if (synchronize) {
      // Must happen before staged_k/staged_v fall out of scope and before the
      // dtype cast below, which is issued on the ambient stream.
      cudaStreamSynchronize(stream);
    }
  }

  // Dequant always lands in fp16 (or the projection operator's dtype); restore
  // whatever dtype the caller originally pushed so mixed-dtype models stay
  // consistent across the demote/resurrect cascade.
  if (k_out.defined() && k_out.scalar_type() != page->orig_dtype) {
    k_out = k_out.to(page->orig_dtype);
    v_out = v_out.to(page->orig_dtype);
  }

  return {k_out, v_out};
}

// ─────────────────────────────────────────────────────────────────────────────

void ArgusCppManager::push_new_tokens(torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(k.device().is_cuda(), "K tensor must be on a CUDA device.");
  TORCH_CHECK(v.device().is_cuda(), "V tensor must be on a CUDA device.");
  TORCH_CHECK(k.device().index() == device_id_, "K tensor device ID mismatch.");
  TORCH_CHECK(v.device().index() == device_id_, "V tensor device ID mismatch.");

  generation_step_++;

  int num_tokens = k.size(-2);

  auto page = std::make_shared<Page>();
  page->page_id = page_counter_++;
  page->pool_slot = active_pages_.size();
  page->page_size = num_tokens;
  page->importance_score = 1.0f;
  page->attention_sum = 0.0f;
  page->last_step_accessed = generation_step_;
  page->tier_name = "active";
  page->orig_dtype = k.scalar_type();
  page->key_tensor = k;
  page->value_tensor = v;

  active_pages_.push_back(page);
  if (verbose_) {
    std::cout << "[ARGUS C++] Added page " << page->page_id
              << " to Active FP16 pool." << std::endl;
  }

  manage_memory_lifecycle();
}

torch::Tensor ArgusCppManager::inplace_paged_attention(
    torch::Tensor q, float scale, c10::optional<torch::Tensor> sink_k_opt,
    c10::optional<torch::Tensor> sink_v_opt, c10::optional<torch::Tensor> anchor_k_opt,
    c10::optional<torch::Tensor> anchor_v_opt, c10::optional<torch::Tensor> k_buffer_opt,
    c10::optional<torch::Tensor> v_buffer_opt, float resurrection_threshold) {
  TORCH_CHECK(q.device().is_cuda(), "Q tensor must be on a CUDA device.");
  TORCH_CHECK(q.device().index() == device_id_, "Q tensor device ID mismatch.");

  torch::Tensor sink_k = sink_k_opt.value_or(torch::Tensor());
  torch::Tensor sink_v = sink_v_opt.value_or(torch::Tensor());
  torch::Tensor anchor_k = anchor_k_opt.value_or(torch::Tensor());
  torch::Tensor anchor_v = anchor_v_opt.value_or(torch::Tensor());
  torch::Tensor k_buffer = k_buffer_opt.value_or(torch::Tensor());
  torch::Tensor v_buffer = v_buffer_opt.value_or(torch::Tensor());

  generation_step_++;

  std::vector<at::Tensor> keys_list;
  std::vector<at::Tensor> values_list;
  std::vector<std::shared_ptr<Page>> pages_in_order;

  // 0. Sinks & VIP Anchors
  if (sink_k.defined() && sink_k.numel() > 0) {
    keys_list.push_back(sink_k);
    values_list.push_back(sink_v);
    pages_in_order.push_back(nullptr);
  }
  if (anchor_k.defined() && anchor_k.numel() > 0) {
    keys_list.push_back(anchor_k);
    values_list.push_back(anchor_v);
    pages_in_order.push_back(nullptr);
  }

  // 1. Active FP16 Pages
  for (auto &page : active_pages_) {
    keys_list.push_back(page->key_tensor);
    values_list.push_back(page->value_tensor);
    pages_in_order.push_back(page);
  }

  // 1.5 Temp Active Buffer
  if (k_buffer.defined() && k_buffer.numel() > 0) {
    keys_list.push_back(k_buffer);
    values_list.push_back(v_buffer);
    pages_in_order.push_back(nullptr);
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id_).stream();

  // 2. Compressed tiers, coldest-last, decompressed through the tier's codec.
  for (const auto &tier : tier_pipeline_) {
    const TierCodec &codec = codec_for(tier);
    for (auto &page : pages_by_tier_[tier]) {
      if (streaming_attention_) {
        // Keep only metadata in the block list. The online-softmax loop below
        // materializes and releases one compressed page at a time, bounding
        // the FP16 reconstruction workspace to a single page.
        keys_list.push_back(at::Tensor());
        values_list.push_back(at::Tensor());
        pages_in_order.push_back(page);
        continue;
      }
      at::Tensor k_temp;
      at::Tensor v_temp;

      {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        auto cache_it = prefetch_cache_.find(page->page_id);
        if (cache_it != prefetch_cache_.end()) {
          k_temp = cache_it->second.first;
          v_temp = cache_it->second.second;
          prefetch_hit_count_++;
        }
      }

      if (!k_temp.defined()) {
        std::tie(k_temp, v_temp) = decompress_page(
            page, codec, stream, /*allow_python_callbacks=*/true,
            /*synchronize=*/false);
      }

      TORCH_CHECK(k_temp.defined(),
                  "[ARGUS C++] failed to decompress page ", page->page_id,
                  " from tier '", tier, "'.");

      keys_list.push_back(k_temp);
      values_list.push_back(v_temp);
      pages_in_order.push_back(page);
    }
  }

  if (keys_list.empty()) {
    return torch::zeros_like(q);
  }

  at::Tensor k_full;
  at::Tensor v_full;
  at::Tensor attn_output;
  std::vector<float> streaming_block_weights;

  if (streaming_attention_) {
    // FlashAttention's online-softmax recurrence, expressed in ATen as a
    // correctness-first prototype.  Each K/V page is consumed independently;
    // no context-sized contiguous K/V tensor is materialized.  The temporary
    // score tensor is bounded by one page (and q_len), while accumulation is
    // performed in fp32 for numerical stability.
    at::NoGradGuard no_grad;
    at::Tensor q_float = q.to(torch::kFloat32);
    std::vector<int64_t> stat_shape(q.sizes().begin(), q.sizes().end());
    stat_shape.back() = 1;
    std::vector<int64_t> out_shape(q.sizes().begin(), q.sizes().end());
    out_shape.back() = q.size(-1);
    auto float_options = q.options().dtype(torch::kFloat32);
    at::Tensor running_max = at::full(
        stat_shape, -std::numeric_limits<float>::infinity(), float_options);
    at::Tensor running_sum = at::zeros(stat_shape, float_options);
    at::Tensor running_out = at::zeros(out_shape, float_options);
    std::vector<at::Tensor> block_logsumexp;
    block_logsumexp.reserve(keys_list.size());

    for (size_t i = 0; i < keys_list.size(); ++i) {
      at::Tensor materialized_k = keys_list[i];
      at::Tensor materialized_v = values_list[i];
      if (!materialized_k.defined()) {
        auto page = pages_in_order[i];
        TORCH_CHECK(page, "Compressed attention block is missing page metadata.");
        {
          std::lock_guard<std::mutex> lock(prefetch_mutex_);
          auto cache_it = prefetch_cache_.find(page->page_id);
          if (cache_it != prefetch_cache_.end()) {
            materialized_k = cache_it->second.first;
            materialized_v = cache_it->second.second;
            prefetch_hit_count_++;
          }
        }
        if (!materialized_k.defined()) {
          std::tie(materialized_k, materialized_v) = decompress_page(
              page, codec_for(page->tier_name), stream,
              /*allow_python_callbacks=*/true,
              /*synchronize=*/false);
        }
      }
      at::Tensor block_k = materialized_k.to(torch::kFloat32);
      at::Tensor block_v = materialized_v.to(torch::kFloat32);
      TORCH_CHECK(block_k.dim() == 4 && block_v.dim() == 4,
                  "Streaming attention expects rank-4 K/V tensors.");
      TORCH_CHECK(block_k.size(0) == q_float.size(0) &&
                      block_v.size(0) == q_float.size(0),
                  "Streaming attention batch size mismatch.");
      TORCH_CHECK(block_k.size(-2) == block_v.size(-2),
                  "Streaming attention K/V token count mismatch.");
      TORCH_CHECK(block_k.size(-3) == block_v.size(-3),
                  "Streaming attention K/V head count mismatch.");
      TORCH_CHECK(block_k.size(-1) == q_float.size(-1),
                  "Streaming attention query/key head dimension mismatch.");
      TORCH_CHECK(block_v.size(-1) == q_float.size(-1),
                  "Streaming attention currently requires value and query head dimensions to match.");
      const int64_t q_heads = q_float.size(-3);
      const int64_t kv_heads = block_k.size(-3);
      if (q_heads != kv_heads) {
        TORCH_CHECK(q_heads % kv_heads == 0,
                    "Query heads must be divisible by KV heads for GQA.");
        const int64_t groups = q_heads / kv_heads;
        block_k = at::repeat_interleave(block_k, groups, -3);
        block_v = at::repeat_interleave(block_v, groups, -3);
      }

      at::Tensor scores =
          at::matmul(q_float, block_k.transpose(-1, -2)) * scale;
      at::Tensor block_max = std::get<0>(scores.max(-1, true));
      at::Tensor next_max = at::maximum(running_max, block_max);
      at::Tensor prior_scale = at::exp(running_max - next_max);
      at::Tensor block_exp = at::exp(scores - next_max);
      running_out = running_out * prior_scale + at::matmul(block_exp, block_v);
      running_sum = running_sum * prior_scale + block_exp.sum(-1, true);
      running_max = next_max;
      block_logsumexp.push_back(at::logsumexp(scores, {-1}, true));
    }

    attn_output = (running_out / running_sum).to(q.scalar_type());

    // Per-page attention mass for the existing QoS/resurrection policy.  A
    // block's logsumexp minus the global logsumexp is exactly the sum of its
    // normalized softmax probabilities, so this avoids rebuilding scores for
    // a concatenated cache.
    at::Tensor global_logsumexp = running_max + at::log(running_sum);
    streaming_block_weights.reserve(block_logsumexp.size());
    for (const auto &block_lse : block_logsumexp) {
      at::Tensor mass = at::exp(block_lse - global_logsumexp);
      mass = mass.mean(0).mean(0);
      mass = mass.select(0, mass.size(0) - 1);
      streaming_block_weights.push_back(mass.item<float>());
    }
  } else {
    k_full = at::cat(keys_list, -2);
    v_full = at::cat(values_list, -2);

    const int64_t q_heads = q.size(-3);
    const int64_t kv_heads = k_full.size(-3);
    if (q_heads != kv_heads) {
      TORCH_CHECK(q_heads % kv_heads == 0,
                  "Query heads must be divisible by KV heads for GQA.");
      const int64_t groups = q_heads / kv_heads;
      k_full = at::repeat_interleave(k_full, groups, -3);
      v_full = at::repeat_interleave(v_full, groups, -3);
    }

    // Fused FlashAttention / SDPA execution
    attn_output = at::scaled_dot_product_attention(
        q, k_full, v_full,
        /*attn_mask=*/c10::nullopt,
        /*dropout_p=*/0.0,
        /*is_causal=*/false,
        /*scale=*/scale);
  }

  // Eager Bypass: with no memory pressure and nothing compressed yet, skip
  // the QoS/importance/resurrection bookkeeping entirely — full native
  // SDPA speed on the common case. Mirrors the pre-C++-port Python behavior;
  // without it, importance_score churns on every call and prematurely
  // triggers variable-granularity splitting for pages that were never
  // actually under pressure.
  bool has_compressed = false;
  for (const auto &tier : tier_pipeline_) {
    if (!pages_by_tier_[tier].empty()) {
      has_compressed = true;
      break;
    }
  }
  bool under_pressure = active_pages_.size() >= static_cast<size_t>(max_active_pages_);
  if (!has_compressed && !under_pressure && !force_qos_) {
    return attn_output;
  }

  // Compute QoS attention weights and trigger resurrection
  at::Tensor weights;
  if (!streaming_attention_) {
    at::NoGradGuard no_grad;
    at::Tensor q_scaled = q * scale;
    at::Tensor scores = at::matmul(q_scaled, k_full.transpose(-1, -2));
    weights = at::softmax(scores, -1);
    weights = weights.mean(0).mean(0);
    weights = weights.select(0, weights.size(0) - 1);
  }

  int idx_start = 0;
  std::vector<std::shared_ptr<Page>> resurrect_list;
  for (size_t i = 0; i < pages_in_order.size(); ++i) {
    auto page = pages_in_order[i];
    int block_len = keys_list[i].defined()
                        ? keys_list[i].size(-2)
                        : (page ? page->page_size : 0);

    if (!page) {
      idx_start += block_len;
      continue;
    }

    float block_w = streaming_attention_
                        ? streaming_block_weights[i]
                        : weights.slice(0, idx_start, idx_start + block_len)
                              .sum()
                              .to(torch::kFloat32)
                              .item<float>();
    page->attention_sum += block_w;
    if (block_w > 0.0f) {
      page->last_step_accessed = generation_step_;
    }

    // Let the pluggable eviction policy observe this access (reference bits,
    // heat registers, importance recalculation) — mirrors the pre-C++-port
    // Python hot path so custom EvictionPolicy subclasses stay effective.
    if (!on_page_access_callback_.is_none()) {
      on_page_access_callback_(page, block_w, generation_step_);
    }

    if (page->tier_name != "active" && block_w > resurrection_threshold) {
      resurrect_list.push_back(page);
    }

    idx_start += block_len;
  }

  for (auto &page : resurrect_list) {
    resurrect_page(page, page->tier_name);
  }

  manage_memory_lifecycle();

  return attn_output;
}

std::shared_ptr<Page> ArgusCppManager::select_active_victim() {
  if (active_pages_.empty()) {
    return nullptr;
  }

  // Defer to the pluggable eviction policy (Python) when one is registered;
  // otherwise fall back to the built-in lowest-importance scan. Safe to call
  // back into Python here: this path only runs synchronously from
  // push_new_tokens/inplace_paged_attention, both invoked from Python with
  // the GIL already held.
  if (!active_pool_victim_selector_.is_none()) {
    int idx = active_pool_victim_selector_(active_pages_).cast<int>();
    TORCH_CHECK(idx >= 0 && static_cast<size_t>(idx) < active_pages_.size(),
                "[ARGUS C++] active_pool_victim_selector returned out-of-range index.");
    return active_pages_[idx];
  }

  int lowest_idx = 0;
  float lowest_importance = active_pages_[0]->importance_score;
  for (size_t i = 1; i < active_pages_.size(); ++i) {
    if (active_pages_[i]->importance_score < lowest_importance) {
      lowest_importance = active_pages_[i]->importance_score;
      lowest_idx = i;
    }
  }
  return active_pages_[lowest_idx];
}

void ArgusCppManager::manage_memory_lifecycle() {
  // 1. Active Pool Kontrolü (max_active_pages sınırını aşarsa demote et)
  while (active_pages_.size() > static_cast<size_t>(max_active_pages_)) {
    auto page_to_demote = select_active_victim();
    if (!page_to_demote) {
      break;
    }
    active_pages_.erase(std::remove(active_pages_.begin(), active_pages_.end(),
                                    page_to_demote),
                        active_pages_.end());

    for (size_t i = 0; i < active_pages_.size(); ++i) {
      active_pages_[i]->pool_slot = i;
    }

    demote_to_next_tier(page_to_demote);
  }

  // 2. Kademeli Katman Sınırı Kontrolleri (Tiers Spec limitlerini aşarsa bir
  // alt katmana demote et)
  for (size_t i = 0; i < tier_pipeline_.size(); ++i) {
    std::string current_tier = tier_pipeline_[i];
    std::string next_tier = (i + 1 < tier_pipeline_.size()) ? tier_pipeline_[i+1] : "";

    while (pages_by_tier_[current_tier].size() >
           static_cast<size_t>(tier_max_pages_[current_tier])) {
      // En eski sayfayı bul ve demote et
      auto page_to_cascade = pages_by_tier_[current_tier].front();

      // Mevcut katmandan sil
      auto &list = pages_by_tier_[current_tier];
      list.erase(list.begin());

      if (next_tier.empty()) {
        // Terminal tier (or a single-tier pipeline) has no lower tier to
        // cascade into — the page is evicted from the cache entirely.
        host_pool_->free_tensor(page_to_cascade->compressed_key);
        host_pool_->free_tensor(page_to_cascade->compressed_value);
        continue;
      }

      // Round-trip through fp16: decompress from the current tier, then
      // recompress into the next one.
      resurrect_page(page_to_cascade, current_tier);
      // Cascaded sayfayı active_pages_'den temizliyoruz çünkü sadece bir alt
      // katmana geçiş yapıyoruz.
      active_pages_.erase(std::remove(active_pages_.begin(),
                                      active_pages_.end(), page_to_cascade),
                          active_pages_.end());

      page_to_cascade->tier_name = next_tier;
      demote_to_next_tier(page_to_cascade);
    }
  }
}

void ArgusCppManager::demote_to_next_tier(std::shared_ptr<Page> page) {
  // When demotion comes straight from the active pool the target is the first
  // tier of whatever pipeline is configured — NOT a hardcoded "fp8". A custom
  // single-tier pipeline (e.g. int8-only) must demote into its own first tier,
  // or pages never reach it. Otherwise manage_memory_lifecycle already set
  // page->tier_name to the cascade target.
  std::string target_tier = page->tier_name;
  if (target_tier == "active") {
    TORCH_CHECK(!tier_pipeline_.empty(),
                "[ARGUS C++] cannot demote from the active pool: the tier "
                "pipeline is empty. Configure at least one tier.");
    target_tier = tier_pipeline_.front();
  }

  page->tier_name = target_tier;

  const TierCodec &codec = codec_for(target_tier);
  compress_page(page, codec);
  pages_by_tier_[target_tier].push_back(page);

  if (verbose_) {
    std::cout << "[ARGUS C++] Demoted page " << page->page_id << " to tier '"
              << target_tier << "' (" << codec.effective_bits()
              << " effective bits, swapped to zero-copy pinned host)."
              << std::endl;
  }
}

void ArgusCppManager::resurrect_page(std::shared_ptr<Page> page,
                                     std::string current_tier) {
  // Fast path: the background prefetcher already materialized this page.
  {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    auto cache_it = prefetch_cache_.find(page->page_id);
    if (cache_it != prefetch_cache_.end()) {
      page->key_tensor = cache_it->second.first;
      page->value_tensor = cache_it->second.second;

      auto &old_list = pages_by_tier_[current_tier];
      old_list.erase(std::remove(old_list.begin(), old_list.end(), page),
                     old_list.end());

      if (page->key_tensor.scalar_type() != page->orig_dtype) {
        page->key_tensor = page->key_tensor.to(page->orig_dtype);
        page->value_tensor = page->value_tensor.to(page->orig_dtype);
      }

      page->tier_name = "active";
      page->last_step_accessed = generation_step_;
      active_pages_.push_back(page);

      prefetch_cache_.erase(cache_it);
      prefetch_hit_count_++;

      if (verbose_) {
        std::cout << "[ARGUS C++] Resurrected page " << page->page_id
                  << " from prefetch cache (zero dequant latency)." << std::endl;
      }
      return;
    }
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id_).stream();
  const TierCodec &codec = codec_for(current_tier);

  auto kv = decompress_page(page, codec, stream, /*allow_python_callbacks=*/true,
                            /*synchronize=*/true);
  TORCH_CHECK(kv.first.defined(), "[ARGUS C++] failed to resurrect page ",
              page->page_id, " from tier '", current_tier, "'.");

  page->key_tensor = kv.first;
  page->value_tensor = kv.second;

  // Free pinned backing host memory
  host_pool_->free_tensor(page->compressed_key);
  host_pool_->free_tensor(page->compressed_value);

  page->compressed_key = torch::Tensor();
  page->compressed_value = torch::Tensor();

  page->tier_name = "active";
  page->last_step_accessed = generation_step_;

  active_pages_.push_back(page);

  auto &old_list = pages_by_tier_[current_tier];
  old_list.erase(std::remove(old_list.begin(), old_list.end(), page),
                 old_list.end());

  if (verbose_) {
    std::cout << "[ARGUS C++] Resurrected page " << page->page_id << " from '"
              << current_tier << "' back to active FP16." << std::endl;
  }
}

std::pair<at::Tensor, at::Tensor> ArgusCppManager::peek_decompress_page(
    std::shared_ptr<Page> page, std::string current_tier) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id_).stream();
  auto kv = decompress_page(page, codec_for(current_tier), stream,
                            /*allow_python_callbacks=*/true,
                            /*synchronize=*/true);
  TORCH_CHECK(kv.first.defined(), "[ARGUS C++] failed to peek-decompress page ",
              page->page_id, " from tier '", current_tier, "'.");
  return kv;
}

void ArgusCppManager::speculate_and_prefetch(const std::vector<int> &page_ids) {
  std::lock_guard<std::mutex> lock(prefetch_mutex_);

  // Clear old prefetch queue and cache to limit footprint
  prefetch_cache_.clear();
  while (!prefetch_queue_.empty())
    prefetch_queue_.pop();

  for (int pid : page_ids) {
    std::shared_ptr<Page> target_page = nullptr;
    for (const auto &pair : pages_by_tier_) {
      for (auto &page : pair.second) {
        if (page->page_id == pid) {
          target_page = page;
          break;
        }
      }
      if (target_page)
        break;
    }
    if (target_page) {
      prefetch_queue_.push(target_page);
    }
  }
  prefetch_cv_.notify_one();
}

void ArgusCppManager::prefetch_worker_loop() {
  // Set active CUDA device for the worker thread
  cudaSetDevice(device_id_);

  auto mark_idle = [this]() {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    prefetch_worker_busy_ = false;
    prefetch_cv_.notify_all();
  };

  while (true) {
    std::shared_ptr<Page> page = nullptr;
    {
      std::unique_lock<std::mutex> lock(prefetch_mutex_);
      prefetch_cv_.wait(lock, [this]() {
        return stop_prefetch_worker_ || !prefetch_queue_.empty();
      });

      if (stop_prefetch_worker_) {
        break;
      }

      page = prefetch_queue_.front();
      prefetch_queue_.pop();
      prefetch_worker_busy_ = true;
    }

    // Skip if already resurrected to the active pool, or if the compressed
    // payload was released out from under us.
    if (!page || page->tier_name == "active" ||
        !page->compressed_key.defined()) {
      mark_idle();
      continue;
    }

    // No GIL on this thread — decompress_page must not invoke the Python
    // JL providers, so a projection page with no cached operator is skipped.
    auto kv = decompress_page(page, codec_for(page->tier_name),
                              prefetch_stream_,
                              /*allow_python_callbacks=*/false,
                              /*synchronize=*/true);

    if (!kv.first.defined()) {
      // Caching undefined tensors would surface as an empty/garbage page on
      // the next attention call; drop the speculation instead.
      mark_idle();
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      prefetch_cache_[page->page_id] = std::make_pair(kv.first, kv.second);
      prefetch_worker_busy_ = false;
      prefetch_cv_.notify_all();
    }
  }
}

int ArgusCppManager::get_page_count() const {
  int total = active_pages_.size();
  for (const auto &pair : pages_by_tier_) {
    total += pair.second.size();
  }
  return total;
}

int ArgusCppManager::get_active_page_count() const {
  return active_pages_.size();
}

int ArgusCppManager::get_tier_page_count(const std::string &tier_name) const {
  auto it = pages_by_tier_.find(tier_name);
  return (it != pages_by_tier_.end()) ? it->second.size() : 0;
}
