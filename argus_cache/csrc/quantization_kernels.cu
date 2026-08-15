// Generic ARGUS dequantization kernel.
//
// This file previously carried five near-identical kernels (fp8, int8, int4,
// int2, one_bit) that differed only in bit width and how a stored integer maps
// back to a float. They are now one kernel parameterized by the same TierCodec
// descriptor the manager dispatches on, so registering a new quantized tier
// gets a working CUDA path without adding a kernel.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// Mirrors argus::CodecKind. Kept as a plain int in the ABI so the .cu unit
// doesn't need to include the C++ header.
#define ARGUS_KIND_SIGNED_LINEAR 0
#define ARGUS_KIND_UNSIGNED_AFFINE 1
#define ARGUS_KIND_SIGN_PACKED 2
#define ARGUS_KIND_GGML_Q8_0 5
#define ARGUS_KIND_GGML_Q4_0 6

namespace {

constexpr int ggml_block_size = 32;
constexpr int ggml_q8_0_block_bytes = 34;
constexpr int ggml_q4_0_block_bytes = 18;

__global__ void quantize_ggml_block_kernel(const float *__restrict__ input,
                                           uint8_t *__restrict__ output,
                                           const int num_blocks,
                                           const int kind) {
  const int block_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_index >= num_blocks) {
    return;
  }

  const float *src = input + block_index * ggml_block_size;
  float amax = 0.0f;
  float signed_max = 0.0f;
  for (int i = 0; i < ggml_block_size; ++i) {
    const float value = src[i];
    const float magnitude = fabsf(value);
    if (magnitude > amax) {
      amax = magnitude;
      signed_max = value;
    }
  }

  if (kind == ARGUS_KIND_GGML_Q8_0) {
    uint8_t *dst = output + block_index * ggml_q8_0_block_bytes;
    const float scale = amax / 127.0f;
    *reinterpret_cast<half *>(dst) = __float2half(scale);
    const float inverse_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
    for (int i = 0; i < ggml_block_size; ++i) {
      const int quant = static_cast<int>(roundf(src[i] * inverse_scale));
      dst[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(quant));
    }
    return;
  }

  uint8_t *dst = output + block_index * ggml_q4_0_block_bytes;
  const float scale = signed_max / -8.0f;
  *reinterpret_cast<half *>(dst) = __float2half(scale);
  const float inverse_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
  for (int i = 0; i < ggml_block_size / 2; ++i) {
    const float low_value = src[i] * inverse_scale;
    const float high_value = src[i + ggml_block_size / 2] * inverse_scale;
    const int low = max(0, min(15, static_cast<int>(low_value + 8.5f)));
    const int high = max(0, min(15, static_cast<int>(high_value + 8.5f)));
    dst[2 + i] = static_cast<uint8_t>(low | (high << 4));
  }
}

__global__ void dequantize_ggml_block_kernel(
    const uint8_t *__restrict__ input, float *__restrict__ output,
    const int num_blocks, const int kind) {
  const int block_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_index >= num_blocks) {
    return;
  }

  float *dst = output + block_index * ggml_block_size;
  const int block_bytes = kind == ARGUS_KIND_GGML_Q8_0
                              ? ggml_q8_0_block_bytes
                              : ggml_q4_0_block_bytes;
  const uint8_t *src = input + block_index * block_bytes;
  const float scale = __half2float(*reinterpret_cast<const half *>(src));

  if (kind == ARGUS_KIND_GGML_Q8_0) {
    for (int i = 0; i < ggml_block_size; ++i) {
      const int8_t quant = static_cast<int8_t>(src[2 + i]);
      dst[i] = static_cast<float>(quant) * scale;
    }
    return;
  }

  for (int i = 0; i < ggml_block_size / 2; ++i) {
    const uint8_t packed = src[2 + i];
    dst[i] = (static_cast<int>(packed & 0x0f) - 8) * scale;
    dst[i + ggml_block_size / 2] =
        (static_cast<int>(packed >> 4) - 8) * scale;
  }
}

} // namespace

// One thread per stored byte; each writes `pack_factor` output elements.
__global__ void dequantize_generic_kernel(const uint8_t *__restrict__ input,
                                          const float scale,
                                          const float min_val,
                                          half *__restrict__ output,
                                          const int num_bytes, const int kind,
                                          const int bits,
                                          const int pack_factor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_bytes) {
    return;
  }

  const uint8_t packed = input[idx];

  if (kind == ARGUS_KIND_SIGNED_LINEAR) {
    // Stored as a signed integer occupying the whole byte; value = q * scale.
    const float value = static_cast<float>(static_cast<int8_t>(packed)) * scale;
    output[idx] = __float2half(value);
    return;
  }

  if (kind == ARGUS_KIND_SIGN_PACKED) {
    // Eight sign bits per byte; set bit means +scale, clear means -scale.
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const bool sign = (packed >> b) & 0x01;
      output[idx * 8 + b] = __float2half(sign ? scale : -scale);
    }
    return;
  }

  // ARGUS_KIND_UNSIGNED_AFFINE: `pack_factor` unsigned fields of `bits` bits
  // each, little-end first; value = q * scale + min.
  const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
  const int base = idx * pack_factor;
  for (int i = 0; i < pack_factor; ++i) {
    const uint8_t q = (packed >> (bits * i)) & mask;
    output[base + i] = __float2half(static_cast<float>(q) * scale + min_val);
  }
}

extern "C" {

// `num_bytes` is the element count of the *stored* tensor; the kernel writes
// num_bytes * pack_factor halves into `output`.
void launch_dequantize_generic(const void *input, const float scale,
                               const float min_val, half *output,
                               const int num_bytes, const int kind,
                               const int bits, const int pack_factor,
                               cudaStream_t stream) {
  if (num_bytes <= 0) {
    return;
  }
  const int threads = 256;
  const int blocks = (num_bytes + threads - 1) / threads;
  dequantize_generic_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t *>(input), scale, min_val, output,
      num_bytes, kind, bits, pack_factor);
}

cudaError_t launch_quantize_ggml_block(const float *input, uint8_t *output,
                                       const int num_blocks, const int kind,
                                       cudaStream_t stream) {
  if (num_blocks <= 0) {
    return cudaSuccess;
  }
  constexpr int threads = 256;
  const int blocks = (num_blocks + threads - 1) / threads;
  quantize_ggml_block_kernel<<<blocks, threads, 0, stream>>>(
      input, output, num_blocks, kind);
  return cudaPeekAtLastError();
}

cudaError_t launch_dequantize_ggml_block(const void *input, float *output,
                                         const int num_blocks, const int kind,
                                         cudaStream_t stream) {
  if (num_blocks <= 0) {
    return cudaSuccess;
  }
  constexpr int threads = 256;
  const int blocks = (num_blocks + threads - 1) / threads;
  dequantize_ggml_block_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t *>(input), output, num_blocks, kind);
  return cudaPeekAtLastError();
}

} // extern "C"
