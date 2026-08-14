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

} // extern "C"
