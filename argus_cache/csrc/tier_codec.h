// ARGUS tier codec registry.
//
// Before this existed, manager.cpp dispatched compression/decompression by
// comparing the tier *name* against "fp8"/"int8"/"int4"/"int2"/"one_bit"/"jl"
// in four separate if/else-if chains (demote, resurrect, peek, prefetch).
// Adding a tier meant editing all four; a plugin tier could never be more than
// an uncompressed passthrough.
//
// A TierCodec describes a tier's storage format numerically instead — bit
// width, packing factor, and how the stored integers map back to floats — so
// every call site can share one compress/decompress implementation and a new
// tier is a registry entry, not a new branch.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace argus {

// How a codec's stored integers relate to the original float values.
enum class CodecKind {
  // q is a signed integer; value = q * scale. (fp8-simulated, int8)
  SignedLinear = 0,
  // q is an unsigned integer packed pack_factor-per-byte;
  // value = q * scale + min. (int4, int2)
  UnsignedAffine = 1,
  // q is a single sign bit packed 8-per-byte; value = ±scale. (one_bit)
  SignPacked = 2,
  // Compression is a learned/random linear map, not a quantizer; the page is
  // stored as matmul(w_proj, x) and restored via matmul(recon_op, q). (jl)
  Projection = 3,
  // No compression — page is moved to host memory verbatim. Default for
  // tiers registered without a format, so an unknown tier degrades to a
  // correct (if not space-saving) spill rather than corrupting data.
  Passthrough = 4,
};

struct TierCodec {
  std::string name;
  CodecKind kind = CodecKind::Passthrough;

  // Bits per stored element. 8 for SignedLinear, 4/2 for UnsignedAffine,
  // 1 for SignPacked, 16 for Projection/Passthrough.
  int bits = 16;

  // Whether the codec discards information. Capability metadata, surfaced to
  // Python so policy code can reason about "is this tier lossy" without
  // knowing the tier's name.
  bool lossy = true;

  // Elements packed into each stored byte. Derived from `bits`: sub-byte
  // codecs pack 8/bits elements along the last axis; byte-or-wider codecs
  // store one element per slot.
  int pack_factor() const { return bits < 8 ? 8 / bits : 1; }

  // Largest representable quantization level. SignedLinear reserves a sign
  // bit; UnsignedAffine uses the full unsigned range; SignPacked is ±scale so
  // it has a single magnitude level.
  float levels() const {
    switch (kind) {
    case CodecKind::SignedLinear:
      return static_cast<float>((1 << (bits - 1)) - 1);
    case CodecKind::UnsignedAffine:
      return static_cast<float>((1 << bits) - 1);
    case CodecKind::SignPacked:
      return 1.0f;
    default:
      return 1.0f;
    }
  }

  // True when decompression needs the per-page min offset (affine codecs).
  bool needs_min() const { return kind == CodecKind::UnsignedAffine; }

  // True when the codec runs through the generic pack/unpack CUDA path.
  // Projection and Passthrough tiers take their own route.
  bool is_quantized() const {
    return kind == CodecKind::SignedLinear ||
           kind == CodecKind::UnsignedAffine || kind == CodecKind::SignPacked;
  }

  // Storage cost relative to an fp16 baseline, e.g. 0.0625 for one_bit.
  // Projection tiers report their ratio via `compression_ratio` instead,
  // since their cost depends on the projection rank, not a bit width.
  float compression_ratio = 1.0f;

  float effective_bits() const { return 16.0f * compression_ratio; }
};

// Name -> codec mapping. Ships with ARGUS's six built-in tiers registered;
// Python may add, replace, or remove entries at runtime (see
// argus_cache.plugins).
class CodecRegistry {
public:
  CodecRegistry();

  // Registers or replaces a codec. Replacing a codec whose pages are already
  // resident is the caller's responsibility to avoid — the manager validates
  // this before allowing a swap.
  void register_codec(const TierCodec &codec);
  void unregister_codec(const std::string &name);

  bool has(const std::string &name) const;

  // Returns the registered codec, or a Passthrough codec named `name` when
  // the tier was never registered with a format. Never returns null, so call
  // sites don't need an unknown-tier branch.
  const TierCodec &get(const std::string &name) const;

  std::vector<std::string> names() const;

private:
  std::unordered_map<std::string, TierCodec> codecs_;
  TierCodec fallback_;
};

} // namespace argus
