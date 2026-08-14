#include "tier_codec.h"

namespace argus {

CodecRegistry::CodecRegistry() {
  fallback_.name = "";
  fallback_.kind = CodecKind::Passthrough;
  fallback_.bits = 16;
  fallback_.lossy = false;
  fallback_.compression_ratio = 1.0f;

  // ARGUS's six built-in tiers. These are defaults, not assumptions: the
  // manager reads every format decision from whatever is registered here, so
  // Python can replace or drop any of them (including one_bit) without the
  // manager needing to know.
  TierCodec fp8;
  fp8.name = "fp8";
  fp8.kind = CodecKind::SignedLinear;
  fp8.bits = 8;
  fp8.lossy = true;
  fp8.compression_ratio = 0.5f;
  register_codec(fp8);

  TierCodec int8;
  int8.name = "int8";
  int8.kind = CodecKind::SignedLinear;
  int8.bits = 8;
  int8.lossy = true;
  int8.compression_ratio = 0.5f;
  register_codec(int8);

  TierCodec int4;
  int4.name = "int4";
  int4.kind = CodecKind::UnsignedAffine;
  int4.bits = 4;
  int4.lossy = true;
  int4.compression_ratio = 0.25f;
  register_codec(int4);

  TierCodec int2;
  int2.name = "int2";
  int2.kind = CodecKind::UnsignedAffine;
  int2.bits = 2;
  int2.lossy = true;
  int2.compression_ratio = 0.125f;
  register_codec(int2);

  TierCodec one_bit;
  one_bit.name = "one_bit";
  one_bit.kind = CodecKind::SignPacked;
  one_bit.bits = 1;
  one_bit.lossy = true;
  one_bit.compression_ratio = 0.0625f;
  register_codec(one_bit);

  TierCodec jl;
  jl.name = "jl";
  jl.kind = CodecKind::Projection;
  jl.bits = 16;
  jl.lossy = true;
  // Default JL rank is page_size/4; the manager recomputes the real ratio
  // from the projection matrix's actual shape when one is installed.
  jl.compression_ratio = 0.25f;
  register_codec(jl);
}

void CodecRegistry::register_codec(const TierCodec &codec) {
  codecs_[codec.name] = codec;
}

void CodecRegistry::unregister_codec(const std::string &name) {
  codecs_.erase(name);
}

bool CodecRegistry::has(const std::string &name) const {
  return codecs_.find(name) != codecs_.end();
}

const TierCodec &CodecRegistry::get(const std::string &name) const {
  auto it = codecs_.find(name);
  if (it != codecs_.end()) {
    return it->second;
  }
  return fallback_;
}

std::vector<std::string> CodecRegistry::names() const {
  std::vector<std::string> out;
  out.reserve(codecs_.size());
  for (const auto &kv : codecs_) {
    out.push_back(kv.first);
  }
  return out;
}

} // namespace argus
