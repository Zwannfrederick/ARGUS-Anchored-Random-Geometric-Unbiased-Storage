#pragma once
#include "ggml_disk_buffer.h"

struct ArgusPolicyStats {
    uint64_t promotions, demotions, rejected, nanoseconds;
};
// Missing/off selects the reference path. Unknown values are errors.
bool argus_kv_policy_enabled();
// Call before allocating attention scratch; observe after the scratch is released.
void argus_kv_policy_prepare(size_t gpu_bytes, size_t pinned_bytes);
// Scratch the last prepare() reserved in a tier. Neither promotion nor appends fill it.
size_t argus_kv_policy_headroom(ArgusTier tier);
void argus_kv_policy_observe(const ggml_tensor * tensor);
ArgusPolicyStats argus_kv_policy_stats();
