#pragma once
#include "ggml_disk_buffer.h"

struct ArgusPolicyStats {
    uint64_t promotions, demotions, rejected, nanoseconds;
};
// Missing/off selects the reference path. Unknown values are errors.
bool argus_kv_policy_enabled();
// Call before allocating attention scratch, then observe while scratch is still held.
void argus_kv_policy_prepare(size_t gpu_bytes, size_t pinned_bytes);
void argus_kv_policy_observe(const ggml_tensor * tensor);
ArgusPolicyStats argus_kv_policy_stats();
