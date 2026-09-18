#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

// Diagnostic only. CPU scopes are inclusive; CUDA event time overlaps CPU waits.
// Never add these counters together as if they were an exclusive wall-time partition.
namespace argus_profile {
enum Metric {
    attention, set_rows, write_page, disk_read, disk_write, checksum,
    page_lookup, descriptor_scan, staging, copy_enqueue, synchronization,
    allocation, release, policy, kernel_gpu, h2d_gpu, d2d_gpu, tier_read, tier_write,
    metric_count
};
inline constexpr const char * names[] = {
    "attention", "set_rows", "write_page", "disk_read", "disk_write", "checksum",
    "page_lookup", "descriptor_scan", "staging", "copy_enqueue", "synchronization",
    "allocation", "release", "policy", "kernel_gpu", "h2d_gpu", "d2d_gpu", "tier_read", "tier_write"
};
enum Phase { other, prefill, decode };
inline constexpr const char * phases[] = {"other", "prefill", "decode"};
inline std::atomic<uint64_t> nanoseconds[3][metric_count]{}, calls[3][metric_count]{};
inline std::atomic<uint64_t> exclusive_nanoseconds[3][metric_count]{};
inline std::atomic<uint64_t> h2d_bytes[3]{}, d2d_bytes[3]{}, disk_read_bytes[3]{}, disk_write_bytes[3]{};
inline thread_local Phase phase = other;
inline bool enabled() {
    static const bool value = [] { const char * p = std::getenv("ARGUS_KV_PROFILE"); return p && (std::strcmp(p, "1") == 0 || std::strcmp(p, "cpu") == 0); }();
    return value;
}
inline bool cuda_events() {
    static const bool value = [] { const char * p = std::getenv("ARGUS_KV_PROFILE"); return p && std::strcmp(p, "1") == 0; }();
    return value;
}
inline void add(Metric metric, uint64_t ns) {
    if (enabled()) { nanoseconds[phase][metric].fetch_add(ns, std::memory_order_relaxed); ++calls[phase][metric]; }
}
struct Scope;
inline thread_local Scope * active_scope = nullptr;
struct Scope {
    Metric metric;
    std::chrono::steady_clock::time_point start;
    Scope * parent = nullptr;
    uint64_t children_ns = 0;
    explicit Scope(Metric value) : metric(value) {
        if (enabled()) { parent = active_scope; active_scope = this; start = std::chrono::steady_clock::now(); }
    }
    ~Scope() {
        if (!enabled()) { return; }
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        add(metric, ns);
        exclusive_nanoseconds[phase][metric] += static_cast<uint64_t>(ns) - children_ns;
        active_scope = parent;
        if (parent) { parent->children_ns += ns; }
    }
};
struct InPhase {
    Phase previous = phase;
    explicit InPhase(bool prompt) { phase = prompt ? prefill : decode; }
    ~InPhase() { phase = previous; }
};
inline std::string json_fields() {
    std::string result;
    if (!enabled()) { return result; }
    for (int p = 0; p < 3; ++p) {
        const std::string prefix = ",\"profile_" + std::string(phases[p]) + "_";
        for (int m = 0; m < metric_count; ++m) {
            result += prefix + names[m] + "_ns\":" + std::to_string(nanoseconds[p][m].load());
            result += prefix + names[m] + "_calls\":" + std::to_string(calls[p][m].load());
            result += prefix + names[m] + "_exclusive_ns\":" + std::to_string(exclusive_nanoseconds[p][m].load());
        }
        result += prefix + "h2d_bytes\":" + std::to_string(h2d_bytes[p].load());
        result += prefix + "d2d_bytes\":" + std::to_string(d2d_bytes[p].load());
        result += prefix + "disk_read_bytes\":" + std::to_string(disk_read_bytes[p].load());
        result += prefix + "disk_write_bytes\":" + std::to_string(disk_write_bytes[p].load());
    }
    return result;
}
} // namespace argus_profile
