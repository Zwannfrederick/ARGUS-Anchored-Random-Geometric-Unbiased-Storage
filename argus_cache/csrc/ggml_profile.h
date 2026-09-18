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
inline std::atomic<uint64_t> kernel_launches[3]{}, resident_calls[3]{}, resident_table_bytes[3]{};
inline thread_local Phase phase = other;

// Resident fast-path eligibility. Always counted: one reason per rejected invocation,
// the first failing check in evaluation order.
enum Reject { reject_q1, reject_forced_staged, reject_alignment, reject_gpu_table_budget, reject_staging_budget,
              reject_codec, reject_key_page, reject_value_page, reject_cold_budget, reject_count };
inline constexpr const char * reject_names[] = {"q1", "forced_staged", "alignment", "gpu_table_budget",
    "staging_budget", "codec", "nonresident_key_page", "nonresident_value_page", "cold_scratch_budget"};
// cold_pages: written non-GPU pages staged into scratch for accepted invocations.
inline std::atomic<uint64_t> resident_accepted[3]{}, resident_rejected[3][reject_count]{}, resident_cold_pages[3]{};
// Accepted invocations that ran a lane-per-cell kernel (default, cells-mlp or cells).
inline std::atomic<uint64_t> resident_cell_kernel[3]{};
inline void reject(Reject reason) { ++resident_rejected[phase][reason]; }

// Residency census at eligibility time (profiling only; walks every K/V view page).
// Cold = written but not GPU-resident. Bins: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, >=64.
enum PageState { page_gpu, page_pinned, page_ram, page_disk, page_unwritten, page_state_count };
inline constexpr const char * page_state_names[] = {"gpu", "pinned", "ram_pageable", "disk", "unwritten"};
inline constexpr int bins = 8;
inline constexpr const char * bin_names[] = {"0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+"};
inline constexpr int percent_bins = 6;
inline constexpr const char * percent_names[] = {"lt50", "50-90", "90-95", "95-99", "99-lt100", "100"};
struct Census {
    std::atomic<uint64_t> invocations{0}, cold_key_invocations{0}, cold_value_invocations{0};
    std::atomic<uint64_t> cold_runs{0}, transitions{0}, max_cold_pages{0};
    std::atomic<uint64_t> pages[page_state_count]{}, run_length[bins]{}, frontier_distance[bins]{};
    std::atomic<uint64_t> cold_access[bins]{}, resident_percent[percent_bins]{};
};
inline Census census[3];
inline int bin(uint64_t value) {
    if (!value) { return 0; }
    const int width = 64 - __builtin_clzll(value);
    return width < bins - 1 ? width : bins - 1;
}
inline int percent_bin(uint64_t gpu, uint64_t written) {
    if (!written || gpu == written) { return 5; }
    const double percent = 100.0 * double(gpu) / double(written);
    return percent < 50 ? 0 : percent < 90 ? 1 : percent < 95 ? 2 : percent < 99 ? 3 : 4;
}
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
        result += prefix + "kernel_launches\":" + std::to_string(kernel_launches[p].load());
        result += prefix + "resident_calls\":" + std::to_string(resident_calls[p].load());
        result += prefix + "resident_table_bytes\":" + std::to_string(resident_table_bytes[p].load());
    }
    return result;
}
inline std::string residency_json() {
    std::string result;
    const auto field = [&](const std::string & name, uint64_t value) { result += ",\"" + name + "\":" + std::to_string(value); };
    for (int p = 1; p < 3; ++p) {
        const std::string prefix = std::string("resident_") + phases[p] + "_";
        field(prefix + "accepted", resident_accepted[p]);
        field(prefix + "cold_pages", resident_cold_pages[p]);
        field(prefix + "cell_kernel", resident_cell_kernel[p]);
        for (int r = 0; r < reject_count; ++r) { field(prefix + "reject_" + reject_names[r], resident_rejected[p][r]); }
        if (!enabled()) { continue; }
        const auto & c = census[p];
        const std::string at = std::string("census_") + phases[p] + "_";
        field(at + "invocations", c.invocations);
        field(at + "cold_key_invocations", c.cold_key_invocations);
        field(at + "cold_value_invocations", c.cold_value_invocations);
        field(at + "cold_runs", c.cold_runs);
        field(at + "transitions", c.transitions);
        field(at + "max_cold_pages", c.max_cold_pages);
        for (int s = 0; s < page_state_count; ++s) { field(at + "pages_" + page_state_names[s], c.pages[s]); }
        for (int b = 0; b < bins; ++b) {
            field(at + "run_length_" + bin_names[b], c.run_length[b]);
            field(at + "frontier_distance_" + bin_names[b], c.frontier_distance[b]);
            field(at + "cold_access_count_" + bin_names[b], c.cold_access[b]);
        }
        for (int b = 0; b < percent_bins; ++b) { field(at + "resident_percent_" + percent_names[b], c.resident_percent[b]); }
    }
    return result;
}
} // namespace argus_profile
