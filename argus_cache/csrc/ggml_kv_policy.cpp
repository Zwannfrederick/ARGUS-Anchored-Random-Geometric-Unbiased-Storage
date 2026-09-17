#include "ggml_kv_policy.h"
#include "ggml_cuda_attention.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

namespace {
constexpr size_t page_bytes = 4096;
std::atomic<uint64_t> promotions{0}, demotions{0}, rejected{0};

size_t available(ArgusTier tier) {
    const auto budget = argus_tier_budget(tier);
    return budget.live < budget.limit ? budget.limit - budget.live : 0;
}

struct Victim {
    ArgusTier tier;
    ArgusDiskPageDescriptor page{};
    bool found = false;
    bool scanned = false;
};

void choose_victim(const ArgusDiskPageDescriptor & page, void * context) {
    auto & choice = *static_cast<Victim *>(context);
    if (page.placement != choice.tier) { return; }
    if (!choice.found || page.access_count < choice.page.access_count) {
        choice.page = page;
        choice.found = true;
    }
}

bool move(const ArgusDiskPageDescriptor & page, ArgusTier tier) {
    try {
        argus_disk_move_page(page.revision, tier);
    } catch (const std::exception &) {
        // Budget races, stale content and failed verification leave the source intact.
        ++rejected;
        return false;
    }
    if (tier == ArgusTier::disk) { ++demotions; }
    else { ++promotions; }
    return true;
}

bool evict(Victim & victim, uint64_t ceiling, bool force = false) {
    // ponytail: O(total pages) per eviction; add an indexed queue if profiling warrants it.
    // Retain a negative decision through this pass instead of rescanning for every page.
    if (!victim.scanned) {
        victim.found = false;
        argus_disk_visit_resident_pages(choose_victim, &victim);
        victim.scanned = true;
    }
    if (!victim.found || (!force && victim.page.access_count >= ceiling)) { return false; }
    victim.scanned = false;
    return move(victim.page, ArgusTier::disk);
}
} // namespace

bool argus_kv_policy_enabled() {
    const char * mode = std::getenv("ARGUS_KV_POLICY");
    if (!mode || std::strcmp(mode, "off") == 0) { return false; }
    if (std::strcmp(mode, "on") == 0) { return true; }
    throw std::invalid_argument("ARGUS_KV_POLICY must be off or on");
}

void argus_kv_policy_prepare(size_t gpu_bytes, size_t pinned_bytes) {
    if (!argus_kv_policy_enabled()) { return; }
    for (auto tier : {ArgusTier::gpu, ArgusTier::pinned}) {
        const size_t needed = tier == ArgusTier::gpu ? gpu_bytes : pinned_bytes;
        if (needed > argus_tier_budget(tier).limit) {
            throw std::runtime_error("ARGUS tier budget cannot hold attention scratch");
        }
        Victim victim{tier};
        while (available(tier) < needed) {
            if (!evict(victim, UINT64_MAX, true)) { break; }
        }
    }
}

void argus_kv_policy_observe(const ggml_tensor * tensor) {
    if (!argus_kv_policy_enabled()) { return; }
    if (!argus_ggml_is_disk_tensor(tensor)) { throw std::invalid_argument("ARGUS policy requires disk tensor"); }
    // Full physical pages only; a view may begin/end inside a page.
    const size_t within = reinterpret_cast<uintptr_t>(tensor->data) % page_bytes;
    const size_t first = within ? page_bytes - within : 0;
    const size_t bytes = ggml_nbytes(tensor);
    std::array<Victim, 3> victims{{{ArgusTier::ram}, {ArgusTier::pinned}, {ArgusTier::gpu}}};
    for (size_t offset = first; offset <= bytes && page_bytes <= bytes - offset; offset += page_bytes) {
        const auto page = argus_disk_page_descriptor(tensor, offset);
        // Two reads admit reuse while avoiding promotion of untouched/one-shot pages.
        if (!page.written || page.codec == GGML_TYPE_COUNT || page.access_count < 2) { continue; }
        for (auto tier : {ArgusTier::gpu, ArgusTier::pinned, ArgusTier::ram}) {
            if (page.placement == tier) { break; }
            if (argus_tier_budget(tier).limit < page_bytes) { continue; }
            auto & victim = victims[static_cast<size_t>(tier) - 1];
            if (available(tier) < page_bytes && !evict(victim, page.access_count)) { continue; }
            if (move(page, tier)) { victim.scanned = false; break; }
        }
    }
}

ArgusPolicyStats argus_kv_policy_stats() {
    return {promotions.load(), demotions.load(), rejected.load()};
}
