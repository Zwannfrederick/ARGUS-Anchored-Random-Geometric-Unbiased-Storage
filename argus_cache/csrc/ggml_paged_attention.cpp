// Exact causal attention over ARGUS-owned GGML KV, one cell block at a time.
// Blocks are merged with log-sum-exp, so the result is full softmax attention;
// only residency changes: cold blocks return to storage after they are read.
#include "ggml_host_buffer.h"
#include "ggml_disk_buffer.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
int64_t block_cells() {
    const char * text = std::getenv("ARGUS_KV_BLOCK_CELLS");
    if (!text) { return 256; }
    int64_t value = 0;
    const char * end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value <= 0) {
        throw std::runtime_error("ARGUS_KV_BLOCK_CELLS must be a positive integer");
    }
    return value;
}

float mask_value(const ggml_tensor * mask, int64_t cell, int64_t token) {
    const char * address = static_cast<char *>(mask->data) + cell * mask->nb[0] + token * mask->nb[1];
    return mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(address))
                                       : *reinterpret_cast<const float *>(address);
}

// Cold blocks are returned to storage only once every worker has moved past them,
// so a fast thread never evicts a block a slower one is about to read.
struct Coordination {
    static constexpr int max_threads = GGML_MAX_N_THREADS;
    std::atomic<int64_t> evicted{0};
    std::atomic<int64_t> progress[max_threads];
    int arrivals = 0; // Protected by coordination_for's mutex.
    Coordination() { for (auto & cell : progress) { cell.store(0); } }
};

std::shared_ptr<Coordination> coordination_for(const void * operation, int nth) {
    // GGML invokes every worker once per operation and joins them before reuse.
    // Remove the registry entry once all workers own it; the last worker frees it.
    static std::mutex lock;
    static std::unordered_map<const void *, std::shared_ptr<Coordination>> states;
    std::lock_guard<std::mutex> guard(lock);
    auto & entry = states[operation];
    if (!entry) { entry = std::make_shared<Coordination>(); }
    auto state = entry;
    if (++state->arrivals == nth) { states.erase(operation); }
    return state;
}

const char * at(const ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2) {
    return static_cast<const char *>(tensor->data) + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2];
}

// One op invocation as seen by one worker: tensor geometry plus this worker's rows.
struct Pass {
    const ggml_tensor * q, * k, * v, * mask;
    ggml_tensor * dst;
    float scale;
    int64_t dk, dv, n_head_kv, n_kv, n_head, n_tokens, row0, row1;
    const ggml_type_traits_cpu * k_cpu;
    size_t query_row;

    bool visible(int64_t cell, int64_t t0, int64_t t1) const {
        for (int64_t token = t0; token < t1; ++token) {
            if (mask_value(mask, cell, token) != -std::numeric_limits<float>::infinity()) { return true; }
        }
        return false;
    }
    float * out(int64_t row) const {
        return reinterpret_cast<float *>(const_cast<char *>(at(dst, 0, row % n_head, row / n_head)));
    }
};

// Queries in K's vec_dot representation, as GGML's own attention kernels use them.
void prepare_rows(const Pass & p, std::pmr::vector<char> & queries) {
    const auto * query_cpu = ggml_get_type_traits_cpu(p.k_cpu->vec_dot_type);
    queries.resize(static_cast<size_t>(p.row1 - p.row0) * p.query_row);
    for (int64_t row = p.row0; row < p.row1; ++row) {
        const auto * source = reinterpret_cast<const float *>(at(p.q, 0, row % p.n_head, row / p.n_head));
        auto * target = queries.data() + (row - p.row0) * p.query_row;
        if (p.k_cpu->vec_dot_type == GGML_TYPE_F32 || !query_cpu->from_float) {
            std::copy_n(reinterpret_cast<const char *>(source), p.query_row, target);
        } else {
            query_cpu->from_float(source, target, p.dk);
        }
        std::fill_n(p.out(row), p.dv, 0.0f);
    }
}

struct Scratch {
    std::pmr::vector<char> queries, keys, encoded_values, next_keys, next_values;
    std::pmr::vector<double> lse, mixed;
    std::pmr::vector<float> logits, values, partials, work;
    explicit Scratch(std::pmr::memory_resource * resource = std::pmr::get_default_resource())
        : queries(resource), keys(resource), encoded_values(resource), next_keys(resource), next_values(resource), lse(resource), mixed(resource),
          logits(resource), values(resource), partials(resource), work(resource) {}
};

bool stock_accumulator(const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask) {
    return mask && mask->type == GGML_TYPE_F16 && v->type != GGML_TYPE_F16 &&
           (ggml_is_quantized(k->type) || ggml_is_quantized(v->type));
}

struct DiskScratchSize { int64_t block; size_t bytes; };

DiskScratchSize disk_scratch_size(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                                 const ggml_tensor * mask) {
    const size_t query_bytes = ggml_row_size(ggml_get_type_traits_cpu(k->type)->vec_dot_type, k->ne[0]);
    const size_t rows = static_cast<size_t>(q->ne[1] * q->ne[2]);
    // PMR vector alignment plus a direct-I/O page are included before choosing block size.
    size_t fixed = rows * (query_bytes + sizeof(double)) + v->ne[0] * sizeof(double) + 256;
    if (stock_accumulator(k, v, mask)) {
        fixed += (rows * (v->ne[0] + 2) + k->ne[0] + 2 * v->ne[0] + 64) * sizeof(float);
    }
    const size_t per_cell = 2 * (k->nb[2] + v->nb[2]) + k->ne[1] * v->ne[0] * sizeof(float) + sizeof(float);
    const size_t maximum = argus_disk_staging_limit() / 4096 * 4096;
    const size_t overhead = 4096 + ArgusDiskPrefetch::stack_bytes;
    if (maximum <= overhead || fixed >= maximum - overhead || per_cell > maximum - overhead - fixed) {
        throw std::runtime_error("ARGUS staging budget cannot hold one attention block; reduce ubatch or increase staging");
    }
    const auto block = std::min({block_cells(), k->ne[2], static_cast<int64_t>((maximum - overhead - fixed) / per_cell)});
    return {block, fixed + static_cast<size_t>(block) * per_cell};
}

void attend_stock_block(const Pass & p, int64_t c0, int64_t c1, Scratch & scratch,
                        const char * keys, const char * values, bool final_block) {
    ggml_tensor q = *p.q, k = *p.k, v = *p.v, mask = *p.mask, out = *p.dst;
    k.data = const_cast<char *>(keys ? keys : at(p.k, 0, 0, c0));
    v.data = const_cast<char *>(values ? values : at(p.v, 0, 0, c0));
    k.ne[2] = v.ne[2] = c1 - c0;
    for (auto * tensor : {&q, &k, &v}) {
        std::swap(tensor->ne[1], tensor->ne[2]);
        std::swap(tensor->nb[1], tensor->nb[2]);
    }
    mask.data = static_cast<char *>(mask.data) + c0 * mask.nb[0];
    mask.ne[0] = c1 - c0;
    std::memset(out.op_params, 0, sizeof(out.op_params));
    std::memcpy(out.op_params, &p.scale, sizeof(p.scale));
    out.src[0] = &q; out.src[1] = &k; out.src[2] = &v; out.src[3] = &mask; out.src[4] = nullptr;
    ggml_cpu_flash_attn_ext_accumulate(&out, scratch.partials.data(), scratch.work.data(), final_block);
}

// Softmax over one block, merged into each row's running output by log-sum-exp.
void attend_block(const Pass & p, int64_t c0, int64_t c1, Scratch & s,
                  const char * keys = nullptr, const char * values = nullptr) {
    const auto v_to_float = ggml_get_type_traits(p.v->type)->to_float;
    for (int64_t cell = c0; cell < c1; ++cell) {
        for (int64_t h = 0; h < p.n_head_kv; ++h) {
            const char * source = values ? values + (cell - c0) * p.v->nb[2] + h * p.v->nb[1] : at(p.v, 0, h, cell);
            float * target = s.values.data() + ((cell - c0) * p.n_head_kv + h) * p.dv;
            if (p.v->type == GGML_TYPE_F32) {
                std::copy_n(reinterpret_cast<const float *>(source), p.dv, target);
            } else {
                v_to_float(source, target, p.dv);
            }
        }
    }
    const float minus_inf = -std::numeric_limits<float>::infinity();
    for (int64_t row = p.row0; row < p.row1; ++row) {
        const int64_t token = row / p.n_head, head_kv = (row % p.n_head) * p.n_head_kv / p.n_head;
        const char * query = s.queries.data() + (row - p.row0) * p.query_row;
        float best = minus_inf;
        for (int64_t cell = c0; cell < c1; ++cell) {
            const float bias = mask_value(p.mask, cell, token);
            float score = minus_inf;
            if (bias != minus_inf) {
                const char * key = keys ? keys + (cell - c0) * p.k->nb[2] + head_kv * p.k->nb[1] : at(p.k, 0, head_kv, cell);
                p.k_cpu->vec_dot(static_cast<int>(p.dk), &score, 0, key, 0, query, 0, 1);
                score = score * p.scale + bias;
            }
            s.logits[cell - c0] = score;
            best = std::max(best, score);
        }
        if (best == minus_inf) { continue; }
        double total = 0.0;
        for (int64_t cell = c0; cell < c1; ++cell) { total += std::exp(double(s.logits[cell - c0]) - best); }
        const double block_lse = best + std::log(total);
        std::fill(s.mixed.begin(), s.mixed.end(), 0.0);
        for (int64_t cell = c0; cell < c1; ++cell) {
            const double weight = std::exp(double(s.logits[cell - c0]) - block_lse);
            if (weight == 0.0) { continue; }
            const float * value = s.values.data() + ((cell - c0) * p.n_head_kv + head_kv) * p.dv;
            for (int64_t i = 0; i < p.dv; ++i) { s.mixed[i] += weight * value[i]; }
        }
        float * out = p.out(row);
        double & running = s.lse[row - p.row0];
        const double merged = running == -std::numeric_limits<double>::infinity()
            ? block_lse : std::max(running, block_lse) + std::log1p(std::exp(-std::abs(running - block_lse)));
        const double keep = std::exp(running - merged), add = std::exp(block_lse - merged);
        for (int64_t i = 0; i < p.dv; ++i) { out[i] = static_cast<float>(keep * out[i] + add * s.mixed[i]); }
        running = merged;
    }
}

// Cold prefix [0, cold_end) is released once the slowest worker has moved past it.
// Cells hidden by a crop or sliding window are part of that prefix too.
class Evictor {
  public:
    static constexpr int64_t done = std::numeric_limits<int64_t>::max();

    Evictor(const Pass & pass, Coordination & coordination, int ith, int nth, int64_t cold_end, bool paging)
        : p_(pass), coordination_(coordination), ith_(ith), nth_(nth), cold_end_(cold_end), paging_(paging) {}

    void advance(int64_t cell) {
        if (!paging_) { return; }
        coordination_.progress[ith_].store(cell);
        int64_t floor = cold_end_;
        for (int j = 0; j < nth_; ++j) { floor = std::min(floor, coordination_.progress[j].load()); }
        int64_t evicted = coordination_.evicted.load();
        while (floor > evicted && !coordination_.evicted.compare_exchange_weak(evicted, floor)) {}
        // The claimed range continues the cold prefix, so bytes before it are cold too.
        if (floor > evicted) { release(evicted, floor, evicted > 0); }
    }

    void release(int64_t start, int64_t end, bool cold_before) {
        if (end <= start) { return; }
        argus_kv_page_out(at(p_.k, 0, 0, start), (end - start) * p_.k->nb[2], cold_before);
        argus_kv_page_out(at(p_.v, 0, 0, start), (end - start) * p_.v->nb[2], cold_before);
        paged_out += (end - start) * (p_.k->nb[2] + p_.v->nb[2]);
    }

    size_t paged_out = 0;

  private:
    const Pass & p_;
    Coordination & coordination_;
    int ith_, nth_;
    int64_t cold_end_;
    bool paging_;
};

void attend(ggml_tensor * dst, int ith, int nth, void * userdata) try {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    const bool disk = argus_ggml_is_disk_tensor(k);
    const bool stock = stock_accumulator(k, v, dst->src[3]);
    // GGML invokes custom ops on every graph worker despite the n_tasks hint.
    if (disk || stock) {
        if (ith != 0) { return; }
        nth = 1;
    }
    const int64_t rows = q->ne[1] * q->ne[2];
    const auto * k_cpu = ggml_get_type_traits_cpu(k->type);
    const Pass p{q, k, v, dst->src[3], dst, *static_cast<const float *>(userdata),
                 k->ne[0], v->ne[0], k->ne[1], k->ne[2], q->ne[1], q->ne[2],
                 rows * ith / nth, rows * (ith + 1) / nth, k_cpu, ggml_row_size(k_cpu->vec_dot_type, k->ne[0])};
    const bool paging = !disk && argus_kv_resident_budget() > 0 && nth <= Coordination::max_threads;
    auto coordination = coordination_for(dst, nth);
    if (paging) {
        coordination->progress[ith].store(p.row0 < p.row1 ? 0 : Evictor::done);
    }
    if (p.row0 >= p.row1) { return; }

    thread_local Scratch host_scratch;
    std::optional<ArgusStagingBuffer> arena;
    std::optional<std::pmr::monotonic_buffer_resource> resource;
    std::optional<Scratch> disk_scratch;
    std::optional<ArgusDiskPrefetch> prefetch;
    int64_t block = std::min(block_cells(), p.n_kv);
    if (disk) {
        const auto size = disk_scratch_size(q, k, v, p.mask);
        block = size.block;
        arena.emplace(size.bytes);
        resource.emplace(arena->data(), arena->size(), std::pmr::null_memory_resource());
        disk_scratch.emplace(&*resource);
    }
    Scratch & scratch = disk ? *disk_scratch : host_scratch;
    prepare_rows(p, scratch.queries);
    scratch.lse.assign(static_cast<size_t>(p.row1 - p.row0), -std::numeric_limits<double>::infinity());
    scratch.logits.resize(static_cast<size_t>(block));
    scratch.values.resize(static_cast<size_t>(block * p.n_head_kv * p.dv));
    scratch.mixed.resize(static_cast<size_t>(p.dv));
    if (stock) {
        scratch.partials.assign(static_cast<size_t>(rows * (p.dv + 2)), 0.0f);
        for (int64_t row = 0; row < rows; ++row) {
            scratch.partials[row * (p.dv + 2)] = -std::numeric_limits<float>::infinity();
        }
        scratch.work.resize(static_cast<size_t>(p.dk + 2 * p.dv + 64));
    }
    if (disk) {
        scratch.keys.resize(static_cast<size_t>(block) * k->nb[2]);
        scratch.encoded_values.resize(static_cast<size_t>(block) * v->nb[2]);
        scratch.next_keys.resize(scratch.keys.size());
        scratch.next_values.resize(scratch.encoded_values.size());
    }

    // GGML pads n_kv; cells past the last visible one are never read.
    int64_t used_end = p.n_kv;
    while (used_end > 0 && !p.visible(used_end - 1, 0, p.n_tokens)) { --used_end; }
    const int64_t kv_size = k->view_src ? k->view_src->ne[1] : p.n_kv;
    const int64_t hot_cells = std::max<int64_t>(0, static_cast<int64_t>(kv_size * argus_kv_hot_fraction()) - block);
    Evictor evictor(p, *coordination, ith, nth, used_end - hot_cells, paging);

    size_t read = 0;
    const int64_t token0 = p.row0 / p.n_head, token1 = (p.row1 - 1) / p.n_head + 1;
    const auto next_visible = [&](int64_t start) {
        for (; start < used_end; start += block) {
            const auto end = std::min(start + block, used_end);
            for (int64_t cell = start; cell < end; ++cell) {
                if (p.visible(cell, token0, token1)) { return start; }
            }
        }
        return used_end;
    };
    if (disk && used_end) {
        prefetch.emplace();
        const auto first = next_visible(0);
        prefetch->submit(k, v, first, std::min(first + block, used_end) - first,
                         scratch.keys.data(), scratch.encoded_values.data());
    }
    for (int64_t c0 = 0; c0 < used_end; c0 += block) {
        const int64_t c1 = std::min(c0 + block, used_end);
        bool any_visible = false;
        for (int64_t cell = c0; cell < c1 && !any_visible; ++cell) { any_visible = p.visible(cell, token0, token1); }
        if (any_visible) {
            if (disk) {
                prefetch->take();
                const auto next = next_visible(c1);
                if (next < used_end) {
                    prefetch->submit(k, v, next, std::min(next + block, used_end) - next,
                                     scratch.next_keys.data(), scratch.next_values.data());
                }
                if (stock) {
                    attend_stock_block(p, c0, c1, scratch, scratch.keys.data(), scratch.encoded_values.data(), c1 == used_end);
                } else {
                    attend_block(p, c0, c1, scratch, scratch.keys.data(), scratch.encoded_values.data());
                }
                scratch.keys.swap(scratch.next_keys);
                scratch.encoded_values.swap(scratch.next_values);
            } else {
                // Request this block and the next; OS advice does not impose a resident limit.
                const int64_t p1 = std::min(c1 + block, used_end);
                argus_kv_prefetch(at(k, 0, 0, c0), (p1 - c0) * k->nb[2]);
                argus_kv_prefetch(at(v, 0, 0, c0), (p1 - c0) * v->nb[2]);
                if (stock) { attend_stock_block(p, c0, c1, scratch, nullptr, nullptr, c1 == used_end); }
                else { attend_block(p, c0, c1, scratch); }
            }
            read += (c1 - c0) * (k->nb[2] + v->nb[2]);
        }
        evictor.advance(c1);
    }
    evictor.advance(Evictor::done);
    // Past the last visible cell; the page before it may still be hot.
    if (paging && ith == 0) { evictor.release(used_end, p.n_kv, false); }
    if (disk) { argus_disk_publish_stats(); }
    else {
        argus_kv_note_attention(read, evictor.paged_out);
        if (ith == 0) { argus_kv_publish_stats(); }
    }
} catch (const std::exception & error) {
    GGML_ABORT("ARGUS attention failed: %s", error.what());
}

const float * intern_scale(float scale) {
    static std::mutex lock;
    static std::deque<float> values;
    std::lock_guard<std::mutex> guard(lock);
    for (const auto & value : values) {
        if (value == scale) { return &value; }
    }
    values.push_back(scale);
    return &values.back();
}
} // namespace

ggml_tensor * argus_ggml_paged_attention(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * mask, bool unsupported_features, float scale) {
    const ggml_tensor * storage = k->view_src ? k->view_src : k;
    const bool disk = argus_ggml_is_disk_tensor(k);
    if (!disk && (!argus_kv_resident_budget() || !storage->buffer ||
        ggml_backend_buffer_get_type(storage->buffer) != argus_ggml_host_buffer_type())) {
        return nullptr;
    }
    // Validate before entering GGML worker threads, where exceptions cannot unwind safely.
    block_cells();
    const bool v_transposed = v->nb[1] > v->nb[2];
    if (unsupported_features || v_transposed || !mask || k->ne[3] != 1 || q->ne[3] != 1 ||
        q->type != GGML_TYPE_F32 || (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) ||
        mask->ne[0] < k->ne[2] || mask->ne[1] < q->ne[2] || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        q->ne[0] != k->ne[0] || k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2] || q->ne[1] % k->ne[1] != 0 ||
        k->ne[0] % ggml_blck_size(k->type) != 0 || v->ne[0] % ggml_blck_size(v->type) != 0 ||
        !ggml_get_type_traits_cpu(k->type)->vec_dot ||
        (v->type != GGML_TYPE_F32 && !ggml_get_type_traits(v->type)->to_float)) {
        throw std::runtime_error(
            "ARGUS paged KV supports single-stream causal attention with flash attention enabled "
            "(-fa on), no sinks, ALiBi, soft-capping or KQ bias");
    }
    if (disk) {
        if (!argus_ggml_is_disk_tensor(v)) { throw std::runtime_error("ARGUS disk K requires disk V"); }
        disk_scratch_size(q, k, v, mask);
    }
    ggml_tensor * args[] = {q, k, v, mask};
    // ponytail: one disk attention worker; parallel workers require shared bounded blocks.
    return ggml_custom_4d(ctx, GGML_TYPE_F32, v->ne[0], q->ne[1], q->ne[2], 1, args, 4,
                          attend, (disk || stock_accumulator(k, v, mask)) ? 1 : GGML_N_TASKS_MAX,
                          const_cast<float *>(intern_scale(scale)));
}
