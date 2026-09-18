#include "ggml_cuda_attention.h"
#include "ggml_kv_policy.h"
#include "ggml_profile.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string("ARGUS CUDA: ") + cudaGetErrorString(status)); }
}
size_t aligned(size_t bytes) {
    if (!bytes || bytes > SIZE_MAX - 4095) { throw std::invalid_argument("ARGUS invalid allocation size"); }
    return (bytes + 4095) / 4096 * 4096;
}
size_t setting(const char * name) {
    const char * text = std::getenv(name);
    if (!text || !*text) { throw std::invalid_argument(std::string(name) + " is required"); }
    size_t value = 0;
    const auto result = std::from_chars(text, text + std::strlen(text), value);
    if (result.ec != std::errc{} || *result.ptr || !value) { throw std::invalid_argument(std::string(name) + " must be positive bytes"); }
    return value;
}
std::mutex memory_mutex;
size_t live[4]{}, peak[4]{};
std::atomic<size_t> calls{0};
const char * budget_name(ArgusTier tier) {
    switch (tier) {
        case ArgusTier::gpu: return "ARGUS_KV_GPU_BYTES";
        case ArgusTier::pinned: return "ARGUS_KV_PINNED_BYTES";
        case ArgusTier::ram: return "ARGUS_KV_RAM_BYTES";
        default: throw std::invalid_argument("disk is not a resident allocation");
    }
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    // argus_disk_stage_cuda drains every transfer (also on failure) before it
    // releases source pages. There is no pending work requiring a second wait.
    ~Stream() { if (value) { cudaStreamDestroy(value); } }
    Stream(const Stream &) = delete;
    Stream & operator=(const Stream &) = delete;
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreateWithFlags(&value, argus_profile::cuda_events() ? cudaEventDefault : cudaEventDisableTiming)); }
    ~Event() { if (value) { cudaEventDestroy(value); } }
    Event(const Event &) = delete;
    Event & operator=(const Event &) = delete;
};
struct Drain {
    cudaStream_t value;
    bool pending = true;
    ~Drain() { if (pending) { argus_profile::Scope timer(argus_profile::synchronization); if (cudaStreamSynchronize(value) != cudaSuccess) { GGML_ABORT("ARGUS CUDA stream failed"); } } }
};

void elapsed(argus_profile::Metric metric, const Event & start, const Event & end) {
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start.value, end.value));
    argus_profile::add(metric, static_cast<uint64_t>(ms * 1000000.0));
}
struct CopyEvents { Event start, end; bool device_source = false; };
// Profiling metadata only: reusable events bounded by copies in one existing staging wait.
thread_local std::vector<std::unique_ptr<CopyEvents>> copy_events;
thread_local size_t pending_copies = 0;

struct ResidentKV {
    const void * const * keys;
    const void * const * values;
    size_t key_offset, value_offset;
};
__device__ float resident_value(const void * const * pages, size_t offset, size_t index) {
    const size_t position = offset + index * sizeof(half);
    const auto * page = static_cast<const char *>(pages[position / 4096]);
    return page ? __half2float(*reinterpret_cast<const half *>(page + position % 4096)) : 0.0f;
}
// Start of a row inside one resident page, or null for a logical-zero page.
__device__ const half * page_row(const void * const * pages, size_t position) {
    const auto * page = static_cast<const char *>(pages[position / 4096]);
    return page ? reinterpret_cast<const half *>(page + position % 4096) : nullptr;
}

// One block per query/head. Online softmax state survives tile boundaries.
// ponytail: scalar FP16 tiles first; replace with tiled MMA after mechanism parity.
template<bool resident>
__global__ void attention_tile(const char * query, const half * keys, const half * values,
        const char * mask, bool half_mask, float * output, float * state,
        int dk, int dv, int heads, int kv_heads, int tokens, int first, int count, bool last,
        size_t q_head, size_t q_token, size_t mask_cell, size_t mask_token, float scale, ResidentKV pages) {
    const int row = blockIdx.x, head = row % heads, token = row / heads, d = threadIdx.x;
    if (token >= tokens) { return; }
    const int kv_head = head / (heads / kv_heads);
    __shared__ float reduction[256];
    __shared__ float alpha, beta, maximum, sum;
    float acc = first ? (d < dv ? output[row * dv + d] : 0.0f) : 0.0f;
    if (d == 0) { maximum = first ? state[2 * row] : -INFINITY; sum = first ? state[2 * row + 1] : 0.0f; }
    __syncthreads();
    const auto * q = reinterpret_cast<const float *>(query + head * q_head + token * q_token);
    for (int cell = 0; cell < count; ++cell) {
        const char * entry = mask + (first + cell) * mask_cell + token * mask_token;
        const float bias = half_mask ? __half2float(*reinterpret_cast<const half *>(entry)) : *reinterpret_cast<const float *>(entry);
        if (bias == -INFINITY) { continue; }
        float key = 0.0f;
        if (d < dk) {
            if constexpr (resident) { key = resident_value(pages.keys, pages.key_offset, (size_t(first + cell) * kv_heads + kv_head) * dk + d); }
            else { key = __half2float(keys[(cell * kv_heads + kv_head) * dk + d]); }
        }
        reduction[d] = d < dk ? q[d] * key : 0.0f;
        __syncthreads();
        for (int width = 128; width; width /= 2) {
            if (d < width) { reduction[d] += reduction[d + width]; }
            __syncthreads();
        }
        if (d == 0) {
            const float score = __fmaf_rn(reduction[0], scale, bias);
            const float next = fmaxf(maximum, score);
            alpha = expf(maximum - next); beta = expf(score - next);
            sum = __fmaf_rn(alpha, sum, beta); maximum = next;
        }
        __syncthreads();
        if (d < dv) {
            float value;
            if constexpr (resident) { value = resident_value(pages.values, pages.value_offset, (size_t(first + cell) * kv_heads + kv_head) * dv + d); }
            else { value = __half2float(values[(cell * kv_heads + kv_head) * dv + d]); }
            acc = __fmaf_rn(alpha, acc, __fmul_rn(beta, value));
        }
        __syncthreads();
    }
    if (d < dv) { output[row * dv + d] = last ? (sum > 0 ? acc / sum : 0.0f) : acc; }
    if (d == 0) { state[2 * row] = maximum; state[2 * row + 1] = sum; }
}
void cpu_refused(ggml_tensor *, int, int, void *) { GGML_ABORT("ARGUS CUDA attention assigned to CPU"); }

// Four query/head warps per block, full context in one invocation. The staged
// 256-lane reduction tree is reproduced in registers, then warp shuffles: no
// tensor-core/F16 accumulation or reassociation of the FP32 dot product.
// D > 0 fixes Dk = Dv = D (a multiple of 32) at compile time; D = 0 is the generic
// Dk/Dv <= 256 kernel. Slots at or beyond D/32 hold the staged tree's structural
// +0 lanes; skipping those additions changes at most the sign of a zero sum.
// rows: every Dk-half K row and Dv-half V row starts at a multiple of its own size
// (a divisor of 4096), so it lies inside one page: resolve its page once per row.
template<int D, bool rows = false>
__global__ void attention_resident_batch(const char * query, const char * mask, bool half_mask,
        float * output, int dk, int dv, int heads, int kv_heads, int tokens, int cells,
        size_t q_head, size_t q_token, size_t mask_cell, size_t mask_token, float scale, ResidentKV pages) {
    constexpr int slots = D ? D / 32 : 8;
    if constexpr (D != 0) { dk = D; dv = D; }
    const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= heads * tokens) { return; }
    const int head = row % heads, token = row / heads, kv_head = head / (heads / kv_heads);
    const auto * q = reinterpret_cast<const float *>(query + head * q_head + token * q_token);
    float query_values[slots]{}, acc[slots]{};
    #pragma unroll
    for (int j = 0; j < slots; ++j) { if (lane + j * 32 < dk) { query_values[j] = q[lane + j * 32]; } }
    float maximum = -INFINITY, sum = 0.0f;
    for (int cell = 0; cell < cells; ++cell) {
        const char * entry = mask + size_t(cell) * mask_cell + token * mask_token;
        const float bias = half_mask ? __half2float(*reinterpret_cast<const half *>(entry)) : *reinterpret_cast<const float *>(entry);
        if (bias == -INFINITY) { continue; }
        float product[8]{};
        if constexpr (rows) {
            const half * key_row = page_row(pages.keys, pages.key_offset + (size_t(cell) * kv_heads + kv_head) * D * sizeof(half));
            #pragma unroll
            for (int j = 0; j < slots; ++j) {
                product[j] = __fmul_rn(query_values[j], key_row ? __half2float(__ldg(key_row + lane + j * 32)) : 0.0f);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < slots; ++j) {
                const int d = lane + j * 32;
                if (d < dk) { product[j] = __fmul_rn(query_values[j], resident_value(pages.keys, pages.key_offset, (size_t(cell) * kv_heads + kv_head) * dk + d)); }
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) { if (j + 4 < slots) { product[j] = __fadd_rn(product[j], product[j + 4]); } }
        #pragma unroll
        for (int j = 0; j < 2; ++j) { if (j + 2 < slots) { product[j] = __fadd_rn(product[j], product[j + 2]); } }
        float dot = slots > 1 ? __fadd_rn(product[0], product[1]) : product[0];
        #pragma unroll
        for (int width = 16; width; width /= 2) {
            const float other = __shfl_down_sync(0xffffffff, dot, width);
            if (lane < width) { dot = __fadd_rn(dot, other); }
        }
        float alpha = 0.0f, beta = 0.0f;
        if (lane == 0) {
            // Explicit forms of the contractions NVCC emits for the staged kernel.
            const float score = __fmaf_rn(dot, scale, bias);
            const float next = fmaxf(maximum, score);
            alpha = expf(maximum - next); beta = expf(score - next);
            sum = __fmaf_rn(alpha, sum, beta); maximum = next;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
        beta = __shfl_sync(0xffffffff, beta, 0);
        const half * value_row = nullptr;
        if constexpr (rows) { value_row = page_row(pages.values, pages.value_offset + (size_t(cell) * kv_heads + kv_head) * D * sizeof(half)); }
        #pragma unroll
        for (int j = 0; j < slots; ++j) {
            const int d = lane + j * 32;
            if (d < dv) {
                const float value = rows ? (value_row ? __half2float(__ldg(value_row + d)) : 0.0f)
                    : resident_value(pages.values, pages.value_offset, (size_t(cell) * kv_heads + kv_head) * dv + d);
                // NVCC otherwise contracts beta*value in the final unrolled lane,
                // unlike the staged kernel's alpha*acc FMA (D=256 exact-parity gate).
                acc[j] = __fmaf_rn(alpha, acc[j], __fmul_rn(beta, value));
            }
        }
    }
    sum = __shfl_sync(0xffffffff, sum, 0);
    #pragma unroll
    for (int j = 0; j < slots; ++j) {
        const int d = lane + j * 32;
        if (d < dv) { output[row * dv + d] = sum > 0 ? acc[j] / sum : 0.0f; }
    }
}

struct ResidentRequest {
    ggml_tensor * dst;
    cudaStream_t stream;
    const void ** table;
    size_t state_bytes;
    bool batched;
    size_t table_bytes = 0;
    // Written non-GPU pages copied for this invocation only (see ArgusColdStaging).
    std::unique_ptr<ArgusStagingBuffer> cold_host;
    std::unique_ptr<ArgusStagingReservation> cold_charge;
    std::unique_ptr<ArgusTierBuffer> cold_device;
    // Last member, destroyed first: an upload left queued by a failure drains before its buffers go.
    Drain cold_drain{nullptr, false};
};
void * reserve_cold(size_t pages, void * raw) {
    auto & request = *static_cast<ResidentRequest *>(raw);
    const size_t bytes = pages * 4096;
    // Host bounce and device copy, plus the table/state scratch resident_compute takes next.
    // Decline rather than evict: attention never acts as placement policy.
    const auto gpu = argus_tier_budget(ArgusTier::gpu);
    const size_t device = bytes + request.table_bytes + request.state_bytes;
    if (argus_disk_staging_free() < 2 * bytes + request.table_bytes + request.state_bytes ||
        gpu.live > gpu.limit || gpu.limit - gpu.live < device) { return nullptr; }
    request.cold_host = std::make_unique<ArgusStagingBuffer>(bytes);
    request.cold_charge = std::make_unique<ArgusStagingReservation>(bytes);
    request.cold_device = std::make_unique<ArgusTierBuffer>(ArgusTier::gpu, bytes);
    return request.cold_host->data();
}
const char * upload_cold(size_t pages, void * raw) {
    auto & request = *static_cast<ResidentRequest *>(raw);
    // Arm in place: a temporary Drain would synchronize when it is destroyed.
    request.cold_drain.value = request.stream;
    request.cold_drain.pending = true;
    argus_cuda_copy(request.cold_device->data(), request.cold_host->data(), pages * 4096, false, request.stream);
    return static_cast<const char *>(request.cold_device->data());
}
void resident_compute(size_t k_pages, size_t v_pages, void * raw) {
    auto & request = *static_cast<ResidentRequest *>(raw);
    auto * dst = request.dst;
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    const size_t bytes = (k_pages + v_pages) * sizeof(void *);
    ArgusStagingReservation scratch(aligned(bytes) + request.state_bytes);
    ArgusTierBuffer table(ArgusTier::gpu, bytes);
    std::unique_ptr<ArgusTierBuffer> state;
    if (request.state_bytes) { state = std::make_unique<ArgusTierBuffer>(ArgusTier::gpu, request.state_bytes); }
    Drain drain{request.stream};
    argus_cuda_copy(table.data(), request.table, bytes, false, request.stream);
    const auto * pointers = static_cast<const void * const *>(table.data());
    const ResidentKV pages{pointers, pointers + k_pages,
        reinterpret_cast<uintptr_t>(k->data) % 4096, reinterpret_cast<uintptr_t>(v->data) % 4096};
    float scale;
    std::memcpy(&scale, reinterpret_cast<const char *>(dst->op_params) + sizeof(ggml_custom_op_params), sizeof scale);
    std::unique_ptr<Event> begin, end;
    if (argus_profile::cuda_events()) {
        begin = std::make_unique<Event>(); end = std::make_unique<Event>();
        check(cudaEventRecord(begin->value, request.stream));
    }
    if (request.batched) {
        const bool d64 = q->ne[0] == 64 && v->ne[0] == 64;
        const bool rows = pages.key_offset % (64 * sizeof(half)) == 0 && pages.value_offset % (64 * sizeof(half)) == 0;
        const auto kernel = !d64 ? attention_resident_batch<0> : rows ? attention_resident_batch<64, true> : attention_resident_batch<64>;
        kernel<<<(q->ne[1] * q->ne[2] + 3) / 4, 128, 0, request.stream>>>(
            static_cast<const char *>(q->data), static_cast<const char *>(mask->data), mask->type == GGML_TYPE_F16,
            static_cast<float *>(dst->data), q->ne[0], v->ne[0], q->ne[1], k->ne[1], q->ne[2], k->ne[2],
            q->nb[1], q->nb[2], mask->nb[0], mask->nb[1], scale, pages);
        check(cudaGetLastError());
        if (argus_profile::enabled()) { ++argus_profile::kernel_launches[argus_profile::phase]; }
    } else for (int64_t first = 0; first < k->ne[2]; first += 32) {
        const size_t count = std::min<int64_t>(32, k->ne[2] - first);
        attention_tile<true><<<q->ne[1] * q->ne[2], 256, 0, request.stream>>>(
            static_cast<const char *>(q->data), nullptr, nullptr, static_cast<const char *>(mask->data),
            mask->type == GGML_TYPE_F16, static_cast<float *>(dst->data), static_cast<float *>(state->data()),
            q->ne[0], v->ne[0], q->ne[1], k->ne[1], q->ne[2], first, count, first + count == size_t(k->ne[2]),
            q->nb[1], q->nb[2], mask->nb[0], mask->nb[1], scale, pages);
        check(cudaGetLastError());
        if (argus_profile::enabled()) { ++argus_profile::kernel_launches[argus_profile::phase]; }
    }
    if (end) { check(cudaEventRecord(end->value, request.stream)); }
    // The sole wait protects borrowed pages and the pointer table/state lifetime.
    argus_cuda_wait(request.stream);
    drain.pending = false;
    if (end) { elapsed(argus_profile::kernel_gpu, *begin, *end); }
    if (argus_profile::enabled()) {
        ++argus_profile::resident_calls[argus_profile::phase];
        argus_profile::resident_table_bytes[argus_profile::phase] += bytes;
    }
}

bool try_resident(ggml_tensor * dst, cudaStream_t stream, size_t state_bytes) {
    const char * path = std::getenv("ARGUS_KV_ATTENTION_PATH");
    if (path && std::strcmp(path, "staged") != 0 && std::strcmp(path, "direct") != 0 && std::strcmp(path, "batched") != 0) {
        throw std::invalid_argument("ARGUS_KV_ATTENTION_PATH must be staged, direct or batched");
    }
    using namespace argus_profile;
    if (dst->src[0]->ne[2] <= 1) { reject(reject_q1); return false; }
    if (path && std::strcmp(path, "staged") == 0) { reject(reject_forced_staged); return false; }
    const bool batched = !path || std::strcmp(path, "batched") == 0;
    if (batched) { state_bytes = 0; }
    size_t count = 0;
    for (auto * tensor : {dst->src[1], dst->src[2]}) {
        if (reinterpret_cast<uintptr_t>(tensor->data) % sizeof(half)) { reject(reject_alignment); return false; }
        count += (reinterpret_cast<uintptr_t>(tensor->data) % 4096 + ggml_nbytes(tensor) + 4095) / 4096;
    }
    const size_t bytes = aligned(count * sizeof(void *));
    const auto budget = argus_tier_budget(ArgusTier::gpu);
    if (bytes + state_bytes > budget.limit || budget.live > budget.limit - bytes - state_bytes) {
        reject(reject_gpu_table_budget); return false;
    }
    if (2 * bytes + state_bytes > argus_disk_staging_limit()) { reject(reject_staging_budget); return false; }
    ArgusStagingBuffer table(bytes);
    ResidentRequest request{dst, stream, static_cast<const void **>(table.data()), state_bytes, batched};
    request.table_bytes = bytes;
    static constexpr ArgusColdStaging cold{reserve_cold, upload_cold};
    if (!argus_disk_read_resident(dst->src[1], dst->src[2], request.table, count, resident_compute, &request, &cold)) { return false; }
    request.cold_drain.pending = false; // resident_compute drained the stream.
    ++resident_accepted[phase];
    return true;
}

void compute_impl(ggml_tensor * dst, int device, void * raw_stream) {
    argus_profile::Scope total(argus_profile::attention);
    if (device != 0) { throw std::runtime_error("ARGUS v0.5 CUDA supports device 0 only"); }
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    const auto kr = argus_disk_revision(k), vr = argus_disk_revision(v);
    const size_t cells = std::min<size_t>(32, k->ne[2]);
    const size_t key_bytes = cells * k->nb[2], value_bytes = cells * v->nb[2];
    const size_t tile_bytes = aligned(key_bytes + value_bytes);
    const size_t state_bytes = aligned(2 * sizeof(float) * q->ne[1] * q->ne[2]);
    argus_kv_policy_prepare(2 * tile_bytes + state_bytes, 2 * tile_bytes);
    if (try_resident(dst, stream, state_bytes)) {
        if (!(kr == argus_disk_revision(k)) || !(vr == argus_disk_revision(v))) { throw std::runtime_error("ARGUS stale CUDA attention"); }
        argus_kv_policy_observe(k);
        argus_kv_policy_observe(v);
        ++calls;
        return;
    }
    // Includes both double-buffered host/device tiles and the online-softmax state.
    ArgusStagingReservation reservation(4 * tile_bytes + state_bytes);
    std::array<std::unique_ptr<ArgusTierBuffer>, 2> host, gpu;
    for (int i = 0; i < 2; ++i) {
        host[i] = std::make_unique<ArgusTierBuffer>(ArgusTier::pinned, tile_bytes);
        gpu[i] = std::make_unique<ArgusTierBuffer>(ArgusTier::gpu, tile_bytes);
    }
    ArgusTierBuffer state(ArgusTier::gpu, state_bytes);
    Stream transfer;
    Event done[2];
    std::unique_ptr<Event[]> start_events;
    if (argus_profile::cuda_events()) { start_events = std::make_unique<Event[]>(2); }
    bool pending_kernel[2]{};
    // Construct after buffers: all queued work drains before their destructors.
    Drain drain{stream};
    float scale;
    std::memcpy(&scale, reinterpret_cast<const char *>(dst->op_params) + sizeof(ggml_custom_op_params), sizeof scale);
    const bool overlap = std::getenv("ARGUS_KV_NO_OVERLAP") == nullptr;
    for (int64_t first = 0, step = 0; first < k->ne[2]; first += cells, ++step) {
        const int slot = step % 2;
        if (step >= 2) {
            { argus_profile::Scope timer(argus_profile::synchronization); check(cudaEventSynchronize(done[slot].value)); }
            if (start_events) { elapsed(argus_profile::kernel_gpu, start_events[slot], done[slot]); pending_kernel[slot] = false; }
        }
        if (!overlap) { argus_profile::Scope timer(argus_profile::synchronization); check(cudaStreamSynchronize(stream)); }
        const size_t count = std::min<int64_t>(cells, k->ne[2] - first);
        argus_disk_stage_cuda(k, host[slot]->data(), gpu[slot]->data(), first * k->nb[2], count * k->nb[2], transfer.value);
        argus_disk_stage_cuda(v, static_cast<char *>(host[slot]->data()) + key_bytes,
            static_cast<char *>(gpu[slot]->data()) + key_bytes, first * v->nb[2], count * v->nb[2], transfer.value);
        if (start_events) { check(cudaEventRecord(start_events[slot].value, stream)); }
        attention_tile<false><<<q->ne[1] * q->ne[2], 256, 0, stream>>>(
            static_cast<const char *>(q->data), static_cast<const half *>(gpu[slot]->data()),
            reinterpret_cast<const half *>(static_cast<char *>(gpu[slot]->data()) + key_bytes),
            static_cast<const char *>(mask->data), mask->type == GGML_TYPE_F16,
            static_cast<float *>(dst->data), static_cast<float *>(state.data()),
            q->ne[0], v->ne[0], q->ne[1], k->ne[1], q->ne[2], first, count, first + count == size_t(k->ne[2]),
            q->nb[1], q->nb[2], mask->nb[0], mask->nb[1], scale, {});
        if (argus_profile::enabled()) { ++argus_profile::kernel_launches[argus_profile::phase]; }
        check(cudaGetLastError());
        check(cudaEventRecord(done[slot].value, stream));
        pending_kernel[slot] = true;
    }
    { argus_profile::Scope timer(argus_profile::synchronization); check(cudaStreamSynchronize(stream)); }
    drain.pending = false;
    if (start_events) {
        for (int slot = 0; slot < 2; ++slot) {
            if (pending_kernel[slot]) { elapsed(argus_profile::kernel_gpu, start_events[slot], done[slot]); }
        }
    }
    if (!(kr == argus_disk_revision(k)) || !(vr == argus_disk_revision(v))) { throw std::runtime_error("ARGUS stale CUDA attention"); }
    argus_kv_policy_observe(k);
    argus_kv_policy_observe(v);
    ++calls;
}
void compute(ggml_tensor * dst, int device, void * raw_stream) {
    argus_profile::InPhase phase(dst->src[0]->ne[2] > 1);
    compute_impl(dst, device, raw_stream);
    // Publish after timers and scratch destructors, including the final attention call.
    argus_disk_publish_stats();
}
} // namespace

ArgusTierBuffer::ArgusTierBuffer(ArgusTier tier, size_t bytes) : tier_(tier), bytes_(aligned(bytes)) {
    argus_profile::Scope timer(argus_profile::allocation);
    const size_t maximum = setting(budget_name(tier));
    const auto index = static_cast<size_t>(tier);
    {
        std::lock_guard<std::mutex> lock(memory_mutex);
        if (bytes_ > maximum || live[index] > maximum - bytes_) { throw std::runtime_error("ARGUS tier budget exceeded"); }
        live[index] += bytes_;
    }
    try {
        if (tier == ArgusTier::gpu) { check(cudaSetDevice(0)); check(cudaMalloc(&data_, bytes_)); }
        else if (tier == ArgusTier::pinned) { check(cudaHostAlloc(&data_, bytes_, cudaHostAllocDefault)); }
        else {
            data_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (data_ == MAP_FAILED) { data_ = nullptr; throw std::bad_alloc(); }
        }
        std::lock_guard<std::mutex> lock(memory_mutex);
        peak[index] = std::max(peak[index], live[index]);
    } catch (...) {
        std::lock_guard<std::mutex> lock(memory_mutex);
        live[index] -= bytes_;
        throw;
    }
}
ArgusTierBuffer::~ArgusTierBuffer() {
    argus_profile::Scope timer(argus_profile::release);
    if (tier_ == ArgusTier::gpu) {
        if (cudaSetDevice(0) != cudaSuccess || cudaFree(data_) != cudaSuccess) { GGML_ABORT("ARGUS CUDA free failed"); }
    } else if (tier_ == ArgusTier::pinned) {
        if (cudaFreeHost(data_) != cudaSuccess) { GGML_ABORT("ARGUS pinned free failed"); }
    } else if (munmap(data_, bytes_)) { GGML_ABORT("ARGUS RAM free failed"); }
    std::lock_guard<std::mutex> lock(memory_mutex);
    live[static_cast<size_t>(tier_)] -= bytes_;
}
void ArgusTierBuffer::read(void * target, size_t offset, size_t bytes) const {
    argus_profile::Scope timer(argus_profile::tier_read);
    if (offset > bytes_ || bytes > bytes_ - offset) { throw std::out_of_range("ARGUS tier read"); }
    const char * source = static_cast<const char *>(data_) + offset;
    if (tier_ == ArgusTier::gpu) { check(cudaSetDevice(0)); check(cudaMemcpy(target, source, bytes, cudaMemcpyDeviceToHost)); }
    else { std::memcpy(target, source, bytes); }
}
void ArgusTierBuffer::write(const void * source, size_t bytes) {
    argus_profile::Scope timer(argus_profile::tier_write);
    if (bytes > bytes_) { throw std::out_of_range("ARGUS tier write"); }
    if (tier_ == ArgusTier::gpu) { check(cudaSetDevice(0)); check(cudaMemcpy(data_, source, bytes, cudaMemcpyHostToDevice)); }
    else { std::memcpy(data_, source, bytes); }
}
ArgusTierUsage argus_tier_usage() {
    std::lock_guard<std::mutex> lock(memory_mutex);
    return {live[3], live[2], live[1], peak[3], peak[2], peak[1], calls.load()};
}
ArgusTierBudget argus_tier_budget(ArgusTier tier) {
    const auto * name = budget_name(tier);
    const size_t maximum = std::getenv(name) ? setting(name) : 0;
    std::lock_guard<std::mutex> lock(memory_mutex);
    return {maximum, live[static_cast<size_t>(tier)]};
}
void argus_cuda_copy(void * target, const void * source, size_t bytes, bool device_source, void * stream) {
    argus_profile::Scope timer(argus_profile::copy_enqueue);
    CopyEvents * events = nullptr;
    if (argus_profile::enabled()) {
        (device_source ? argus_profile::d2d_bytes : argus_profile::h2d_bytes)[argus_profile::phase] += bytes;
    }
    if (argus_profile::cuda_events()) {
        if (pending_copies == copy_events.size()) { copy_events.push_back(std::make_unique<CopyEvents>()); }
        events = copy_events[pending_copies].get();
        events->device_source = device_source;
        check(cudaEventRecord(events->start.value, static_cast<cudaStream_t>(stream)));
    }
    check(cudaMemcpyAsync(target, source, bytes, device_source ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                          static_cast<cudaStream_t>(stream)));
    if (events) { check(cudaEventRecord(events->end.value, static_cast<cudaStream_t>(stream))); ++pending_copies; }
}
void argus_cuda_wait(void * stream) {
    { argus_profile::Scope timer(argus_profile::synchronization); check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))); }
    for (size_t i = 0; i < pending_copies; ++i) {
        const auto & events = *copy_events[i];
        elapsed(events.device_source ? argus_profile::d2d_gpu : argus_profile::h2d_gpu, events.start, events.end);
    }
    pending_copies = 0;
}

ggml_tensor * argus_ggml_cuda_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                       ggml_tensor * v, ggml_tensor * mask, float scale) {
    if (!q || !k || !v || !mask || q->type != GGML_TYPE_F32 ||
        !argus_ggml_is_disk_tensor(k) || !argus_ggml_is_disk_tensor(v) ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[1] <= 0 || k->ne[2] <= 0 ||
        q->ne[0] <= 0 || v->ne[0] <= 0 || q->ne[1] <= 0 || q->ne[2] <= 0 ||
        q->ne[0] != k->ne[0] || q->ne[1] % k->ne[1] || k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2] ||
        (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) ||
        mask->ne[0] < k->ne[2] || mask->ne[1] < q->ne[2] || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || q->ne[0] > 256 || v->ne[0] > 256 ||
        q->nb[0] != sizeof(float) || k->nb[1] != size_t(k->ne[0]) * 2 || v->nb[1] != size_t(v->ne[0]) * 2 ||
        k->nb[2] != k->nb[1] * k->ne[1] || v->nb[2] != v->nb[1] * v->ne[1] ||
        q->ne[1] > INT_MAX / q->ne[2] / std::max(q->ne[0], v->ne[0]) ||
        k->ne[1] > INT_MAX / 32 / std::max(k->ne[0], v->ne[0]) ||
        k->ne[2] > INT_MAX || !std::isfinite(scale)) {
        throw std::runtime_error("ARGUS CUDA attention requires FP16 KV, packed cells, D<=256 and single-stream GQA");
    }
    setting("ARGUS_KV_GPU_BYTES"); setting("ARGUS_KV_PINNED_BYTES");
    static const bool registered = [] {
        ggml_backend_cuda_register_custom_op(cpu_refused, compute, argus_ggml_disk_buffer_type());
        return true;
    }();
    (void) registered;
    ggml_tensor * args[] = {q, k, v, mask};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, v->ne[0], q->ne[1], q->ne[2], 1,
                                  args, 4, cpu_refused, 1, nullptr);
    static_assert(sizeof(ggml_custom_op_params) + sizeof scale <= GGML_MAX_OP_PARAMS);
    std::memcpy(reinterpret_cast<char *>(result->op_params) + sizeof(ggml_custom_op_params), &scale, sizeof scale);
    return result;
}
bool argus_ggml_is_cuda_attention(const ggml_tensor * tensor) {
    if (tensor->op != GGML_OP_CUSTOM) { return false; }
    ggml_custom_op_params params;
    std::memcpy(&params, tensor->op_params, sizeof params);
    return params.fun == cpu_refused;
}
