#include "ggml_cuda_attention.h"
#include "ggml_kv_policy.h"
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
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
    Stream(const Stream &) = delete;
    Stream & operator=(const Stream &) = delete;
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    ~Event() { if (value) { cudaEventDestroy(value); } }
    Event(const Event &) = delete;
    Event & operator=(const Event &) = delete;
};
struct Drain {
    cudaStream_t value;
    ~Drain() { if (cudaStreamSynchronize(value) != cudaSuccess) { GGML_ABORT("ARGUS CUDA stream failed"); } }
};

// One block per query/head. Online softmax state survives tile boundaries.
// ponytail: scalar FP16 tiles first; replace with tiled MMA after mechanism parity.
__global__ void attention_tile(const char * query, const half * keys, const half * values,
        const char * mask, bool half_mask, float * output, float * state,
        int dk, int dv, int heads, int kv_heads, int tokens, int first, int count, bool last,
        size_t q_head, size_t q_token, size_t mask_cell, size_t mask_token, float scale) {
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
        reduction[d] = d < dk ? q[d] * __half2float(keys[(cell * kv_heads + kv_head) * dk + d]) : 0.0f;
        __syncthreads();
        for (int width = 128; width; width /= 2) {
            if (d < width) { reduction[d] += reduction[d + width]; }
            __syncthreads();
        }
        if (d == 0) {
            const float score = reduction[0] * scale + bias;
            const float next = fmaxf(maximum, score);
            alpha = expf(maximum - next); beta = expf(score - next);
            sum = alpha * sum + beta; maximum = next;
        }
        __syncthreads();
        if (d < dv) { acc = alpha * acc + beta * __half2float(values[(cell * kv_heads + kv_head) * dv + d]); }
        __syncthreads();
    }
    if (d < dv) { output[row * dv + d] = last ? (sum > 0 ? acc / sum : 0.0f) : acc; }
    if (d == 0) { state[2 * row] = maximum; state[2 * row + 1] = sum; }
}
void cpu_refused(ggml_tensor *, int, int, void *) { GGML_ABORT("ARGUS CUDA attention assigned to CPU"); }

void compute(ggml_tensor * dst, int device, void * raw_stream) {
    if (device != 0) { throw std::runtime_error("ARGUS v0.5 CUDA supports device 0 only"); }
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    const auto kr = argus_disk_revision(k), vr = argus_disk_revision(v);
    const size_t cells = std::min<size_t>(32, k->ne[2]);
    const size_t key_bytes = cells * k->nb[2], value_bytes = cells * v->nb[2];
    const size_t tile_bytes = aligned(key_bytes + value_bytes);
    const size_t state_bytes = aligned(2 * sizeof(float) * q->ne[1] * q->ne[2]);
    argus_kv_policy_prepare(2 * tile_bytes + state_bytes, 2 * tile_bytes);
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
    // Construct after buffers: all queued work drains before their destructors.
    Drain drain{stream};
    float scale;
    std::memcpy(&scale, reinterpret_cast<const char *>(dst->op_params) + sizeof(ggml_custom_op_params), sizeof scale);
    const bool overlap = std::getenv("ARGUS_KV_NO_OVERLAP") == nullptr;
    for (int64_t first = 0, step = 0; first < k->ne[2]; first += cells, ++step) {
        const int slot = step % 2;
        if (step >= 2) { check(cudaEventSynchronize(done[slot].value)); }
        if (!overlap) { check(cudaStreamSynchronize(stream)); }
        const size_t count = std::min<int64_t>(cells, k->ne[2] - first);
        argus_disk_stage_cuda(k, host[slot]->data(), gpu[slot]->data(), first * k->nb[2], count * k->nb[2], transfer.value);
        argus_disk_stage_cuda(v, static_cast<char *>(host[slot]->data()) + key_bytes,
            static_cast<char *>(gpu[slot]->data()) + key_bytes, first * v->nb[2], count * v->nb[2], transfer.value);
        attention_tile<<<q->ne[1] * q->ne[2], 256, 0, stream>>>(
            static_cast<const char *>(q->data), static_cast<const half *>(gpu[slot]->data()),
            reinterpret_cast<const half *>(static_cast<char *>(gpu[slot]->data()) + key_bytes),
            static_cast<const char *>(mask->data), mask->type == GGML_TYPE_F16,
            static_cast<float *>(dst->data), static_cast<float *>(state.data()),
            q->ne[0], v->ne[0], q->ne[1], k->ne[1], q->ne[2], first, count, first + count == size_t(k->ne[2]),
            q->nb[1], q->nb[2], mask->nb[0], mask->nb[1], scale);
        check(cudaGetLastError());
        check(cudaEventRecord(done[slot].value, stream));
    }
    check(cudaStreamSynchronize(stream));
    if (!(kr == argus_disk_revision(k)) || !(vr == argus_disk_revision(v))) { throw std::runtime_error("ARGUS stale CUDA attention"); }
    argus_kv_policy_observe(k);
    argus_kv_policy_observe(v);
    ++calls;
    argus_disk_publish_stats();
}
} // namespace

ArgusTierBuffer::ArgusTierBuffer(ArgusTier tier, size_t bytes) : tier_(tier), bytes_(aligned(bytes)) {
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
    if (tier_ == ArgusTier::gpu) {
        if (cudaSetDevice(0) != cudaSuccess || cudaFree(data_) != cudaSuccess) { GGML_ABORT("ARGUS CUDA free failed"); }
    } else if (tier_ == ArgusTier::pinned) {
        if (cudaFreeHost(data_) != cudaSuccess) { GGML_ABORT("ARGUS pinned free failed"); }
    } else if (munmap(data_, bytes_)) { GGML_ABORT("ARGUS RAM free failed"); }
    std::lock_guard<std::mutex> lock(memory_mutex);
    live[static_cast<size_t>(tier_)] -= bytes_;
}
void ArgusTierBuffer::read(void * target, size_t offset, size_t bytes) const {
    if (offset > bytes_ || bytes > bytes_ - offset) { throw std::out_of_range("ARGUS tier read"); }
    const char * source = static_cast<const char *>(data_) + offset;
    if (tier_ == ArgusTier::gpu) { check(cudaSetDevice(0)); check(cudaMemcpy(target, source, bytes, cudaMemcpyDeviceToHost)); }
    else { std::memcpy(target, source, bytes); }
}
void ArgusTierBuffer::write(const void * source, size_t bytes) {
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
    check(cudaMemcpyAsync(target, source, bytes, device_source ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                          static_cast<cudaStream_t>(stream)));
}
void argus_cuda_wait(void * stream) { check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))); }

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
