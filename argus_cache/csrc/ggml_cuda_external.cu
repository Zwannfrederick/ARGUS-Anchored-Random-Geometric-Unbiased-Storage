// The one external custom operation ARGUS registers with the CUDA backend, and the kind
// dispatch behind it.
#include "ggml_cuda_external.h"
#include "ggml_cuda_attention.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <cstring>

namespace {
void cpu_refused(ggml_tensor *, int, int, void *) { GGML_ABORT("ARGUS CUDA operation assigned to CPU"); }

constexpr size_t scale_offset = sizeof(ggml_custom_op_params);
constexpr size_t kind_offset = scale_offset + sizeof(float);
constexpr size_t extra_offset = kind_offset + sizeof(int32_t);
static_assert(extra_offset + sizeof(int64_t) <= GGML_MAX_OP_PARAMS, "op_params hold params, scale, kind and extra");

void compute(ggml_tensor * dst, int device, void * stream) {
    switch (argus_cuda_external_kind(dst)) {
        case argus_kind_attention: argus_cuda_attention_compute(dst, device, stream); return;
        case argus_kind_set_rows: argus_cuda_set_rows_compute(dst, device, stream); return;
        case argus_kind_decode_partials:
        case argus_kind_partials_reduce:
        case argus_kind_tier_append: argus_cuda_split_compute(dst, device, stream); return;
        default: GGML_ABORT("ARGUS CUDA node of unknown kind");
    }
}

void register_external_op() {
    static const bool registered = [] {
        ggml_backend_cuda_register_custom_op(cpu_refused, compute, argus_ggml_disk_buffer_type());
        return true;
    }();
    (void) registered;
}
} // namespace

ggml_tensor * argus_cuda_external_op(ggml_context * ctx, int64_t ne0, int64_t ne1, int64_t ne2,
                                     ggml_tensor ** args, int n_args, ArgusCudaKind kind, float scale,
                                     int64_t extra) {
    register_external_op();
    ggml_tensor * result = ggml_custom_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, 1, args, n_args, cpu_refused, 1, nullptr);
    std::memcpy(reinterpret_cast<char *>(result->op_params) + scale_offset, &scale, sizeof scale);
    const int32_t value = kind;
    std::memcpy(reinterpret_cast<char *>(result->op_params) + kind_offset, &value, sizeof value);
    std::memcpy(reinterpret_cast<char *>(result->op_params) + extra_offset, &extra, sizeof extra);
    return result;
}

ggml_tensor * argus_cuda_external_inplace(ggml_context * ctx, ggml_tensor * a, ggml_tensor ** args, int n_args,
                                          ArgusCudaKind kind, float scale, int64_t extra) {
    register_external_op();
    ggml_tensor * result = ggml_custom_inplace(ctx, a, args, n_args, cpu_refused, 1, nullptr);
    std::memcpy(reinterpret_cast<char *>(result->op_params) + scale_offset, &scale, sizeof scale);
    const int32_t value = kind;
    std::memcpy(reinterpret_cast<char *>(result->op_params) + kind_offset, &value, sizeof value);
    std::memcpy(reinterpret_cast<char *>(result->op_params) + extra_offset, &extra, sizeof extra);
    return result;
}

int32_t argus_cuda_external_kind(const ggml_tensor * tensor) {
    if (tensor->op != GGML_OP_CUSTOM) { return -1; }
    ggml_custom_op_params params;
    std::memcpy(&params, tensor->op_params, sizeof params);
    if (params.fun != cpu_refused) { return -1; }
    int32_t kind;
    std::memcpy(&kind, reinterpret_cast<const char *>(tensor->op_params) + kind_offset, sizeof kind);
    return kind;
}

float argus_cuda_external_scale(const ggml_tensor * tensor) {
    float scale;
    std::memcpy(&scale, reinterpret_cast<const char *>(tensor->op_params) + scale_offset, sizeof scale);
    return scale;
}

int64_t argus_cuda_external_extra(const ggml_tensor * tensor) {
    int64_t extra;
    std::memcpy(&extra, reinterpret_cast<const char *>(tensor->op_params) + extra_offset, sizeof extra);
    return extra;
}

bool argus_ggml_is_cuda_attention(const ggml_tensor * tensor) { return argus_cuda_external_kind(tensor) == argus_kind_attention; }
bool argus_ggml_is_cuda_set_rows(const ggml_tensor * tensor) { return argus_cuda_external_kind(tensor) == argus_kind_set_rows; }
