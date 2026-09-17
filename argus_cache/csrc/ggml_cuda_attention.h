#pragma once
#include "ggml_disk_buffer.h"

class ArgusTierBuffer {
public:
    ArgusTierBuffer(ArgusTier tier, size_t bytes);
    ~ArgusTierBuffer();
    ArgusTierBuffer(const ArgusTierBuffer &) = delete;
    ArgusTierBuffer & operator=(const ArgusTierBuffer &) = delete;
    void * data() const { return data_; }
    size_t size() const { return bytes_; }
    ArgusTier tier() const { return tier_; }
    void read(void * target, size_t offset, size_t bytes) const;
    void write(const void * source, size_t bytes);
private:
    ArgusTier tier_;
    size_t bytes_;
    void * data_ = nullptr;
};

struct ArgusTierUsage {
    size_t gpu, pinned, ram, peak_gpu, peak_pinned, peak_ram;
    size_t attention_calls;
};
ArgusTierUsage argus_tier_usage();
// One aligned physical page per transaction. A range may span several transactions.
void argus_disk_move_page(const ggml_tensor * tensor, size_t offset, ArgusTier tier,
                         ArgusDiskPageRevision expected);
// Source stays locked until the transfer stream completes. Host staging belongs to caller.
void argus_disk_stage_cuda(const ggml_tensor * tensor, void * host, void * device,
                           size_t offset, size_t bytes, void * stream);
void argus_cuda_copy(void * device, const void * source, size_t bytes, bool device_source, void * stream);
void argus_cuda_wait(void * stream);

ggml_tensor * argus_ggml_cuda_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                       ggml_tensor * v, ggml_tensor * mask, float scale);
bool argus_ggml_is_cuda_attention(const ggml_tensor * tensor);
