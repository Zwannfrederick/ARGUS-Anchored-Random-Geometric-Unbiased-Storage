// PCIe transfer bandwidth: host<->device copies on one stream, pinned and pageable host
// memory, plus both directions at once on two streams. Median of 10 timed copies.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

static double median_gbps(void * dst, const void * src, size_t bytes, cudaMemcpyKind kind, cudaStream_t s) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    std::vector<float> ms;
    for (int i = 0; i < 12; ++i) {
        CHECK(cudaEventRecord(a, s));
        CHECK(cudaMemcpyAsync(dst, src, bytes, kind, s));
        CHECK(cudaEventRecord(b, s));
        CHECK(cudaEventSynchronize(b));
        float t;
        CHECK(cudaEventElapsedTime(&t, a, b));
        if (i >= 2) { ms.push_back(t); } // two warm-ups
    }
    std::sort(ms.begin(), ms.end());
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return bytes / (ms[ms.size() / 2] * 1e6);
}

int main() {
    const size_t max_bytes = size_t(256) << 20;
    void *device, *device2, *pinned, *pinned2;
    CHECK(cudaMalloc(&device, max_bytes));
    CHECK(cudaMalloc(&device2, max_bytes));
    CHECK(cudaMallocHost(&pinned, max_bytes));
    CHECK(cudaMallocHost(&pinned2, max_bytes));
    void * pageable = std::malloc(max_bytes);
    std::fill_n(static_cast<char *>(pageable), max_bytes, 1);
    std::fill_n(static_cast<char *>(pinned), max_bytes, 1);
    cudaStream_t s, s2;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    std::printf("{\n  \"single_direction_gbps\": [\n");
    const size_t sizes[] = {size_t(4) << 10, size_t(64) << 10, size_t(1) << 20, size_t(16) << 20, max_bytes};
    for (size_t i = 0; i < 5; ++i) {
        const size_t n = sizes[i];
        std::printf("    {\"bytes\": %zu, \"h2d_pinned\": %.2f, \"d2h_pinned\": %.2f, \"h2d_pageable\": %.2f, \"d2h_pageable\": %.2f}%s\n", n,
                    median_gbps(device, pinned, n, cudaMemcpyHostToDevice, s),
                    median_gbps(pinned, device, n, cudaMemcpyDeviceToHost, s),
                    median_gbps(device, pageable, n, cudaMemcpyHostToDevice, s),
                    median_gbps(pageable, device, n, cudaMemcpyDeviceToHost, s), i < 4 ? "," : "");
    }
    // Both directions at once: total bytes moved / wall time.
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    std::vector<float> ms;
    for (int i = 0; i < 12; ++i) {
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(a, 0));
        CHECK(cudaMemcpyAsync(device, pinned, max_bytes, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(pinned2, device2, max_bytes, cudaMemcpyDeviceToHost, s2));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(b, 0));
        CHECK(cudaEventSynchronize(b));
        float t;
        CHECK(cudaEventElapsedTime(&t, a, b));
        if (i >= 2) { ms.push_back(t); }
    }
    std::sort(ms.begin(), ms.end());
    std::printf("  ],\n  \"bidirectional_pinned_total_gbps\": %.2f\n}\n", 2.0 * max_bytes / (ms[ms.size() / 2] * 1e6));
    return 0;
}
