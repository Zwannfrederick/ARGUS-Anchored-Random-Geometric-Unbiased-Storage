// Per-flush cost of the GPU-control round trip: current per-page sync vs async-batched
// vs one contiguous copy. G pages per set_rows flush, 1500 flushes per prefill.
#include <cstdio>
#include <chrono>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>

static const size_t P = 4096;
static const int G = GROUP;
static const int F = 12000 / GROUP;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
    void * pageable = mmap(nullptr, G * P, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(pageable, 0x5a, G * P);
    void * pinned = nullptr;
    cudaHostAlloc(&pinned, G * P, cudaHostAllocDefault);
    memset(pinned, 0x5a, G * P);
    void * scattered[G];
    for (int i = 0; i < G; ++i) { cudaMalloc(&scattered[i], P); }
    void * slab = nullptr;
    cudaMalloc(&slab, G * P);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpy(slab, pinned, G * P, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto t0 = clk::now();
    for (int f = 0; f < F; ++f) {
        for (int i = 0; i < G; ++i) {
            cudaMemcpy(scattered[i], (char *) pageable + i * P, P, cudaMemcpyHostToDevice);
            cudaMemcpy((char *) pageable + i * P, scattered[i], P, cudaMemcpyDeviceToHost);
        }
    }
    auto t1 = clk::now();
    for (int f = 0; f < F; ++f) {
        for (int i = 0; i < G; ++i) {
            cudaMemcpyAsync(scattered[i], (char *) pinned + i * P, P, cudaMemcpyHostToDevice, stream);
        }
        for (int i = 0; i < G; ++i) {
            cudaMemcpyAsync((char *) pinned + i * P, scattered[i], P, cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
    }
    auto t2 = clk::now();
    for (int f = 0; f < F; ++f) {
        for (int i = 0; i < G; ++i) {
            cudaMemcpyAsync(scattered[i], (char *) pageable + i * P, P, cudaMemcpyHostToDevice, stream);
        }
        for (int i = 0; i < G; ++i) {
            cudaMemcpyAsync((char *) pageable + i * P, scattered[i], P, cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
    }
    auto t3 = clk::now();
    for (int f = 0; f < F; ++f) {
        cudaMemcpy(slab, pinned, G * P, cudaMemcpyHostToDevice);
        cudaMemcpy(pinned, slab, G * P, cudaMemcpyDeviceToHost);
    }
    auto t4 = clk::now();
    for (int f = 0; f < F; ++f) {
        cudaMemcpy(slab, pageable, G * P, cudaMemcpyHostToDevice);
        cudaMemcpy(pageable, slab, G * P, cudaMemcpyDeviceToHost);
    }
    auto t5 = clk::now();

    printf("%d flushes x %d pages = %d pages; ms per prefill / us per page\n", F, G, F * G);
    printf("  A per-page sync, pageable, scattered   %7.1f ms  %6.2f us\n", ms(t0,t1), ms(t0,t1)*1000/(F*G));
    printf("  B per-page async, pinned,  1 sync      %7.1f ms  %6.2f us\n", ms(t1,t2), ms(t1,t2)*1000/(F*G));
    printf("  C per-page async, pageable,1 sync      %7.1f ms  %6.2f us\n", ms(t2,t3), ms(t2,t3)*1000/(F*G));
    printf("  D one contiguous copy, pinned          %7.1f ms  %6.2f us\n", ms(t3,t4), ms(t3,t4)*1000/(F*G));
    printf("  E one contiguous copy, pageable        %7.1f ms  %6.2f us\n", ms(t4,t5), ms(t4,t5)*1000/(F*G));
    return 0;
}
