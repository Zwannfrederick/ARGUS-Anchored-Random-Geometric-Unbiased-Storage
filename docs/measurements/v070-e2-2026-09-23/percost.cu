// Minimal-diff variants: keep verification per page, change only how the two copies sync.
#include <cstdio>
#include <chrono>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>
static const size_t P = 4096;
static const int N = 12000;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
int main() {
    void * pageable = mmap(nullptr, P, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(pageable, 0x5a, P);
    void * pinned = nullptr; cudaHostAlloc(&pinned, P, cudaHostAllocDefault); memset(pinned, 0x5a, P);
    void * dev = nullptr; cudaMalloc(&dev, P);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaMemcpy(dev, pinned, P, cudaMemcpyHostToDevice); cudaDeviceSynchronize();

    auto t0 = clk::now();
    for (int i = 0; i < N; ++i) {
        cudaMemcpy(dev, pageable, P, cudaMemcpyHostToDevice);
        cudaMemcpy(pageable, dev, P, cudaMemcpyDeviceToHost);
    }
    auto t1 = clk::now();
    for (int i = 0; i < N; ++i) {
        cudaMemcpyAsync(dev, pinned, P, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(pinned, dev, P, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
    }
    auto t2 = clk::now();
    for (int i = 0; i < N; ++i) {
        cudaMemcpyAsync(dev, pageable, P, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(pageable, dev, P, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
    }
    auto t3 = clk::now();
    for (int i = 0; i < N; ++i) {
        cudaMemcpy(dev, pinned, P, cudaMemcpyHostToDevice);
        cudaMemcpy(pinned, dev, P, cudaMemcpyDeviceToHost);
    }
    auto t4 = clk::now();
    printf("%d pages; ms per prefill / us per page\n", N);
    printf("  A  2x sync memcpy, pageable (today)   %7.1f ms  %6.2f us\n", ms(t0,t1), ms(t0,t1)*1000/N);
    printf("  F  2x async + 1 sync, pinned          %7.1f ms  %6.2f us\n", ms(t1,t2), ms(t1,t2)*1000/N);
    printf("  G  2x async + 1 sync, pageable        %7.1f ms  %6.2f us\n", ms(t2,t3), ms(t2,t3)*1000/N);
    printf("  H  2x sync memcpy, pinned             %7.1f ms  %6.2f us\n", ms(t3,t4), ms(t3,t4)*1000/N);
    return 0;
}
