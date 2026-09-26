#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
int main() {
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    const size_t big = size_t(1) << 30;
    void *a, *b, *small, *pinned; cudaMalloc(&a, big); cudaMalloc(&b, big); cudaMalloc(&small, 4096); cudaHostAlloc(&pinned, 4096, 0);
    auto time = [&](const char * label, auto op) {
        cudaMemcpyAsync(b, a, big, cudaMemcpyDeviceToDevice, s);  // ~ms of pending work
        auto t0 = std::chrono::steady_clock::now();
        op();
        auto t1 = std::chrono::steady_clock::now();
        cudaError_t q = cudaStreamQuery(s);
        std::printf("%-28s %8.3f ms   stream after: %s\n", label, std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    q == cudaSuccess ? "idle" : "busy");
        cudaStreamSynchronize(s);
    };
    time("nothing", [] {});
    time("cudaFree(small)", [&] { cudaFree(small); cudaMalloc(&small, 4096); });
    time("cudaFreeHost(pinned)", [&] { cudaFreeHost(pinned); cudaHostAlloc(&pinned, 4096, 0); });
    time("cudaMalloc(4096)", [&] { void * p; cudaMalloc(&p, 4096); });
    time("cudaMemcpy D2H 4KiB legacy", [&] { char h[4096]; cudaMemcpy(h, small, 4096, cudaMemcpyDeviceToHost); });
    return 0;
}
