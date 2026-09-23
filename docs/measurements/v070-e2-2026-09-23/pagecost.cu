// Measures the exact per-page costs of the ARGUS GPU-control write path on this GPU.
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <nmmintrin.h>

static const size_t P = 4096;
static const int N = 12000;   // pages per prefill
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static uint32_t fnv(const void * d) {
    uint32_t h = 2166136261u;
    const auto * b = static_cast<const unsigned char *>(d);
    for (size_t i = 0; i < P; ++i) { h = (h ^ b[i]) * 16777619u; }
    return h;
}
static uint32_t crc32c(const void * d) {
    uint64_t h = 0xffffffffu;
    const auto * b = static_cast<const unsigned char *>(d);
    for (size_t i = 0; i < P; i += 8) { h = _mm_crc32_u64(h, *reinterpret_cast<const uint64_t *>(b + i)); }
    return static_cast<uint32_t>(h ^ 0xffffffffu);
}

int main() {
    void * pageable = mmap(nullptr, P, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(pageable, 0x5a, P);
    void * pinned = nullptr;
    cudaHostAlloc(&pinned, P, cudaHostAllocDefault);
    memset(pinned, 0x5a, P);
    void * dev = nullptr;
    cudaMalloc(&dev, P);
    // warm the context
    cudaMemcpy(dev, pinned, P, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    volatile uint32_t sink = 0;
    auto t0 = clk::now();
    for (int i = 0; i < N; ++i) { sink = fnv(pageable); }
    auto t1 = clk::now();
    for (int i = 0; i < N; ++i) { sink = crc32c(pageable); }
    auto t2 = clk::now();
    for (int i = 0; i < N; ++i) { cudaMemcpy(dev, pageable, P, cudaMemcpyHostToDevice); }
    auto t3 = clk::now();
    for (int i = 0; i < N; ++i) { cudaMemcpy(pageable, dev, P, cudaMemcpyDeviceToHost); }
    auto t4 = clk::now();
    for (int i = 0; i < N; ++i) { cudaMemcpy(dev, pinned, P, cudaMemcpyHostToDevice); }
    auto t5 = clk::now();
    for (int i = 0; i < N; ++i) { cudaMemcpy(pinned, dev, P, cudaMemcpyDeviceToHost); }
    auto t6 = clk::now();
    void ** blocks = (void **) malloc(sizeof(void *) * N);
    for (int i = 0; i < N; ++i) { cudaMalloc(&blocks[i], P); }
    auto t7 = clk::now();
    for (int i = 0; i < N; ++i) { cudaFree(blocks[i]); }
    auto t8 = clk::now();
    // slab: one 4 MiB allocation per 1024 pages
    void ** slabs = (void **) malloc(sizeof(void *) * (N / 1024 + 1));
    int s = 0;
    for (int i = 0; i < N; i += 1024) { cudaMalloc(&slabs[s++], 1024 * P); }
    auto t9 = clk::now();
    for (int i = 0; i < s; ++i) { cudaFree(slabs[i]); }
    auto t10 = clk::now();
    (void) sink;

    printf("per prefill (%d pages), total ms / per-call us\n", N);
    printf("  fnv1a checksum x1     %8.1f ms  %6.2f us\n", ms(t0,t1), ms(t0,t1)*1000/N);
    printf("  crc32c checksum x1    %8.1f ms  %6.2f us\n", ms(t1,t2), ms(t1,t2)*1000/N);
    printf("  H2D 4K pageable       %8.1f ms  %6.2f us\n", ms(t2,t3), ms(t2,t3)*1000/N);
    printf("  D2H 4K pageable       %8.1f ms  %6.2f us\n", ms(t3,t4), ms(t3,t4)*1000/N);
    printf("  H2D 4K pinned         %8.1f ms  %6.2f us\n", ms(t4,t5), ms(t4,t5)*1000/N);
    printf("  D2H 4K pinned         %8.1f ms  %6.2f us\n", ms(t5,t6), ms(t5,t6)*1000/N);
    printf("  cudaMalloc 4K         %8.1f ms  %6.2f us\n", ms(t6,t7), ms(t6,t7)*1000/N);
    printf("  cudaFree 4K           %8.1f ms  %6.2f us\n", ms(t7,t8), ms(t7,t8)*1000/N);
    printf("  slab malloc 4M x%-3d   %8.1f ms\n", s, ms(t8,t9));
    printf("  slab free             %8.1f ms\n", ms(t9,t10));
    return 0;
}
