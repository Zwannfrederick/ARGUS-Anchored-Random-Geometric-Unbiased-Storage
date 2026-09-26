// Host RAM streaming-read bandwidth with 1..8 threads over a 2 GiB buffer (far beyond
// the 8 MiB L3). Each thread sums its slice; median of 7 passes after one warm-up.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
    const size_t words = (size_t(2) << 30) / sizeof(uint64_t);
    std::vector<uint64_t> data(words);
    for (size_t i = 0; i < words; ++i) { data[i] = i; }
    std::printf("{\n  \"buffer_bytes\": %zu,\n  \"read_gbps_by_threads\": {", words * sizeof(uint64_t));
    const int counts[] = {1, 2, 4, 6, 8};
    for (int c = 0; c < 5; ++c) {
        const int threads = counts[c];
        std::vector<double> seconds;
        uint64_t sink = 0;
        for (int pass = 0; pass < 8; ++pass) {
            std::vector<uint64_t> sums(threads);
            std::vector<std::thread> pool;
            const auto start = std::chrono::steady_clock::now();
            for (int t = 0; t < threads; ++t) {
                pool.emplace_back([&, t] {
                    const size_t begin = words * t / threads, end = words * (t + 1) / threads;
                    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    for (size_t i = begin; i + 4 <= end; i += 4) { s0 += data[i]; s1 += data[i + 1]; s2 += data[i + 2]; s3 += data[i + 3]; }
                    sums[t] = s0 + s1 + s2 + s3;
                });
            }
            for (auto & th : pool) { th.join(); }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            for (auto v : sums) { sink += v; }
            if (pass) { seconds.push_back(s); }
        }
        std::sort(seconds.begin(), seconds.end());
        std::printf("%s\"%d\": %.2f", c ? ", " : "", threads, words * sizeof(uint64_t) / seconds[seconds.size() / 2] / 1e9);
        if (sink == 42) { std::printf(" "); } // keep the sums observable
    }
    std::printf("}\n}\n");
    return 0;
}
