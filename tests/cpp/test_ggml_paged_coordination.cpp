// Include the implementation to test invocation ownership without a public test API.
#include "ggml_paged_attention.cpp"

#include <cassert>
#include <thread>

int main() {
    const int operation = 0;
    for (int nth : {1, 4, 16}) {
        for (int repeat = 0; repeat < 20; ++repeat) {
            std::vector<std::shared_ptr<Coordination>> workers(nth);
            std::vector<std::thread> threads;
            for (int ith = 0; ith < nth; ++ith) {
                threads.emplace_back([&, ith] {
                    workers[ith] = coordination_for(&operation, nth);
                });
            }
            for (auto & thread : threads) { thread.join(); }
            for (const auto & worker : workers) {
                assert(worker == workers[0]);
                assert(worker->evicted.load() == 0);
                for (int ith = 0; ith < nth; ++ith) {
                    assert(worker->progress[ith].load() == 0);
                }
            }
            workers[0]->evicted.store(999);
            workers[0]->progress[0].store(Evictor::done);
            std::weak_ptr<Coordination> old = workers[0];
            workers.clear();
            assert(old.expired());
        }
    }
}
