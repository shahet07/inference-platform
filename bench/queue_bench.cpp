// Compares the mutex queue with the lock-free queue, both spinning when full or empty, at several
// thread counts. Handing over small integers is the lock-free queue's best case. Prints JSON.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include "server/bounded_queue.h"
#include "server/mpmc_queue.h"

namespace {

constexpr std::size_t kCapacity = 1024;
constexpr std::int64_t kItems = 8'000'000;

template <typename Queue, typename Pop>
double measure(Queue& queue, int threads, Pop pop) {
  const std::int64_t per_producer = kItems / threads;
  std::atomic<std::int64_t> consumed{0};
  const std::int64_t total = per_producer * threads;
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads) * 2);
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&queue, per_producer] {
      for (std::int64_t i = 0; i < per_producer; ++i) {
        std::int64_t value = i;
        while (!queue.tryPush(value)) {
          std::this_thread::yield();
        }
      }
    });
    workers.emplace_back([&queue, &consumed, &pop, total] {
      while (consumed.load(std::memory_order_relaxed) < total) {
        if (pop(queue)) {
          consumed.fetch_add(1, std::memory_order_relaxed);
        } else {
          std::this_thread::yield();
        }
      }
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
  return static_cast<double>(total) / elapsed.count();
}

}  // namespace

int main() {
  try {
    nlohmann::json results = nlohmann::json::array();
    for (const int threads : {1, 2, 4, 8}) {
      std::array<double, 3> mutex_runs{};
      std::array<double, 3> lock_free_runs{};
      for (std::size_t run = 0; run < 3; ++run) {
        server::BoundedQueue<std::int64_t> mutex_queue(kCapacity);
        mutex_runs[run] = measure(mutex_queue, threads, [](auto& queue) {
          return queue.popUntil(std::chrono::steady_clock::time_point::min()).has_value();
        });
        server::MpmcQueue<std::int64_t> lock_free_queue(kCapacity);
        lock_free_runs[run] = measure(lock_free_queue, threads,
                                      [](auto& queue) { return queue.tryPop().has_value(); });
      }
      std::ranges::sort(mutex_runs);
      std::ranges::sort(lock_free_runs);
      results.push_back({{"producers_and_consumers", threads},
                         {"mutex_items_per_sec", mutex_runs[1]},
                         {"lock_free_items_per_sec", lock_free_runs[1]}});
    }
    std::cout << results.dump() << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
