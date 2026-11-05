// Single-threaded cost of each streaming operation, on 1000 keys. Prints a JSON object of events/s.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "stream/interval_join.h"
#include "stream/sharded_log.h"
#include "stream/sliding_window.h"
#include "stream/window.h"

namespace {

constexpr std::int64_t kEvents = 3'000'000;

double measure(const std::function<void(std::int64_t, const std::string&)>& operation) {
  std::vector<std::string> keys;
  keys.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    keys.push_back("key-" + std::to_string(i));
  }
  const auto start = std::chrono::steady_clock::now();
  for (std::int64_t i = 0; i < kEvents; ++i) {
    operation(i, keys[static_cast<std::size_t>(i % 1000)]);
  }
  const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
  return static_cast<double>(kEvents) / elapsed.count();
}

}  // namespace

int main() {
  try {
    nlohmann::json results;
    {
      stream::WindowAggregator aggregator({.window_ms = 60000});
      results["tumbling_window"] = measure([&](std::int64_t i, const std::string& key) {
        aggregator.add(key, 1.0, i / 100);
        if (i % 1024 == 0) {
          (void)aggregator.takeClosed();
        }
      });
    }
    {
      stream::SlidingWindowAggregator aggregator({.window_ms = 60000, .slide_ms = 10000});
      results["sliding_window_6_panes"] = measure([&](std::int64_t i, const std::string& key) {
        aggregator.add(key, 1.0, i / 100);
        if (i % 1024 == 0) {
          (void)aggregator.takeClosed();
        }
      });
    }
    {
      stream::IntervalJoin join({.max_gap_ms = 5});
      results["interval_join"] = measure([&](std::int64_t i, const std::string& key) {
        (void)join.add((i % 2 == 0) ? stream::JoinSide::kLeft : stream::JoinSide::kRight, key, 1.0,
                       i / 100);
      });
    }
    {
      stream::ShardedLog log({.shards = 8,
                              .shard_capacity = 65536,
                              .policy = stream::BackpressurePolicy::kDropOldest});
      results["append"] = measure([&](std::int64_t i, const std::string& key) {
        (void)log.append({.key = key, .value = 1.0, .timestamp_ms = i});
      });
    }
    {
      stream::ShardedLog log({.shards = 8,
                              .shard_capacity = 65536,
                              .policy = stream::BackpressurePolicy::kDropOldest});
      results["append_idempotent"] = measure([&](std::int64_t i, const std::string& key) {
        (void)log.appendIdempotent({.key = key,
                                    .value = 1.0,
                                    .timestamp_ms = i,
                                    .producer_id = 1,
                                    .sequence = static_cast<std::uint64_t>(i) + 1});
      });
    }
    std::cout << results.dump() << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
