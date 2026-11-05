// Measures the streaming engine: producers append, one consumer per shard runs windowed
// aggregation and commits. bench/stream_baseline.py runs the same workload in Python.
// Prints one JSON object of results.
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "stream/sharded_log.h"
#include "stream/window.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kKeys = 1000;
constexpr std::uint64_t kLatencySampleEvery = 64;

stream::BackpressurePolicy parsePolicy(const std::string& name) {
  if (name == "block") {
    return stream::BackpressurePolicy::kBlock;
  }
  if (name == "drop_newest") {
    return stream::BackpressurePolicy::kDropNewest;
  }
  if (name == "drop_oldest") {
    return stream::BackpressurePolicy::kDropOldest;
  }
  throw std::invalid_argument("unknown policy: " + name);
}

double percentile(const std::vector<double>& sorted, double q) {
  return sorted.empty()
             ? 0.0
             : sorted[static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1))];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::map<std::string, std::string> flags;
    const std::span args(argv, static_cast<std::size_t>(argc));
    for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
      flags[std::string(args[i]).substr(2)] = args[i + 1];
    }
    const auto number = [&flags](const std::string& name, std::uint64_t fallback) {
      return flags.contains(name) ? std::stoull(flags[name]) : fallback;
    };
    const std::uint64_t events = number("events", 10'000'000);
    const std::uint64_t producers = number("producers", 4);
    const std::uint64_t shards = number("shards", 8);
    const std::uint64_t capacity = number("capacity", 65536);
    const std::uint64_t consumer_delay_us = number("consumer-delay-us", 0);
    // 1 restores waking the consumer for every event, to measure the cost of doing that.
    const std::uint64_t min_batch = number("min-batch", 256);
    const std::string policy = flags.contains("policy") ? flags["policy"] : "block";

    std::vector<std::string> keys;
    keys.reserve(kKeys);
    for (int i = 0; i < kKeys; ++i) {
      keys.push_back("key-" + std::to_string(i));
    }

    stream::ShardedLog log(
        {.shards = shards, .shard_capacity = capacity, .policy = parsePolicy(policy)});
    const auto group = log.registerGroup();

    std::vector<std::uint64_t> consumed(shards);
    std::vector<std::uint64_t> skipped(shards);
    std::vector<std::thread> consumers;
    consumers.reserve(shards);
    for (std::size_t shard = 0; shard < shards; ++shard) {
      consumers.emplace_back([&, shard] {
        stream::WindowAggregator aggregator({.window_ms = 1000});
        std::uint64_t offset = 0;
        while (true) {
          const stream::ReadResult read =
              log.read(shard, offset, 1024, std::chrono::milliseconds(5), min_batch);
          for (const stream::Event& event : read.events) {
            aggregator.add(event.key, event.value, event.timestamp_ms);
          }
          (void)aggregator.takeClosed();
          consumed[shard] += read.events.size();
          skipped[shard] += read.skipped;
          offset = read.next_offset;
          log.commit(group, shard, offset);
          if (read.closed) {
            return;
          }
          if (consumer_delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(consumer_delay_us));
          }
        }
      });
    }

    const Clock::time_point started = Clock::now();
    std::vector<std::vector<double>> latencies(producers);
    std::vector<std::thread> threads;
    threads.reserve(producers);
    for (std::size_t p = 0; p < producers; ++p) {
      threads.emplace_back([&, p] {
        const std::uint64_t count = events / producers;
        latencies[p].reserve(count / kLatencySampleEvery + 1);
        for (std::uint64_t i = 0; i < count; ++i) {
          stream::Event event{.key = keys[(i + (p * 37)) % kKeys],
                              .value = static_cast<double>(i % 100),
                              .timestamp_ms = static_cast<std::int64_t>(i / 1000)};
          if (i % kLatencySampleEvery == 0) {
            const Clock::time_point before = Clock::now();
            (void)log.append(std::move(event));
            latencies[p].push_back(
                std::chrono::duration<double, std::micro>(Clock::now() - before).count());
          } else {
            (void)log.append(std::move(event));
          }
        }
      });
    }
    for (std::thread& t : threads) {
      t.join();
    }
    const double ingest_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    log.close();
    for (std::thread& t : consumers) {
      t.join();
    }
    const double total_seconds = std::chrono::duration<double>(Clock::now() - started).count();

    std::vector<double> all;
    for (const auto& l : latencies) {
      all.insert(all.end(), l.begin(), l.end());
    }
    std::ranges::sort(all);
    std::uint64_t total_consumed = 0;
    std::uint64_t total_skipped = 0;
    for (std::size_t s = 0; s < shards; ++s) {
      total_consumed += consumed[s];
      total_skipped += skipped[s];
    }
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const stream::LogStats stats = log.stats();
    const std::uint64_t offered = (events / producers) * producers;
    std::cout << nlohmann::json{
                     {"impl", "cpp"},
                     {"policy", policy},
                     {"producers", producers},
                     {"shards", shards},
                     {"events", offered},
                     {"ingest_events_per_sec", static_cast<double>(offered) / ingest_seconds},
                     {"processed_events_per_sec",
                      static_cast<double>(total_consumed) / total_seconds},
                     {"consumed", total_consumed},
                     {"dropped_newest", stats.dropped_newest},
                     {"evicted_oldest", stats.evicted_oldest},
                     {"blocked_appends", stats.blocked_appends},
                     {"skipped_by_consumers", total_skipped},
                     {"append_p50_us", percentile(all, 0.50)},
                     {"append_p99_us", percentile(all, 0.99)},
                     {"append_p999_us", percentile(all, 0.999)},
                     {"peak_rss_mb", static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0)},
                 }
                     .dump()
              << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
