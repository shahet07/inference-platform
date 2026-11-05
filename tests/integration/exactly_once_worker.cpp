// Runs a producer and a consumer in one process, restarting from nothing but disk each time. The
// producer sends every event from the beginning on each start, as a client retrying after a crash
// would. When all events have been consumed, this prints the final totals and exits.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "stream/exactly_once.h"
#include "stream/sharded_log.h"

namespace {
constexpr std::uint64_t kEvents = 150000;
constexpr std::uint64_t kKeys = 50;
constexpr std::size_t kShards = 4;
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::span args(argv, static_cast<std::size_t>(argc));
    const std::string dir = args[1];
    stream::WalConfig wal;
    wal.dir = dir + "/log";
    // Survives a process crash, the failure tested here, while staying fast enough for many events.
    wal.fsync = stream::FsyncPolicy::kNever;
    wal.segment_bytes = std::size_t{128} * 1024;
    stream::ShardedLog log({.shards = kShards,
                            .shard_capacity = 8192,
                            .policy = stream::BackpressurePolicy::kBlock,
                            .wal = wal});
    const auto group = log.registerDurableGroup("exactly-once");

    std::thread producer([&log] {
      for (std::uint64_t i = 1; i <= kEvents; ++i) {
        const auto status = log.appendIdempotent({.key = "key-" + std::to_string(i % kKeys),
                                                  .value = static_cast<double>(i % 7),
                                                  .timestamp_ms = 0,
                                                  .producer_id = 1,
                                                  .sequence = i})
                                .status;
        if (status == stream::AppendStatus::kClosed) {
          return;
        }
      }
    });

    std::vector<std::unique_ptr<stream::ExactlyOnceAggregator>> aggregators;
    aggregators.reserve(kShards);
    for (std::size_t shard = 0; shard < kShards; ++shard) {
      aggregators.push_back(
          std::make_unique<stream::ExactlyOnceAggregator>(log, group, shard, dir + "/state"));
    }
    std::vector<std::thread> consumers;
    consumers.reserve(kShards);
    std::atomic<std::uint64_t> consumed{0};
    for (auto& aggregator_ptr : aggregators) {
      consumers.emplace_back([&consumed, aggregator = aggregator_ptr.get()] {
        std::uint64_t already = 0;
        for (const auto& [key, totals] : aggregator->totals()) {
          already += totals.count;
        }
        consumed += already;
        std::uint64_t since_checkpoint = 0;
        while (consumed.load() < kEvents) {
          const std::size_t applied = aggregator->poll(512);
          consumed += applied;
          since_checkpoint += applied;
          if (since_checkpoint >= 1000 || (applied == 0 && since_checkpoint > 0)) {
            aggregator->checkpoint();
            since_checkpoint = 0;
          }
          if (applied == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }
        aggregator->checkpoint();
      });
    }
    producer.join();
    for (std::thread& consumer : consumers) {
      consumer.join();
    }

    std::uint64_t total = 0;
    double sum = 0.0;
    for (const auto& aggregator : aggregators) {
      for (const auto& [key, totals] : aggregator->totals()) {
        std::cout << key << ' ' << totals.count << ' ' << totals.sum << '\n';
        total += totals.count;
        sum += totals.sum;
      }
      if (aggregator->skippedEvents() != 0) {
        std::cerr << "skipped events\n";
        return EXIT_FAILURE;
      }
    }
    const auto stats = log.stats();
    std::cout << "total " << total << ' ' << sum << ' ' << stats.duplicates << ' '
              << stats.recovered << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
