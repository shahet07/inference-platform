// Measures what durability costs. 4 producers append to 8 shards under each fsync policy, with
// latency measured for every append, since these are slow enough that the clock read does not
// matter. Prints one JSON object per policy.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "stream/sharded_log.h"

namespace {

using Clock = std::chrono::steady_clock;

nlohmann::json run(const std::string& name, stream::FsyncPolicy policy, const std::string& dir,
                   double seconds) {
  std::filesystem::remove_all(dir);
  stream::WalConfig wal;
  wal.dir = dir;
  wal.fsync = policy;
  std::vector<std::vector<double>> latencies(4);
  double elapsed = 0.0;
  {
    stream::ShardedLog log({.shards = 8,
                            .shard_capacity = 65536,
                            .policy = stream::BackpressurePolicy::kDropOldest,
                            .wal = wal});
    const Clock::time_point start = Clock::now();
    const auto stop =
        start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    std::vector<std::thread> threads;
    threads.reserve(4);
    for (std::size_t p = 0; p < 4; ++p) {
      threads.emplace_back([&, p] {
        for (std::uint64_t i = 0;; ++i) {
          const Clock::time_point before = Clock::now();
          if (before >= stop) {
            return;
          }
          (void)log.append({.key = "key-" + std::to_string((i + (p * 37)) % 1000),
                            .value = 1.0,
                            .timestamp_ms = static_cast<std::int64_t>(i)});
          latencies[p].push_back(
              std::chrono::duration<double, std::micro>(Clock::now() - before).count());
        }
      });
    }
    for (std::thread& t : threads) {
      t.join();
    }
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  }
  std::filesystem::remove_all(dir);
  std::vector<double> all;
  for (const auto& l : latencies) {
    all.insert(all.end(), l.begin(), l.end());
  }
  std::ranges::sort(all);
  const auto at = [&all](double q) {
    return all.empty() ? 0.0
                       : all[static_cast<std::size_t>(q * static_cast<double>(all.size() - 1))];
  };
  return {{"policy", name},
          {"events_per_sec", static_cast<double>(all.size()) / elapsed},
          {"p50_us", at(0.50)},
          {"p99_us", at(0.99)},
          {"p999_us", at(0.999)}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::span args(argv, static_cast<std::size_t>(argc));
    const std::string dir =
        args.size() > 1 ? args[1] : (std::filesystem::temp_directory_path() / "wal_bench").string();
    for (const auto& [name, policy] : {std::pair{"every_append", stream::FsyncPolicy::kEveryAppend},
                                       std::pair{"interval_10ms", stream::FsyncPolicy::kInterval},
                                       std::pair{"never", stream::FsyncPolicy::kNever}}) {
      std::cout << run(name, policy, dir, 6.0).dump() << '\n' << std::flush;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
