// Appends events as fast as it can, and prints each offset only after its append returned. A parent
// process kills this with SIGKILL, then checks that everything printed survived.
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "stream/sharded_log.h"

namespace {

stream::LogConfig makeConfig(const std::string& dir) {
  stream::WalConfig wal;
  wal.dir = dir;
  wal.fsync = stream::FsyncPolicy::kEveryAppend;
  wal.segment_bytes = std::size_t{256} * 1024;
  return {.shards = 4,
          .shard_capacity = 4096,
          .policy = stream::BackpressurePolicy::kDropOldest,
          .wal = wal};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::span args(argv, static_cast<std::size_t>(argc));
    const std::string mode = args[1];
    stream::ShardedLog log(makeConfig(args[2]));
    if (mode == "dump") {
      // Prints "shard offset value" for everything retained, and how much was truncated.
      std::cout << "truncated " << log.stats().truncated_bytes << '\n';
      for (std::size_t shard = 0; shard < log.shardCount(); ++shard) {
        std::uint64_t offset = 0;
        while (true) {
          const auto read = log.read(shard, offset, 1024);
          const std::uint64_t first = read.next_offset - read.events.size();
          for (std::size_t i = 0; i < read.events.size(); ++i) {
            std::cout << shard << ' ' << first + i << ' '
                      << static_cast<long long>(read.events[i].value) << '\n';
          }
          if (read.events.empty()) {
            std::cout << "next " << shard << ' ' << read.next_offset << '\n';
            break;
          }
          offset = read.next_offset;
        }
      }
      return EXIT_SUCCESS;
    }

    // Each thread prints whole lines, written with one call, so that lines do not interleave.
    std::vector<std::thread> threads;
    threads.reserve(4);
    for (int t = 0; t < 4; ++t) {
      threads.emplace_back([&log, t] {
        for (int i = 0;; ++i) {
          const int value = (t * 100000000) + i;
          const auto result =
              log.append({.key = "k" + std::to_string(i % 97) + "-" + std::to_string(t),
                          .value = static_cast<double>(value),
                          .timestamp_ms = i});
          if (result.status == stream::AppendStatus::kAppended) {
            const std::string line = std::to_string(result.shard) + ' ' +
                                     std::to_string(result.offset) + ' ' + std::to_string(value) +
                                     '\n';
            std::cout.write(line.data(), static_cast<std::streamsize>(line.size()));
            std::cout.flush();
          }
        }
      });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
