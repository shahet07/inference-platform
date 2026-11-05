#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "stream/sharded_log.h"
#include "stream/window.h"

namespace stream {

struct ProcessorConfig {
  WindowConfig window;
  std::size_t batch_size = 1024;
  // A caught-up consumer waits for this many events or this long, whichever comes first, instead
  // of being woken for every event. See docs/benchmarks.md.
  std::size_t min_batch_size = 256;
  std::chrono::milliseconds max_batch_wait{5};
  std::size_t retained_results = 4096;  // Closed windows kept for closedWindows().
  // Current time in ms. When set, idle shards advance their watermark with it.
  std::function<std::int64_t()> clock;
};

// Consumes a log with one thread per shard, each owning a WindowAggregator, and keeps the most
// recently closed windows. Commits after each batch, so full logs drain under kBlock.
class StreamProcessor {
 public:
  StreamProcessor(ShardedLog& log, ProcessorConfig config);
  // Closes the log, consumes what remains, and flushes open windows.
  ~StreamProcessor();

  StreamProcessor(const StreamProcessor&) = delete;
  StreamProcessor& operator=(const StreamProcessor&) = delete;
  StreamProcessor(StreamProcessor&&) = delete;
  StreamProcessor& operator=(StreamProcessor&&) = delete;

  // Stops consuming after draining the log. Safe to call more than once.
  void stop();

  [[nodiscard]] std::vector<WindowResult> closedWindows() const;
  [[nodiscard]] std::uint64_t processedEvents() const;
  [[nodiscard]] std::uint64_t skippedEvents() const;
  [[nodiscard]] std::uint64_t lateEvents() const;

 private:
  void consume(std::size_t shard);
  void publish(std::vector<WindowResult> results);

  ShardedLog& log_;
  ProcessorConfig config_;
  ShardedLog::GroupId group_;
  mutable std::mutex results_mutex_;
  std::deque<WindowResult> results_;
  std::atomic<std::uint64_t> processed_{0};
  std::atomic<std::uint64_t> skipped_{0};
  std::atomic<std::uint64_t> late_{0};
  std::vector<std::thread> threads_;
};

}  // namespace stream
