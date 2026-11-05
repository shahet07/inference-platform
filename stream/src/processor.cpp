#include "stream/processor.h"

#include <utility>

namespace stream {

StreamProcessor::StreamProcessor(ShardedLog& log, ProcessorConfig config)
    : log_(log), config_(std::move(config)), group_(log.registerGroup()) {
  threads_.reserve(log.shardCount());
  for (std::size_t shard = 0; shard < log.shardCount(); ++shard) {
    threads_.emplace_back([this, shard] { consume(shard); });
  }
}

StreamProcessor::~StreamProcessor() { stop(); }

void StreamProcessor::stop() {
  log_.close();
  for (std::thread& thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void StreamProcessor::consume(std::size_t shard) {
  WindowAggregator aggregator(config_.window);
  std::uint64_t offset = log_.committed(group_, shard);
  std::uint64_t reported_late = 0;
  while (true) {
    ReadResult read = log_.read(shard, offset, config_.batch_size, config_.max_batch_wait,
                                config_.min_batch_size);
    for (const Event& event : read.events) {
      aggregator.add(event.key, event.value, event.timestamp_ms);
    }
    processed_.fetch_add(read.events.size(), std::memory_order_relaxed);
    skipped_.fetch_add(read.skipped, std::memory_order_relaxed);
    late_.fetch_add(aggregator.lateEvents() - reported_late, std::memory_order_relaxed);
    reported_late = aggregator.lateEvents();

    offset = read.next_offset;
    log_.commit(group_, shard, offset);

    if (read.events.empty() && config_.clock) {
      aggregator.advanceTo(config_.clock());
    }
    publish(aggregator.takeClosed());
    if (read.closed) {
      publish(aggregator.flush());
      return;
    }
  }
}

void StreamProcessor::publish(std::vector<WindowResult> results) {
  if (results.empty()) {
    return;
  }
  const std::scoped_lock lock(results_mutex_);
  for (WindowResult& result : results) {
    results_.push_back(std::move(result));
    if (results_.size() > config_.retained_results) {
      results_.pop_front();
    }
  }
}

std::vector<WindowResult> StreamProcessor::closedWindows() const {
  const std::scoped_lock lock(results_mutex_);
  return {results_.begin(), results_.end()};
}

std::uint64_t StreamProcessor::processedEvents() const {
  return processed_.load(std::memory_order_relaxed);
}

std::uint64_t StreamProcessor::skippedEvents() const {
  return skipped_.load(std::memory_order_relaxed);
}

std::uint64_t StreamProcessor::lateEvents() const { return late_.load(std::memory_order_relaxed); }

}  // namespace stream
