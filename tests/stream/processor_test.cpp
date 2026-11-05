#include "stream/processor.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;
using stream::BackpressurePolicy;
using stream::ProcessorConfig;
using stream::ShardedLog;
using stream::StreamProcessor;

TEST(StreamProcessorTest, SumsMatchAcrossShardsAfterStop) {
  ShardedLog log({.shards = 4, .shard_capacity = 64, .policy = BackpressurePolicy::kBlock});
  std::map<std::string, std::uint64_t> counts;
  {
    StreamProcessor processor(log, ProcessorConfig{.window = {.window_ms = 1000}});
    for (int i = 0; i < 10000; ++i) {
      (void)log.append({.key = "k" + std::to_string(i % 10), .value = 1.0, .timestamp_ms = i / 10});
    }
    processor.stop();
    EXPECT_EQ(processor.processedEvents(), 10000U);
    for (const auto& window : processor.closedWindows()) {
      counts[window.key] += window.count;
    }
  }
  ASSERT_EQ(counts.size(), 10U);
  for (const auto& [key, count] : counts) {
    EXPECT_EQ(count, 1000U) << key;
  }
}

TEST(StreamProcessorTest, ClockClosesWindowsOnIdleShards) {
  ShardedLog log({.shards = 2, .shard_capacity = 8, .policy = BackpressurePolicy::kDropNewest});
  std::atomic<std::int64_t> now{0};
  StreamProcessor processor(
      log, ProcessorConfig{.window = {.window_ms = 100}, .clock = [&now] { return now.load(); }});
  (void)log.append({.key = "a", .value = 1.0, .timestamp_ms = 10});
  now = 200;

  bool closed = false;
  for (int attempt = 0; attempt < 500 && !closed; ++attempt) {
    closed = !processor.closedWindows().empty();
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_TRUE(closed);
}

TEST(StreamProcessorTest, CommitsFreeSpaceForProducers) {
  ShardedLog log({.shards = 1, .shard_capacity = 4, .policy = BackpressurePolicy::kBlock});
  StreamProcessor processor(log, ProcessorConfig{.window = {.window_ms = 1000}});
  // Far more events than fit, so this only finishes if the processor keeps committing.
  for (int i = 0; i < 2000; ++i) {
    ASSERT_EQ(log.append({.key = "a", .value = 1.0, .timestamp_ms = i}).status,
              stream::AppendStatus::kAppended);
  }
  processor.stop();
  EXPECT_EQ(processor.processedEvents(), 2000U);
}

}  // namespace
