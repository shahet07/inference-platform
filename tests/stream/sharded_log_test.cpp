#include "stream/sharded_log.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using stream::AppendStatus;
using stream::BackpressurePolicy;
using stream::Event;
using stream::LogConfig;
using stream::ShardedLog;

Event eventFor(double value, std::string key = "k") {
  return {.key = std::move(key), .value = value, .timestamp_ms = 0};
}

LogConfig oneShard(std::size_t capacity, BackpressurePolicy policy) {
  return {.shards = 1, .shard_capacity = capacity, .policy = policy};
}

TEST(ShardedLogTest, ReadReturnsEventsInAppendOrder) {
  ShardedLog log(oneShard(8, BackpressurePolicy::kDropNewest));
  EXPECT_EQ(log.append(eventFor(1)).offset, 0U);
  EXPECT_EQ(log.append(eventFor(2)).offset, 1U);

  const auto read = log.read(0, 0, 10);

  ASSERT_EQ(read.events.size(), 2U);
  EXPECT_EQ(read.events[0].value, 1);
  EXPECT_EQ(read.events[1].value, 2);
  EXPECT_EQ(read.next_offset, 2U);
}

TEST(ShardedLogTest, ReadHonorsMaxEventsAndOffset) {
  ShardedLog log(oneShard(8, BackpressurePolicy::kDropNewest));
  for (int i = 0; i < 5; ++i) {
    (void)log.append(eventFor(i));
  }

  const auto read = log.read(0, 1, 2);

  ASSERT_EQ(read.events.size(), 2U);
  EXPECT_EQ(read.events[0].value, 1);
  EXPECT_EQ(read.next_offset, 3U);
}

TEST(ShardedLogTest, SameKeyAlwaysMapsToSameShard) {
  ShardedLog log({.shards = 16, .shard_capacity = 8, .policy = BackpressurePolicy::kDropNewest});
  const auto first = log.append(eventFor(1, "client-a"));
  const auto second = log.append(eventFor(2, "client-a"));
  EXPECT_EQ(first.shard, second.shard);
  EXPECT_EQ(second.offset, first.offset + 1);
}

TEST(ShardedLogTest, KeysSpreadAcrossShards) {
  ShardedLog log({.shards = 4, .shard_capacity = 8, .policy = BackpressurePolicy::kDropNewest});
  std::vector<int> hits(4);
  for (int i = 0; i < 1000; ++i) {
    ++hits[log.shardFor("key-" + std::to_string(i))];
  }
  for (const int count : hits) {
    EXPECT_GT(count, 100);
  }
}

TEST(BackpressureTest, DropNewestRejectsWhenFullAndKeepsHistory) {
  ShardedLog log(oneShard(2, BackpressurePolicy::kDropNewest));
  (void)log.append(eventFor(1));
  (void)log.append(eventFor(2));

  EXPECT_EQ(log.append(eventFor(3)).status, AppendStatus::kDropped);

  const auto read = log.read(0, 0, 10);
  ASSERT_EQ(read.events.size(), 2U);
  EXPECT_EQ(read.events[0].value, 1);
  EXPECT_EQ(log.stats().dropped_newest, 1U);
  EXPECT_EQ(log.stats().retained, 2U);
}

TEST(BackpressureTest, DropOldestEvictsAndReportsSkippedEvents) {
  ShardedLog log(oneShard(2, BackpressurePolicy::kDropOldest));
  for (int i = 1; i <= 5; ++i) {
    EXPECT_EQ(log.append(eventFor(i)).status, AppendStatus::kAppended);
  }

  const auto read = log.read(0, 0, 10);

  EXPECT_EQ(read.skipped, 3U);
  ASSERT_EQ(read.events.size(), 2U);
  EXPECT_EQ(read.events[0].value, 4);
  EXPECT_EQ(read.events[1].value, 5);
  EXPECT_EQ(log.stats().retained, 2U);
}

TEST(BackpressureTest, BlockWaitsUntilConsumerCommits) {
  ShardedLog log(oneShard(1, BackpressurePolicy::kBlock));
  const auto group = log.registerGroup();
  (void)log.append(eventFor(1));

  std::atomic<bool> finished{false};
  std::thread producer([&] {
    (void)log.append(eventFor(2));
    finished = true;
  });
  std::this_thread::sleep_for(20ms);
  EXPECT_FALSE(finished);

  log.commit(group, 0, 1);
  producer.join();
  EXPECT_TRUE(finished);
  EXPECT_EQ(log.stats().blocked_appends, 1U);
}

TEST(BackpressureTest, CloseReleasesBlockedProducer) {
  ShardedLog log(oneShard(1, BackpressurePolicy::kBlock));
  (void)log.append(eventFor(1));

  AppendStatus status = AppendStatus::kAppended;
  std::thread producer([&] { status = log.append(eventFor(2)).status; });
  std::this_thread::sleep_for(5ms);
  log.close();
  producer.join();

  EXPECT_EQ(status, AppendStatus::kClosed);
}

TEST(BackpressureTest, SlowConsumerNeverGrowsLogPastCapacity) {
  constexpr std::size_t kCapacity = 16;
  ShardedLog log(oneShard(kCapacity, BackpressurePolicy::kDropNewest));
  const auto group = log.registerGroup();

  std::atomic<bool> done{false};
  std::thread consumer([&] {
    std::uint64_t offset = 0;
    while (!done) {
      const auto read = log.read(0, offset, 1);
      offset = read.next_offset;
      log.commit(group, 0, offset);
      std::this_thread::sleep_for(100us);
    }
  });
  for (int i = 0; i < 50000; ++i) {
    (void)log.append(eventFor(i));
    ASSERT_LE(log.stats().retained, kCapacity);
  }
  done = true;
  consumer.join();

  const auto stats = log.stats();
  EXPECT_GT(stats.dropped_newest, 0U);
  EXPECT_EQ(stats.appended + stats.dropped_newest, 50000U);
}

TEST(CommitTest, CommitsTrimOnlyPastTheSlowestGroup) {
  ShardedLog log(oneShard(2, BackpressurePolicy::kDropNewest));
  const auto fast = log.registerGroup();
  const auto slow = log.registerGroup();
  (void)log.append(eventFor(1));
  (void)log.append(eventFor(2));

  log.commit(fast, 0, 2);
  EXPECT_EQ(log.append(eventFor(3)).status, AppendStatus::kDropped);

  log.commit(slow, 0, 1);
  EXPECT_EQ(log.append(eventFor(4)).status, AppendStatus::kAppended);
}

TEST(CommitTest, CommitsDoNotMoveBackwardOrPastTheEnd) {
  ShardedLog log(oneShard(4, BackpressurePolicy::kDropNewest));
  const auto group = log.registerGroup();
  (void)log.append(eventFor(1));
  (void)log.append(eventFor(2));

  log.commit(group, 0, 99);
  EXPECT_EQ(log.committed(group, 0), 2U);
  log.commit(group, 0, 1);
  EXPECT_EQ(log.committed(group, 0), 2U);
}

TEST(CommitTest, UnregisteringSlowGroupReleasesSpace) {
  ShardedLog log(oneShard(1, BackpressurePolicy::kDropNewest));
  const auto stuck = log.registerGroup();
  const auto active = log.registerGroup();
  (void)log.append(eventFor(1));
  log.commit(active, 0, 1);
  EXPECT_EQ(log.append(eventFor(2)).status, AppendStatus::kDropped);

  log.unregisterGroup(stuck);

  EXPECT_EQ(log.append(eventFor(3)).status, AppendStatus::kAppended);
}

TEST(ShardedLogTest, ReadWaitsForEvent) {
  ShardedLog log(oneShard(4, BackpressurePolicy::kDropNewest));
  std::thread producer([&] {
    std::this_thread::sleep_for(5ms);
    (void)log.append(eventFor(7));
  });
  const auto read = log.read(0, 0, 1, 10s);
  producer.join();
  ASSERT_EQ(read.events.size(), 1U);
  EXPECT_EQ(read.events[0].value, 7);
}

TEST(ShardedLogTest, ReadWaitsForMinimumBatch) {
  ShardedLog log(oneShard(8, BackpressurePolicy::kDropNewest));
  std::thread producer([&] {
    for (int i = 0; i < 3; ++i) {
      std::this_thread::sleep_for(2ms);
      (void)log.append(eventFor(i));
    }
  });
  const auto read = log.read(0, 0, 10, 10s, 3);
  producer.join();
  EXPECT_EQ(read.events.size(), 3U);
}

TEST(ShardedLogTest, ReadReturnsPartialBatchAtTimeout) {
  ShardedLog log(oneShard(8, BackpressurePolicy::kDropNewest));
  (void)log.append(eventFor(1));
  const auto read = log.read(0, 0, 10, 5ms, 4);
  EXPECT_EQ(read.events.size(), 1U);
}

TEST(ShardedLogTest, MinimumBatchLargerThanCapacityStillWakes) {
  ShardedLog log(oneShard(2, BackpressurePolicy::kDropNewest));
  (void)log.append(eventFor(1));
  (void)log.append(eventFor(2));
  // Could never reach 100 events, so this must not wait the full 10 seconds.
  const auto started = std::chrono::steady_clock::now();
  const auto read = log.read(0, 0, 100, 10s, 100);
  EXPECT_EQ(read.events.size(), 2U);
  EXPECT_LT(std::chrono::steady_clock::now() - started, 5s);
}

TEST(ShardedLogTest, ClosedLogDrainsThenReportsClosed) {
  ShardedLog log(oneShard(4, BackpressurePolicy::kDropNewest));
  (void)log.append(eventFor(1));
  log.close();

  EXPECT_EQ(log.append(eventFor(2)).status, AppendStatus::kClosed);
  const auto read = log.read(0, 0, 10, 10s);
  EXPECT_EQ(read.events.size(), 1U);
  EXPECT_TRUE(read.closed);
}

TEST(ShardedLogTest, ConcurrentProducersAssignUniqueOffsets) {
  ShardedLog log({.shards = 4, .shard_capacity = 64, .policy = BackpressurePolicy::kBlock});
  const auto group = log.registerGroup();
  constexpr int kPerProducer = 5000;
  std::atomic<std::uint64_t> consumed{0};
  std::vector<std::thread> consumers;
  consumers.reserve(4);
  for (std::size_t shard = 0; shard < 4; ++shard) {
    consumers.emplace_back([&, shard] {
      std::uint64_t offset = 0;
      while (true) {
        const auto read = log.read(shard, offset, 16, 5ms);
        consumed += read.events.size();
        offset = read.next_offset;
        log.commit(group, shard, offset);
        if (read.closed) {
          return;
        }
      }
    });
  }
  std::vector<std::thread> producers;
  producers.reserve(4);
  for (int p = 0; p < 4; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kPerProducer; ++i) {
        (void)log.append(eventFor(i, "k" + std::to_string((p * 1000) + (i % 50))));
      }
    });
  }
  for (std::thread& t : producers) {
    t.join();
  }
  log.close();
  for (std::thread& t : consumers) {
    t.join();
  }
  EXPECT_EQ(consumed, 4U * kPerProducer);
}

}  // namespace
