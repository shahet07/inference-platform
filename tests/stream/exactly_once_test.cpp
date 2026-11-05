#include "stream/exactly_once.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "stream/sharded_log.h"

namespace {

namespace fs = std::filesystem;
using stream::AppendStatus;
using stream::BackpressurePolicy;
using stream::Event;
using stream::ExactlyOnceAggregator;
using stream::FsyncPolicy;
using stream::LogConfig;
using stream::ShardedLog;
using stream::WalConfig;

class ExactlyOnceTest : public testing::Test {
 protected:
  ExactlyOnceTest()
      : dir_(fs::temp_directory_path() /
             ("exactly_once_" +
              std::string(testing::UnitTest::GetInstance()->current_test_info()->name()))) {
    fs::remove_all(dir_);
  }
  ~ExactlyOnceTest() override { fs::remove_all(dir_); }

  [[nodiscard]] LogConfig config() const {
    WalConfig wal;
    wal.dir = dir_ / "log";
    wal.fsync = FsyncPolicy::kNever;
    wal.segment_bytes = 256;
    return {.shards = 1, .shard_capacity = 64, .policy = BackpressurePolicy::kBlock, .wal = wal};
  }

  static Event idempotent(std::uint64_t sequence, std::string key = "k") {
    return {.key = std::move(key),
            .value = 1.0,
            .timestamp_ms = 0,
            .producer_id = 7,
            .sequence = sequence};
  }

  fs::path dir_;
};

TEST_F(ExactlyOnceTest, RetriedAppendIsRecognized) {
  ShardedLog log({.shards = 1, .shard_capacity = 8});
  EXPECT_EQ(log.appendIdempotent(idempotent(1)).status, AppendStatus::kAppended);
  EXPECT_EQ(log.appendIdempotent(idempotent(1)).status, AppendStatus::kDuplicate);
  EXPECT_EQ(log.appendIdempotent(idempotent(2)).status, AppendStatus::kAppended);
  EXPECT_EQ(log.read(0, 0, 10).events.size(), 2U);
  EXPECT_EQ(log.stats().duplicates, 1U);
}

TEST_F(ExactlyOnceTest, ProducersAreTrackedSeparately) {
  ShardedLog log({.shards = 1, .shard_capacity = 8});
  Event other = idempotent(1);
  other.producer_id = 8;
  EXPECT_EQ(log.appendIdempotent(idempotent(1)).status, AppendStatus::kAppended);
  EXPECT_EQ(log.appendIdempotent(other).status, AppendStatus::kAppended);
}

TEST_F(ExactlyOnceTest, IdempotentAppendNeedsAProducer) {
  ShardedLog log({.shards = 1, .shard_capacity = 8});
  EXPECT_THROW((void)log.appendIdempotent({.key = "k"}), std::invalid_argument);
}

TEST_F(ExactlyOnceTest, DeduplicationSurvivesRestart) {
  {
    ShardedLog log(config());
    (void)log.appendIdempotent(idempotent(1));
    (void)log.appendIdempotent(idempotent(2));
  }
  ShardedLog log(config());
  EXPECT_EQ(log.appendIdempotent(idempotent(2)).status, AppendStatus::kDuplicate);
  EXPECT_EQ(log.appendIdempotent(idempotent(3)).status, AppendStatus::kAppended);
}

TEST_F(ExactlyOnceTest, DeduplicationSurvivesTheLogBeingTrimmed) {
  const fs::path first_segment = dir_ / "log" / "shard-0" / "00000000000000000000.log";
  {
    ShardedLog log(config());
    const auto group = log.registerGroup();
    for (std::uint64_t s = 1; s <= 10; ++s) {
      ASSERT_EQ(log.appendIdempotent(idempotent(s)).status, AppendStatus::kAppended);
    }
    // Events from no producer then fill the segments that remain, so that once the early ones are
    // trimmed, nothing on disk but the snapshot can say what producer 7 has sent.
    for (int i = 0; i < 40; ++i) {
      (void)log.append({.key = "k", .value = 1.0});
    }
    log.commit(group, 0, 50);
    for (int attempt = 0; attempt < 5000 && fs::exists(first_segment); ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_FALSE(fs::exists(first_segment)) << "the first segment was never trimmed";
    (void)log.append({.key = "k", .value = 1.0});
  }
  ShardedLog log(config());
  EXPECT_EQ(log.appendIdempotent(idempotent(1)).status, AppendStatus::kDuplicate);
  EXPECT_EQ(log.appendIdempotent(idempotent(10)).status, AppendStatus::kDuplicate);
  EXPECT_EQ(log.appendIdempotent(idempotent(11)).status, AppendStatus::kAppended);
}

TEST_F(ExactlyOnceTest, ReplayAfterCheckpointDoesNotDoubleCount) {
  ShardedLog log({.shards = 1, .shard_capacity = 64});
  const auto group = log.registerGroup();
  for (std::uint64_t s = 1; s <= 10; ++s) {
    (void)log.append(idempotent(s));
  }
  {
    ExactlyOnceAggregator aggregator(log, group, 0, dir_);
    EXPECT_EQ(aggregator.poll(6), 6U);
    aggregator.checkpoint();
    EXPECT_EQ(aggregator.poll(4), 4U);  // Applied, then lost: this instance "crashes" here.
  }
  ExactlyOnceAggregator restored(log, group, 0, dir_);
  EXPECT_EQ(restored.totals().at("k").count, 6U);
  EXPECT_EQ(restored.offset(), 6U);
  EXPECT_EQ(restored.poll(100), 4U);
  EXPECT_EQ(restored.totals().at("k").count, 10U);
  EXPECT_DOUBLE_EQ(restored.totals().at("k").sum, 10.0);
}

TEST_F(ExactlyOnceTest, KeysWithSpacesAndQuotesRoundTrip) {
  ShardedLog log({.shards = 1, .shard_capacity = 64});
  const auto group = log.registerGroup();
  const std::string key = R"(a "quoted" key\with slash)";
  (void)log.append({.key = key, .value = 0.1});
  {
    ExactlyOnceAggregator aggregator(log, group, 0, dir_);
    (void)aggregator.poll(10);
    aggregator.checkpoint();
  }
  const ExactlyOnceAggregator restored(log, group, 0, dir_);
  EXPECT_DOUBLE_EQ(restored.totals().at(key).sum, 0.1);
}

TEST_F(ExactlyOnceTest, DamagedCheckpointIsAnErrorInsteadOfStartingOver) {
  ShardedLog log({.shards = 1, .shard_capacity = 8});
  const auto group = log.registerGroup();
  fs::create_directories(dir_);
  std::ofstream(dir_ / "state-0.checkpoint") << "5 1\n\"k\" 3";
  EXPECT_THROW(ExactlyOnceAggregator(log, group, 0, dir_), std::runtime_error);
}

TEST(ParseCheckpointTest, RejectsIncompleteText) {
  std::uint64_t offset = 0;
  std::map<std::string, ExactlyOnceAggregator::KeyTotals> totals;
  for (const char* text : {"", "5", "5 1", "5 1\n\"k\" 3 1.5", "x 0\nend", "5 99999\nend"}) {
    EXPECT_FALSE(ExactlyOnceAggregator::parseCheckpoint(text, offset, totals)) << text;
  }
  EXPECT_TRUE(ExactlyOnceAggregator::parseCheckpoint("5 1\n\"k\" 3 1.5\nend\n", offset, totals));
  EXPECT_EQ(offset, 5U);
}

}  // namespace
