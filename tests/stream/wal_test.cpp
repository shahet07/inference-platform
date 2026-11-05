#include "stream/wal.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "stream/sharded_log.h"

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using stream::AppendStatus;
using stream::BackpressurePolicy;
using stream::Event;
using stream::FsyncPolicy;
using stream::LogConfig;
using stream::ShardedLog;
using stream::WalConfig;

class WalTest : public testing::Test {
 protected:
  WalTest()
      : dir_(fs::temp_directory_path() /
             ("wal_test_" +
              std::string(testing::UnitTest::GetInstance()->current_test_info()->name()))) {
    fs::remove_all(dir_);
  }
  ~WalTest() override { fs::remove_all(dir_); }

  [[nodiscard]] LogConfig config(FsyncPolicy fsync = FsyncPolicy::kNever,
                                 std::size_t segment_bytes = std::size_t{1024} * 1024,
                                 std::shared_ptr<stream::SegmentStore> archive = nullptr) const {
    WalConfig wal;
    wal.dir = dir_;
    wal.fsync = fsync;
    wal.segment_bytes = segment_bytes;
    wal.archive = std::move(archive);
    return {
        .shards = 1, .shard_capacity = 64, .policy = BackpressurePolicy::kDropOldest, .wal = wal};
  }

  [[nodiscard]] fs::path firstSegment() const {
    return dir_ / "shard-0" / "00000000000000000000.log";
  }

  static Event eventFor(double value) { return {.key = "k", .value = value, .timestamp_ms = 7}; }

  fs::path dir_;
};

TEST_F(WalTest, EventsSurviveRestart) {
  {
    ShardedLog log(config());
    (void)log.append(eventFor(1));
    (void)log.append(eventFor(2));
  }
  ShardedLog log(config());
  const auto read = log.read(0, 0, 10);
  ASSERT_EQ(read.events.size(), 2U);
  EXPECT_EQ(read.events[0].value, 1);
  EXPECT_EQ(read.events[0].key, "k");
  EXPECT_EQ(read.events[0].timestamp_ms, 7);
  EXPECT_EQ(read.events[1].value, 2);
  EXPECT_EQ(log.stats().recovered, 2U);
}

TEST_F(WalTest, OffsetsContinueAfterRestart) {
  {
    ShardedLog log(config());
    (void)log.append(eventFor(1));
  }
  ShardedLog log(config());
  EXPECT_EQ(log.append(eventFor(2)).offset, 1U);
}

TEST_F(WalTest, EveryFsyncPolicyPersists) {
  for (const FsyncPolicy policy :
       {FsyncPolicy::kEveryAppend, FsyncPolicy::kInterval, FsyncPolicy::kNever}) {
    fs::remove_all(dir_);
    {
      ShardedLog log(config(policy));
      for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(log.append(eventFor(i)).status, AppendStatus::kAppended);
      }
    }
    ShardedLog log(config(policy));
    EXPECT_EQ(log.read(0, 0, 100).events.size(), 20U);
  }
}

TEST_F(WalTest, AppendNoWaitStillPersistsOnCleanShutdown) {
  {
    ShardedLog log(config(FsyncPolicy::kEveryAppend));
    for (int i = 0; i < 50; ++i) {
      ASSERT_EQ(log.appendNoWait(eventFor(i)).status, AppendStatus::kAppended);
    }
  }
  ShardedLog log(config());
  EXPECT_EQ(log.read(0, 0, 100).events.size(), 50U);
}

TEST_F(WalTest, TornTailIsRemovedAndEarlierEventsKept) {
  {
    ShardedLog log(config());
    (void)log.append(eventFor(1));
    (void)log.append(eventFor(2));
  }
  fs::resize_file(firstSegment(), fs::file_size(firstSegment()) - 3);  // Half-written last record.

  ShardedLog log(config());

  EXPECT_EQ(log.read(0, 0, 10).events.size(), 1U);
  EXPECT_GT(log.stats().truncated_bytes, 0U);
  EXPECT_EQ(log.append(eventFor(3)).offset, 1U);  // Appends follow the last valid record.
}

TEST_F(WalTest, FlippedBitInLastRecordIsRemoved) {
  {
    ShardedLog log(config());
    (void)log.append(eventFor(1));
    (void)log.append(eventFor(2));
  }
  {
    std::fstream file(firstSegment(), std::ios::in | std::ios::out | std::ios::binary);
    file.seekg(-1, std::ios::end);
    const char byte = static_cast<char>(file.get());
    file.seekp(-1, std::ios::end);
    file.put(static_cast<char>(static_cast<unsigned char>(byte) ^ 0x01U));
  }
  ShardedLog log(config());
  EXPECT_EQ(log.read(0, 0, 10).events.size(), 1U);
}

TEST_F(WalTest, CorruptionBeforeTheLastSegmentIsAnError) {
  {
    ShardedLog log(config(FsyncPolicy::kNever, 32));  // Rolls after every write.
    for (int i = 0; i < 4; ++i) {
      (void)log.append(eventFor(i));
    }
  }
  {
    std::fstream file(firstSegment(), std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(10);
    file.put('\x7f');
  }
  EXPECT_THROW(ShardedLog log(config()), std::runtime_error);
}

TEST_F(WalTest, RecoveryKeepsOnlyTheNewestEventsInMemory) {
  {
    ShardedLog log(config());
    for (int i = 0; i < 200; ++i) {
      (void)log.append(eventFor(i));
    }
  }
  ShardedLog log(config());
  const auto read = log.read(0, 0, 1000);
  EXPECT_EQ(read.skipped, 136U);
  ASSERT_EQ(read.events.size(), 64U);
  EXPECT_EQ(read.events.back().value, 199);
}

TEST_F(WalTest, DurableGroupResumesWhereItCommitted) {
  {
    ShardedLog log(config());
    const auto group = log.registerDurableGroup("analytics");
    for (int i = 0; i < 5; ++i) {
      (void)log.append(eventFor(i));
    }
    log.commit(group, 0, 3);
  }
  ShardedLog log(config());
  const auto group = log.registerDurableGroup("analytics");
  EXPECT_EQ(log.committed(group, 0), 3U);
}

TEST_F(WalTest, CommittedSealedSegmentsLeaveDisk) {
  ShardedLog log(config(FsyncPolicy::kNever, 32));
  const auto group = log.registerGroup();
  for (int i = 0; i < 8; ++i) {
    (void)log.append(eventFor(i));
  }
  log.commit(group, 0, 8);
  for (int attempt = 0; attempt < 2000 && fs::exists(firstSegment()); ++attempt) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_FALSE(fs::exists(firstSegment()));
}

TEST_F(WalTest, ArchivedSegmentsCanStillBeRead) {
  const auto archive = std::make_shared<stream::DirectorySegmentStore>(dir_ / "archive");
  ShardedLog log(config(FsyncPolicy::kNever, 32, archive));
  const auto group = log.registerGroup();
  for (int i = 0; i < 8; ++i) {
    (void)log.append(eventFor(i));
  }
  log.commit(group, 0, 8);
  for (int attempt = 0; attempt < 2000 && fs::exists(firstSegment()); ++attempt) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_FALSE(archive->list("shard-0-").empty());
  EXPECT_FALSE(fs::exists(firstSegment()));

  // One archived segment holds at least the first event; later ones may still be on their way.
  const auto read = log.read(0, 0, 1);

  ASSERT_EQ(read.events.size(), 1U);
  EXPECT_EQ(read.events[0].value, 0);
  EXPECT_EQ(read.next_offset, 1U);
  EXPECT_EQ(read.skipped, 0U);
}

TEST_F(WalTest, RandomCutsNeverLoseEarlierRecords) {
  // Cuts the file at every byte of the last record, as a crash during write could.
  {
    ShardedLog log(config());
    for (int i = 0; i < 3; ++i) {
      (void)log.append(eventFor(i));
    }
  }
  const auto full = fs::file_size(firstSegment());
  const fs::path copy = dir_ / "copy";
  fs::copy_file(firstSegment(), copy);
  for (std::uintmax_t cut = full - 37; cut < full; ++cut) {
    fs::remove(firstSegment());
    fs::copy_file(copy, firstSegment());
    fs::resize_file(firstSegment(), cut);
    ShardedLog log(config());
    const auto read = log.read(0, 0, 10);
    EXPECT_GE(read.events.size(), 2U) << cut;
  }
}

}  // namespace
