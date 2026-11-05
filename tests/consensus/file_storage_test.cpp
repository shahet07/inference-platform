#include "consensus/file_storage.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using consensus::FileStorage;
using consensus::LogEntry;

class FileStorageTest : public testing::Test {
 protected:
  FileStorageTest()
      : dir_(fs::temp_directory_path() /
             ("raft_storage_" +
              std::string(testing::UnitTest::GetInstance()->current_test_info()->name()))) {
    fs::remove_all(dir_);
  }
  ~FileStorageTest() override { fs::remove_all(dir_); }

  fs::path dir_;
};

LogEntry entryA() { return {.term = 1, .command = "a"}; }
LogEntry entryB() { return {.term = 1, .command = "bb"}; }
LogEntry entryC() { return {.term = 2, .command = "ccc"}; }

TEST_F(FileStorageTest, TermVoteAndLogSurviveRestart) {
  {
    FileStorage storage(dir_, false);
    storage.saveTermAndVote(7, 2);
    storage.replaceSuffix(0, {entryA(), entryB()});
  }
  const FileStorage storage(dir_, false);
  EXPECT_EQ(storage.term(), 7U);
  EXPECT_EQ(storage.votedFor(), 2U);
  EXPECT_EQ(storage.entries(), (std::vector<LogEntry>{entryA(), entryB()}));
}

TEST_F(FileStorageTest, ReplacingASuffixRemovesEntriesOnDiskToo) {
  {
    FileStorage storage(dir_, false);
    storage.replaceSuffix(0, {entryA(), entryB()});
    storage.replaceSuffix(1, {entryC()});
  }
  const FileStorage storage(dir_, false);
  EXPECT_EQ(storage.entries(), (std::vector<LogEntry>{entryA(), entryC()}));
}

TEST_F(FileStorageTest, AppendsContinueAfterRestart) {
  {
    FileStorage storage(dir_, false);
    storage.replaceSuffix(0, {entryA()});
  }
  {
    FileStorage storage(dir_, false);
    storage.replaceSuffix(1, {entryB()});
  }
  const FileStorage storage(dir_, false);
  EXPECT_EQ(storage.entries(), (std::vector<LogEntry>{entryA(), entryB()}));
}

TEST_F(FileStorageTest, TornRecordIsRemovedAtEveryPossibleCut) {
  {
    FileStorage storage(dir_, false);
    storage.replaceSuffix(0, {entryA(), entryB()});
  }
  const fs::path log = dir_ / "log";
  const auto full = fs::file_size(log);
  const fs::path copy = dir_ / "copy";
  fs::copy_file(log, copy);
  // The last record is 8 bytes of header, 8 of term, and 2 of command.
  for (std::uintmax_t cut = full - 18; cut < full; ++cut) {
    fs::remove(log);
    fs::copy_file(copy, log);
    fs::resize_file(log, cut);
    {
      FileStorage storage(dir_, false);
      EXPECT_EQ(storage.entries(), std::vector<LogEntry>{entryA()}) << cut;
      storage.replaceSuffix(1, {entryC()});  // New data must not land after leftover bytes.
    }
    const FileStorage reopened(dir_, false);
    EXPECT_EQ(reopened.entries(), (std::vector<LogEntry>{entryA(), entryC()})) << cut;
  }
}

TEST_F(FileStorageTest, FlippedBitEndsTheLogThere) {
  {
    FileStorage storage(dir_, false);
    storage.replaceSuffix(0, {entryA(), entryB()});
  }
  {
    std::fstream file(dir_ / "log", std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(-1, std::ios::end);
    file.put('\x7f');
  }
  const FileStorage storage(dir_, false);
  EXPECT_EQ(storage.entries(), std::vector<LogEntry>{entryA()});
}

TEST_F(FileStorageTest, SyncModeWorks) {
  {
    FileStorage storage(dir_, true);
    storage.saveTermAndVote(1, 1);
    storage.replaceSuffix(0, {entryA()});
  }
  EXPECT_EQ(FileStorage(dir_, true).entries().size(), 1U);
}

}  // namespace
