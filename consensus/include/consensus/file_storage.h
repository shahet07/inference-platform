#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "consensus/raft.h"

namespace consensus {

// Keeps a node's term, vote, and log in a directory. With `sync` off, data survives a process crash
// but not a power failure, and Raft's safety guarantees then only hold for process crashes.
class FileStorage final : public Storage {
 public:
  // Loads what is there, removing a half-written record from the end of the log.
  FileStorage(std::filesystem::path dir, bool sync);
  ~FileStorage() override;

  void saveTermAndVote(Term term, NodeId voted_for) override;
  void replaceSuffix(Index keep_up_to, const std::vector<LogEntry>& entries) override;

  [[nodiscard]] Term term() const override { return term_; }
  [[nodiscard]] NodeId votedFor() const override { return voted_for_; }
  [[nodiscard]] std::vector<LogEntry> entries() const override { return entries_; }

  // Parses log records until the first that is incomplete or corrupt, and returns the bytes used.
  // `ends` receives the file position after each record.
  [[nodiscard]] static std::size_t parseLog(std::string_view data, std::vector<LogEntry>& entries,
                                            std::vector<std::uint64_t>& ends);

 private:
  std::filesystem::path dir_;
  bool sync_;
  int log_fd_ = -1;
  Term term_ = 0;
  NodeId voted_for_ = kNoNode;
  std::vector<LogEntry> entries_;
  std::vector<std::uint64_t> ends_;  // File position after each entry, for truncating.
};

}  // namespace consensus
