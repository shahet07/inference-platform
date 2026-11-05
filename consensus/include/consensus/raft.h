#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace consensus {

using NodeId = std::uint32_t;
using Term = std::uint64_t;
using Index = std::uint64_t;  // Log positions start at 1. 0 means "before the log".

inline constexpr NodeId kNoNode = 0;  // Node ids must not be 0.

struct LogEntry {
  Term term = 0;
  std::string command;

  bool operator==(const LogEntry&) const = default;
};

struct RequestVote {
  Term term = 0;
  Index last_log_index = 0;
  Term last_log_term = 0;
};

struct VoteReply {
  Term term = 0;
  bool granted = false;
};

struct AppendEntries {
  Term term = 0;
  Index prev_index = 0;
  Term prev_term = 0;
  std::vector<LogEntry> entries;
  Index leader_commit = 0;
};

struct AppendReply {
  Term term = 0;
  bool success = false;
  Index match_index = 0;  // On success, the last entry known to match the leader's.
  Index retry_from = 0;   // On failure, where the leader should try next.
};

struct Message {
  NodeId from = kNoNode;
  NodeId to = kNoNode;
  std::variant<RequestVote, VoteReply, AppendEntries, AppendReply> body;
};

// What must survive a restart. Each call must be durable when it returns, because the node acts on
// it straight away: replying to a vote before saving it could let a restarted node vote twice in a
// term.
class Storage {
 public:
  Storage() = default;
  virtual ~Storage() = default;
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;

  virtual void saveTermAndVote(Term term, NodeId voted_for) = 0;
  // Removes every entry after `keep_up_to`, then adds `entries`.
  virtual void replaceSuffix(Index keep_up_to, const std::vector<LogEntry>& entries) = 0;

  [[nodiscard]] virtual Term term() const = 0;
  [[nodiscard]] virtual NodeId votedFor() const = 0;
  [[nodiscard]] virtual std::vector<LogEntry> entries() const = 0;
};

// Keeps everything in memory. It models a disk that survives a node's restart, for tests.
class MemoryStorage final : public Storage {
 public:
  void saveTermAndVote(Term term, NodeId voted_for) override;
  void replaceSuffix(Index keep_up_to, const std::vector<LogEntry>& entries) override;
  [[nodiscard]] Term term() const override { return term_; }
  [[nodiscard]] NodeId votedFor() const override { return voted_for_; }
  [[nodiscard]] std::vector<LogEntry> entries() const override { return entries_; }

 private:
  Term term_ = 0;
  NodeId voted_for_ = kNoNode;
  std::vector<LogEntry> entries_;
};

struct RaftConfig {
  // In ticks. Timeouts are random within this range, which keeps candidates from repeatedly
  // splitting the vote. The range should be well above the time a message takes.
  std::uint32_t election_timeout_min = 10;
  std::uint32_t election_timeout_max = 20;
  std::uint32_t heartbeat_interval = 2;
  std::size_t max_entries_per_message = 256;
};

enum class Role : std::uint8_t { kFollower, kCandidate, kLeader };

// The Raft consensus algorithm: leader election and log replication. The class has no threads,
// clock, or network. Time passes through tick(), and messages go in through receive() and out
// through takeOutbox(), which makes every behavior reproducible in tests.
//
// Not included: log compaction with snapshots, so the log grows without bound, and changing the set
// of nodes while running. Not thread-safe.
class RaftNode {
 public:
  RaftNode(NodeId id, std::vector<NodeId> peers, Storage& storage, const RaftConfig& config,
           std::uint64_t seed);

  void tick();
  void receive(const Message& message);

  struct Proposal {
    Index index = 0;  // Where the first command landed.
    Term term = 0;    // A command is committed only if this term's entry is the one at its index.
  };
  // Appends commands to the log, in one write. Returns nullopt if this node is not the leader. A
  // proposal is not safe until it comes out of takeCommitted(): a leader can lose it.
  [[nodiscard]] std::optional<Proposal> propose(const std::vector<std::string>& commands);

  [[nodiscard]] std::vector<Message> takeOutbox();
  // Newly committed entries, in order, each exactly once. A restarted node delivers them all again.
  [[nodiscard]] std::vector<std::pair<Index, LogEntry>> takeCommitted();

  [[nodiscard]] Role role() const { return role_; }
  [[nodiscard]] Term term() const { return term_; }
  [[nodiscard]] NodeId id() const { return id_; }
  [[nodiscard]] NodeId leaderHint() const { return leader_; }
  [[nodiscard]] Index commitIndex() const { return commit_index_; }
  [[nodiscard]] Index lastIndex() const { return static_cast<Index>(log_.size()); }
  [[nodiscard]] const std::vector<LogEntry>& log() const { return log_; }

 private:
  void startElection();
  void becomeFollower(Term term, NodeId leader);
  void becomeLeader();
  void observeTerm(Term term);
  void sendAppend(NodeId peer);
  void broadcastAppends();
  void advanceCommit();
  void resetElectionTimer();
  void persistTermAndVote();
  [[nodiscard]] Term termAt(Index index) const;

  void handle(NodeId from, const RequestVote& message);
  void handle(NodeId from, const VoteReply& message);
  void handle(NodeId from, const AppendEntries& message);
  void handle(NodeId from, const AppendReply& message);

  NodeId id_;
  std::vector<NodeId> peers_;
  Storage& storage_;
  RaftConfig config_;
  std::mt19937_64 rng_;

  Term term_ = 0;
  NodeId voted_for_ = kNoNode;
  std::vector<LogEntry> log_;  // log_[i] holds the entry at index i + 1.

  Role role_ = Role::kFollower;
  NodeId leader_ = kNoNode;
  Index commit_index_ = 0;
  Index delivered_up_to_ = 0;
  std::uint32_t ticks_since_heard_ = 0;
  std::uint32_t election_timeout_ = 0;
  std::uint32_t ticks_since_heartbeat_ = 0;
  std::map<NodeId, bool> votes_;
  std::map<NodeId, Index> next_index_;
  std::map<NodeId, Index> match_index_;
  std::vector<Message> outbox_;
};

}  // namespace consensus
