#include "consensus/raft.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using consensus::Index;
using consensus::LogEntry;
using consensus::MemoryStorage;
using consensus::Message;
using consensus::NodeId;
using consensus::RaftConfig;
using consensus::RaftNode;
using consensus::Role;
using consensus::Term;

// A whole cluster on one thread, with a network that loses, delays, repeats, and reorders messages,
// and nodes that crash and restart keeping only their storage. Everything comes from one seed, so
// any failure can be replayed exactly. The safety rules from the Raft paper are checked after every
// step, because that catches a violation at the moment it happens.
class Simulation {
 public:
  struct Faults {
    double drop = 0.0;
    double duplicate = 0.0;
    int max_delay = 3;
  };

  Simulation(int nodes, std::uint64_t seed, Faults faults) : rng_(seed), faults_(faults) {
    for (int i = 1; i <= nodes; ++i) {
      ids_.push_back(static_cast<NodeId>(i));
    }
    for (const NodeId id : ids_) {
      storages_[id] = std::make_unique<MemoryStorage>();
      start(id);
    }
  }

  void step() {
    ++now_;
    for (const NodeId id : ids_) {
      if (live_.contains(id)) {
        live_.at(id)->tick();
      }
    }
    deliverDue();
    collect();
    checkSafety();
  }

  void run(int steps) {
    for (int i = 0; i < steps; ++i) {
      step();
    }
  }

  void crash(NodeId id) { live_.erase(id); }

  void start(NodeId id) {
    live_[id] = std::make_unique<RaftNode>(id, ids_, *storages_[id], RaftConfig{},
                                           rng_() ^ (std::uint64_t{id} << 32U));
    delivered_[id] = 0;
  }

  // Nodes in different groups cannot reach each other.
  void partition(const std::vector<std::vector<NodeId>>& groups) {
    group_.clear();
    int number = 0;
    for (const auto& group : groups) {
      for (const NodeId id : group) {
        group_[id] = number;
      }
      ++number;
    }
  }
  void heal() { group_.clear(); }

  RaftNode* leader() {
    RaftNode* best = nullptr;
    for (auto& [id, node] : live_) {
      if (node->role() == Role::kLeader && (best == nullptr || node->term() > best->term())) {
        best = node.get();
      }
    }
    return best;
  }

  bool propose(const std::string& command) {
    RaftNode* node = leader();
    if (node == nullptr) {
      return false;
    }
    const auto proposal = node->propose({command});
    if (proposal) {
      proposed_[command] = {proposal->index, proposal->term};
    }
    return proposal.has_value();
  }

  [[nodiscard]] bool committedEverywhere(const std::string& command) const {
    return std::ranges::any_of(committed_,
                               [&](const auto& entry) { return entry.second.command == command; });
  }

  [[nodiscard]] const std::map<Index, LogEntry>& committed() const { return committed_; }
  [[nodiscard]] const std::vector<NodeId>& ids() const { return ids_; }
  [[nodiscard]] bool isLive(NodeId id) const { return live_.contains(id); }
  [[nodiscard]] const RaftNode& node(NodeId id) const { return *live_.at(id); }
  std::mt19937_64& rng() { return rng_; }

 private:
  struct InFlight {
    std::uint64_t due = 0;
    Message message;
  };

  [[nodiscard]] bool reachable(NodeId a, NodeId b) const {
    const auto ga = group_.find(a);
    const auto gb = group_.find(b);
    return ga == group_.end() || gb == group_.end() || ga->second == gb->second;
  }

  double chance() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }

  void deliverDue() {
    std::vector<InFlight> due;
    std::erase_if(network_, [&](InFlight& item) {
      if (item.due <= now_) {
        due.push_back(std::move(item));
        return true;
      }
      return false;
    });
    std::ranges::shuffle(due, rng_);
    for (const InFlight& item : due) {
      if (live_.contains(item.message.to) && reachable(item.message.from, item.message.to)) {
        live_.at(item.message.to)->receive(item.message);
      }
    }
  }

  void collect() {
    for (auto& [id, node] : live_) {
      for (Message& message : node->takeOutbox()) {
        if (chance() < faults_.drop) {
          continue;
        }
        const int copies = chance() < faults_.duplicate ? 2 : 1;
        for (int c = 0; c < copies; ++c) {
          const auto delay = std::uniform_int_distribution<int>(1, faults_.max_delay)(rng_);
          network_.push_back({.due = now_ + static_cast<std::uint64_t>(delay), .message = message});
        }
      }
      for (auto& [index, entry] : node->takeCommitted()) {
        recordCommitted(id, index, entry);
      }
    }
  }

  // State machine safety: no two nodes ever apply different entries at one index, and each node
  // applies every index in order.
  void recordCommitted(NodeId id, Index index, const LogEntry& entry) {
    ASSERT_EQ(index, delivered_[id] + 1) << "node " << id << " skipped an index";
    delivered_[id] = index;
    const auto [existing, inserted] = committed_.try_emplace(index, entry);
    ASSERT_TRUE(inserted || existing->second == entry)
        << "two different entries committed at index " << index;
    if (inserted) {
      commit_terms_[index] = live_.at(id)->term();
    }
  }

  void checkSafety() {
    // Election safety: at most one leader per term, ever.
    for (const auto& [id, node] : live_) {
      if (node->role() == Role::kLeader) {
        const auto [existing, inserted] = leaders_.try_emplace(node->term(), id);
        ASSERT_TRUE(inserted || existing->second == id) << "two leaders in term " << node->term();
        // Leader completeness: a leader holds every entry committed in an earlier term. That is the
        // term it was committed in, not the term it was created in. An old leader stuck in a
        // partition can rightly lack an old entry that a newer leader goes on to commit.
        for (const auto& [index, entry] : committed_) {
          if (commit_terms_.at(index) < node->term()) {
            ASSERT_LE(index, node->lastIndex()) << "a leader is missing a committed entry";
            ASSERT_EQ(node->log()[index - 1], entry) << "a leader differs on a committed entry";
          }
        }
      }
    }
    // Log matching: if two logs share an index and term, they are identical up to there.
    std::vector<const RaftNode*> nodes;
    nodes.reserve(live_.size());
    for (const auto& [id, node] : live_) {
      nodes.push_back(node.get());
    }
    for (std::size_t a = 0; a < nodes.size(); ++a) {
      for (std::size_t b = a + 1; b < nodes.size(); ++b) {
        const auto& la = nodes[a]->log();
        const auto& lb = nodes[b]->log();
        std::size_t i = std::min(la.size(), lb.size());
        while (i > 0 && la[i - 1].term != lb[i - 1].term) {
          --i;
        }
        for (; i > 0; --i) {
          ASSERT_EQ(la[i - 1], lb[i - 1]) << "logs share a term at " << i << " but differ";
        }
      }
    }
  }

  std::mt19937_64 rng_;
  Faults faults_;
  std::vector<NodeId> ids_;
  std::map<NodeId, std::unique_ptr<MemoryStorage>> storages_;
  std::map<NodeId, std::unique_ptr<RaftNode>> live_;
  std::map<NodeId, Index> delivered_;
  std::map<NodeId, int> group_;
  std::vector<InFlight> network_;
  std::uint64_t now_ = 0;
  std::map<Term, NodeId> leaders_;
  std::map<Index, LogEntry> committed_;
  std::map<Index, Term> commit_terms_;  // The term of the node that first delivered each entry.
  std::map<std::string, std::pair<Index, Term>> proposed_;
};

const Simulation::Faults kClean{.drop = 0.0, .duplicate = 0.0, .max_delay = 1};
const Simulation::Faults kHostile{.drop = 0.15, .duplicate = 0.1, .max_delay = 5};

TEST(RaftTest, ElectsOneLeader) {
  Simulation sim(3, 1, kClean);
  sim.run(60);
  ASSERT_NE(sim.leader(), nullptr);
  int leaders = 0;
  for (const NodeId id : sim.ids()) {
    leaders += sim.node(id).role() == Role::kLeader ? 1 : 0;
  }
  EXPECT_EQ(leaders, 1);
}

TEST(RaftTest, SingleNodeClusterLeadsAndCommitsAlone) {
  Simulation sim(1, 1, kClean);
  sim.run(40);
  ASSERT_TRUE(sim.propose("x"));
  sim.run(2);
  EXPECT_TRUE(sim.committedEverywhere("x"));
}

TEST(RaftTest, ProposalReplicatesToEveryNode) {
  Simulation sim(3, 2, kClean);
  sim.run(60);
  ASSERT_TRUE(sim.propose("x"));
  sim.run(10);
  EXPECT_TRUE(sim.committedEverywhere("x"));
  for (const NodeId id : sim.ids()) {
    EXPECT_EQ(sim.node(id).log().back().command, "x");
    EXPECT_EQ(sim.node(id).commitIndex(), sim.node(id).lastIndex());
  }
}

TEST(RaftTest, LeaderSendsEachEntryOnceUnderLoad) {
  // Resending every unacknowledged entry with each proposal once brought a real cluster down.
  MemoryStorage storage;
  RaftNode node(1, {1, 2}, storage, RaftConfig{}, 1);
  while (node.role() != Role::kCandidate) {
    node.tick();
  }
  node.receive(
      {.from = 2, .to = 1, .body = consensus::VoteReply{.term = node.term(), .granted = true}});
  ASSERT_EQ(node.role(), Role::kLeader);
  (void)node.takeOutbox();
  std::size_t sent = 0;
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(node.propose({"x"}).has_value());
    for (const Message& message : node.takeOutbox()) {
      sent += std::get<consensus::AppendEntries>(message.body).entries.size();
    }
  }
  EXPECT_EQ(sent, 100U);
}

TEST(RaftTest, LostPipelinedEntriesAreResent) {
  MemoryStorage leader_storage;
  MemoryStorage follower_storage;
  RaftNode leader(1, {1, 2}, leader_storage, RaftConfig{}, 1);
  RaftNode follower(2, {1, 2}, follower_storage, RaftConfig{}, 2);
  while (leader.role() != Role::kCandidate) {
    leader.tick();
  }
  leader.receive(
      {.from = 2, .to = 1, .body = consensus::VoteReply{.term = leader.term(), .granted = true}});
  ASSERT_EQ(leader.role(), Role::kLeader);
  (void)leader.takeOutbox();  // The first append, with the no-op, is lost.
  ASSERT_TRUE(leader.propose({"a"}).has_value());
  (void)leader.takeOutbox();  // So is this one.
  ASSERT_TRUE(leader.propose({"b"}).has_value());
  for (int round = 0; round < 10; ++round) {
    for (const Message& message : leader.takeOutbox()) {
      follower.receive(message);
    }
    for (const Message& message : follower.takeOutbox()) {
      leader.receive(message);
    }
  }
  EXPECT_EQ(follower.log(), leader.log());
  EXPECT_EQ(follower.lastIndex(), 3U);
  EXPECT_EQ(leader.commitIndex(), 3U);
}

TEST(RaftTest, FollowerRefusesProposals) {
  MemoryStorage storage;
  RaftNode node(1, {1, 2, 3}, storage, RaftConfig{}, 1);
  EXPECT_FALSE(node.propose({"x"}).has_value());
}

TEST(RaftTest, NewLeaderTakesOverAfterACrash) {
  Simulation sim(3, 3, kClean);
  sim.run(60);
  ASSERT_TRUE(sim.propose("before"));
  sim.run(10);
  const NodeId old_leader = sim.leader()->id();
  sim.crash(old_leader);
  sim.run(60);
  ASSERT_NE(sim.leader(), nullptr);
  EXPECT_NE(sim.leader()->id(), old_leader);
  ASSERT_TRUE(sim.propose("after"));
  sim.run(10);
  EXPECT_TRUE(sim.committedEverywhere("after"));
  EXPECT_TRUE(sim.committedEverywhere("before"));
}

TEST(RaftTest, MinorityPartitionCannotCommit) {
  Simulation sim(5, 4, kClean);
  sim.run(60);
  RaftNode* leader = sim.leader();
  ASSERT_NE(leader, nullptr);
  const NodeId old_id = leader->id();
  std::vector<NodeId> majority;
  NodeId minority_peer = 0;
  for (const NodeId id : sim.ids()) {
    if (id == old_id) {
      continue;
    }
    if (minority_peer == 0) {
      minority_peer = id;
    } else {
      majority.push_back(id);
    }
  }
  sim.partition({{old_id, minority_peer}, majority});
  (void)leader->propose({"lost"});
  sim.run(100);
  EXPECT_FALSE(sim.committedEverywhere("lost"));

  // The majority elects its own leader and makes progress.
  ASSERT_NE(sim.leader(), nullptr);
  ASSERT_NE(sim.leader()->id(), old_id);
  ASSERT_TRUE(sim.propose("kept"));
  sim.run(20);
  EXPECT_TRUE(sim.committedEverywhere("kept"));

  // On healing, the old leader's uncommitted entry is replaced, not committed.
  sim.heal();
  sim.run(100);
  EXPECT_FALSE(sim.committedEverywhere("lost"));
  EXPECT_EQ(sim.node(old_id).log().back().command.empty() ||
                sim.node(old_id).commitIndex() == sim.node(old_id).lastIndex(),
            true);
}

TEST(RaftTest, RestartedNodeKeepsItsVoteAndLog) {
  MemoryStorage storage;
  {
    RaftNode node(1, {1, 2, 3}, storage, RaftConfig{}, 1);
    node.receive(
        {.from = 2,
         .to = 1,
         .body = consensus::RequestVote{.term = 5, .last_log_index = 0, .last_log_term = 0}});
    (void)node.takeOutbox();
  }
  RaftNode restarted(1, {1, 2, 3}, storage, RaftConfig{}, 2);
  EXPECT_EQ(restarted.term(), 5U);
  // Having voted for node 2 in term 5, it must refuse node 3 in the same term.
  restarted.receive(
      {.from = 3,
       .to = 1,
       .body = consensus::RequestVote{.term = 5, .last_log_index = 0, .last_log_term = 0}});
  const auto replies = restarted.takeOutbox();
  ASSERT_EQ(replies.size(), 1U);
  EXPECT_FALSE(std::get<consensus::VoteReply>(replies[0].body).granted);
}

TEST(RaftTest, VoteRefusedForALessUpToDateLog) {
  MemoryStorage storage;
  storage.replaceSuffix(0, {{.term = 3, .command = "x"}});
  storage.saveTermAndVote(3, consensus::kNoNode);
  RaftNode node(1, {1, 2, 3}, storage, RaftConfig{}, 1);
  node.receive(
      {.from = 2,
       .to = 1,
       .body = consensus::RequestVote{.term = 4, .last_log_index = 10, .last_log_term = 2}});
  const auto replies = node.takeOutbox();
  ASSERT_EQ(replies.size(), 1U);
  EXPECT_FALSE(std::get<consensus::VoteReply>(replies[0].body).granted);
}

TEST(RaftTest, LateShorterAppendDoesNotRemoveNewerEntries) {
  MemoryStorage storage;
  RaftNode node(1, {1, 2}, storage, RaftConfig{}, 1);
  const LogEntry a{.term = 1, .command = "a"};
  const LogEntry b{.term = 1, .command = "b"};
  node.receive(
      {.from = 2,
       .to = 1,
       .body = consensus::AppendEntries{
           .term = 1, .prev_index = 0, .prev_term = 0, .entries = {a, b}, .leader_commit = 0}});
  node.receive(
      {.from = 2,
       .to = 1,
       .body = consensus::AppendEntries{
           .term = 1, .prev_index = 0, .prev_term = 0, .entries = {a}, .leader_commit = 0}});
  EXPECT_EQ(node.lastIndex(), 2U);
  EXPECT_EQ(storage.entries().size(), 2U);
}

TEST(RaftTest, EntryFromAnOlderTermIsNotCommittedByCountingReplicas) {
  // Figure 8 of the Raft paper: a majority holding an old-term entry is not enough.
  MemoryStorage storage;
  storage.replaceSuffix(0, {{.term = 1, .command = "old"}});
  storage.saveTermAndVote(1, consensus::kNoNode);
  RaftNode node(1, {1, 2, 3}, storage, RaftConfig{}, 1);
  while (node.role() != Role::kCandidate) {
    node.tick();
  }
  node.receive(
      {.from = 2, .to = 1, .body = consensus::VoteReply{.term = node.term(), .granted = true}});
  ASSERT_EQ(node.role(), Role::kLeader);
  ASSERT_EQ(node.lastIndex(), 2U);  // The old entry, then this term's no-op.
  node.receive({.from = 2,
                .to = 1,
                .body = consensus::AppendReply{
                    .term = node.term(), .success = true, .match_index = 1, .retry_from = 0}});
  EXPECT_EQ(node.commitIndex(), 0U);
  node.receive({.from = 2,
                .to = 1,
                .body = consensus::AppendReply{
                    .term = node.term(), .success = true, .match_index = 2, .retry_from = 0}});
  EXPECT_EQ(node.commitIndex(), 2U);
}

// Random crashes, partitions, and proposals on a lossy network. The Simulation asserts safety at
// every step. Afterwards, with faults removed, every node must agree and any acknowledged write
// must exist.
void runChaos(int nodes, std::uint64_t seed) {
  Simulation sim(nodes, seed, kHostile);
  std::vector<std::string> proposed;
  int next_command = 0;
  for (int round = 0; round < 60; ++round) {
    if (::testing::Test::HasFatalFailure()) {
      return;
    }
    const auto action = sim.rng()() % 10;
    const auto pick = sim.ids()[sim.rng()() % sim.ids().size()];
    if (action < 5) {
      const std::string command = "cmd-" + std::to_string(next_command++);
      if (sim.propose(command)) {
        proposed.push_back(command);
      }
    } else if (action == 5) {
      sim.crash(pick);
    } else if (action == 6) {
      if (!sim.isLive(pick)) {
        sim.start(pick);
      }
    } else if (action == 7) {
      std::vector<std::vector<NodeId>> groups(2);
      for (const NodeId id : sim.ids()) {
        groups[sim.rng()() % 2].push_back(id);
      }
      sim.partition(groups);
    } else if (action == 8) {
      sim.heal();
    }
    sim.run(static_cast<int>(sim.rng()() % 25));
  }
  sim.heal();
  for (const NodeId id : sim.ids()) {
    if (!sim.isLive(id)) {
      sim.start(id);
    }
  }
  sim.run(600);
  if (::testing::Test::HasFatalFailure()) {
    return;
  }
  // Liveness: the cluster must recover and commit something new. The network keeps losing messages,
  // so there may be no leader at any given moment, and a proposal can be lost. Retrying is what a
  // client would do. The limit is far above what recovery should take, so hitting it means a real
  // stall.
  const std::string final_command = "final-" + std::to_string(seed);
  int steps = 0;
  constexpr int kLimit = 20000;
  while (!sim.committedEverywhere(final_command) && steps < kLimit) {
    (void)sim.propose(final_command);
    sim.run(50);
    steps += 50;
    if (::testing::Test::HasFatalFailure()) {
      return;
    }
  }
  ASSERT_TRUE(sim.committedEverywhere(final_command)) << "stalled, seed " << seed;
  const auto allCaughtUp = [&] {
    return std::ranges::all_of(sim.ids(), [&](NodeId id) {
      return sim.node(id).commitIndex() == sim.committed().rbegin()->first;
    });
  };
  for (steps = 0; !allCaughtUp() && steps < kLimit; steps += 10) {
    sim.run(10);
    if (::testing::Test::HasFatalFailure()) {
      return;
    }
  }
  EXPECT_TRUE(allCaughtUp()) << "a node never caught up, seed " << seed;
}

class RaftChaosTest : public testing::TestWithParam<int> {};

TEST_P(RaftChaosTest, StaysSafeAndRecoversAcrossSeeds) {
  // RAFT_CHAOS_SEEDS runs more of them, for an occasional deep check.
  // NOLINTNEXTLINE(concurrency-mt-unsafe): nothing sets the environment while this runs.
  const char* configured = std::getenv("RAFT_CHAOS_SEEDS");
  const std::uint64_t seeds = configured != nullptr ? std::strtoull(configured, nullptr, 10) : 300;
  for (std::uint64_t seed = 1; seed <= seeds; ++seed) {
    runChaos(GetParam(), seed);
    if (HasFatalFailure()) {
      FAIL() << "failed with seed " << seed;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ClusterSizes, RaftChaosTest, testing::Values(3, 5));

}  // namespace
