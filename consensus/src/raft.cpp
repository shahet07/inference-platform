#include "consensus/raft.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>

namespace consensus {

void MemoryStorage::saveTermAndVote(Term term, NodeId voted_for) {
  term_ = term;
  voted_for_ = voted_for;
}

void MemoryStorage::replaceSuffix(Index keep_up_to, const std::vector<LogEntry>& entries) {
  entries_.resize(keep_up_to);
  entries_.insert(entries_.end(), entries.begin(), entries.end());
}

RaftNode::RaftNode(NodeId id, std::vector<NodeId> peers, Storage& storage, const RaftConfig& config,
                   std::uint64_t seed)
    : id_(id),
      peers_(std::move(peers)),
      storage_(storage),
      config_(config),
      rng_(seed),
      term_(storage.term()),
      voted_for_(storage.votedFor()),
      log_(storage.entries()) {
  std::erase(peers_, id_);
  resetElectionTimer();
}

void RaftNode::resetElectionTimer() {
  ticks_since_heard_ = 0;
  std::uniform_int_distribution<std::uint32_t> range(config_.election_timeout_min,
                                                     config_.election_timeout_max);
  election_timeout_ = range(rng_);
}

void RaftNode::persistTermAndVote() { storage_.saveTermAndVote(term_, voted_for_); }

Term RaftNode::termAt(Index index) const { return index == 0 ? 0 : log_[index - 1].term; }

void RaftNode::tick() {
  if (role_ == Role::kLeader) {
    if (++ticks_since_heartbeat_ >= config_.heartbeat_interval) {
      broadcastAppends();
    }
    return;
  }
  if (++ticks_since_heard_ >= election_timeout_) {
    startElection();
  }
}

void RaftNode::startElection() {
  role_ = Role::kCandidate;
  leader_ = kNoNode;
  ++term_;
  voted_for_ = id_;
  persistTermAndVote();
  resetElectionTimer();
  votes_.clear();
  votes_[id_] = true;
  if (peers_.empty()) {
    becomeLeader();
    return;
  }
  for (const NodeId peer : peers_) {
    outbox_.push_back(
        {.from = id_,
         .to = peer,
         .body = RequestVote{
             .term = term_, .last_log_index = lastIndex(), .last_log_term = termAt(lastIndex())}});
  }
}

void RaftNode::becomeFollower(Term term, NodeId leader) {
  if (term > term_) {
    term_ = term;
    voted_for_ = kNoNode;
    persistTermAndVote();
  }
  role_ = Role::kFollower;
  leader_ = leader;
}

void RaftNode::observeTerm(Term term) {
  if (term > term_) {
    becomeFollower(term, kNoNode);
  }
}

void RaftNode::becomeLeader() {
  role_ = Role::kLeader;
  leader_ = id_;
  for (const NodeId peer : peers_) {
    next_index_[peer] = lastIndex() + 1;
    match_index_[peer] = 0;
  }
  // A leader may only count replicas for entries of its own term, so older entries can only commit
  // once one of its own does. Without this entry, they could wait indefinitely for a client.
  const LogEntry no_op{.term = term_, .command = ""};
  storage_.replaceSuffix(lastIndex(), {no_op});
  log_.push_back(no_op);
  advanceCommit();
  broadcastAppends();
}

std::optional<RaftNode::Proposal> RaftNode::propose(const std::vector<std::string>& commands) {
  if (role_ != Role::kLeader) {
    return std::nullopt;
  }
  std::vector<LogEntry> entries;
  entries.reserve(commands.size());
  for (const std::string& command : commands) {
    entries.push_back({.term = term_, .command = command});
  }
  const Proposal proposal{.index = lastIndex() + 1, .term = term_};
  storage_.replaceSuffix(lastIndex(), entries);
  log_.insert(log_.end(), entries.begin(), entries.end());
  advanceCommit();
  broadcastAppends();
  return proposal;
}

void RaftNode::sendAppend(NodeId peer) {
  const Index next = next_index_[peer];
  const Index prev = next - 1;
  AppendEntries message{.term = term_,
                        .prev_index = prev,
                        .prev_term = termAt(prev),
                        .entries = {},
                        .leader_commit = commit_index_};
  const Index end = std::min<Index>(lastIndex(), prev + config_.max_entries_per_message);
  message.entries.assign(log_.begin() + static_cast<std::ptrdiff_t>(prev),
                         log_.begin() + static_cast<std::ptrdiff_t>(end));
  // Pipelining: assume this arrives and carry on from after it, instead of resending everything
  // unacknowledged with each new entry, which grows quadratically and swamps the network under
  // load. If it was lost, the next message fails the follower's check, and the reply moves this
  // back.
  next_index_[peer] = end + 1;
  outbox_.push_back({.from = id_, .to = peer, .body = std::move(message)});
}

void RaftNode::broadcastAppends() {
  ticks_since_heartbeat_ = 0;
  for (const NodeId peer : peers_) {
    sendAppend(peer);
  }
}

void RaftNode::advanceCommit() {
  if (role_ != Role::kLeader) {
    return;
  }
  std::vector<Index> matches{lastIndex()};
  for (const NodeId peer : peers_) {
    matches.push_back(match_index_[peer]);
  }
  std::ranges::sort(matches, std::greater<>());
  const Index majority_match = matches[matches.size() / 2];
  // Only entries from this term count by replica. Older ones commit with them. This is the rule
  // that stops a committed entry being overwritten (figure 8 in the Raft paper).
  if (majority_match > commit_index_ && termAt(majority_match) == term_) {
    commit_index_ = majority_match;
  }
}

void RaftNode::receive(const Message& message) {
  std::visit([this, from = message.from](const auto& body) { handle(from, body); }, message.body);
}

void RaftNode::handle(NodeId from, const RequestVote& message) {
  observeTerm(message.term);
  bool granted = false;
  if (message.term == term_ && (voted_for_ == kNoNode || voted_for_ == from)) {
    // Votes only go to logs at least as up to date as this one, so a leader always holds every
    // committed entry.
    const Term my_last_term = termAt(lastIndex());
    const bool up_to_date =
        message.last_log_term > my_last_term ||
        (message.last_log_term == my_last_term && message.last_log_index >= lastIndex());
    if (up_to_date) {
      voted_for_ = from;
      persistTermAndVote();
      resetElectionTimer();
      granted = true;
    }
  }
  outbox_.push_back(
      {.from = id_, .to = from, .body = VoteReply{.term = term_, .granted = granted}});
}

void RaftNode::handle(NodeId from, const VoteReply& message) {
  observeTerm(message.term);
  if (role_ != Role::kCandidate || message.term != term_ || !message.granted) {
    return;
  }
  votes_[from] = true;
  if (votes_.size() > (peers_.size() + 1) / 2) {
    becomeLeader();
  }
}

void RaftNode::handle(NodeId from, const AppendEntries& message) {
  observeTerm(message.term);
  const auto reply = [&](bool success, Index match, Index retry) {
    outbox_.push_back(
        {.from = id_,
         .to = from,
         .body = AppendReply{
             .term = term_, .success = success, .match_index = match, .retry_from = retry}});
  };
  if (message.term < term_) {
    reply(false, 0, 0);
    return;
  }
  // Same term, so this is the term's one leader. A candidate steps down for it.
  role_ = Role::kFollower;
  leader_ = from;
  resetElectionTimer();

  if (message.prev_index > lastIndex()) {
    reply(false, 0, lastIndex() + 1);
    return;
  }
  if (termAt(message.prev_index) != message.prev_term) {
    // Skips the whole conflicting term instead of one entry per round trip.
    const Term conflict = termAt(message.prev_index);
    Index first = message.prev_index;
    while (first > 1 && termAt(first - 1) == conflict) {
      --first;
    }
    reply(false, 0, first);
    return;
  }

  // Only truncates at a real conflict. Messages can arrive late and out of order, and an old
  // shorter message must not remove entries that a newer one added and this node may have
  // acknowledged.
  Index index = message.prev_index;
  std::size_t next_new = 0;
  while (next_new < message.entries.size() && index + 1 <= lastIndex() &&
         termAt(index + 1) == message.entries[next_new].term) {
    ++index;
    ++next_new;
  }
  if (next_new < message.entries.size()) {
    const std::vector<LogEntry> added(
        message.entries.begin() + static_cast<std::ptrdiff_t>(next_new), message.entries.end());
    storage_.replaceSuffix(index, added);
    log_.resize(index);
    log_.insert(log_.end(), added.begin(), added.end());
  }

  const Index last_matching = message.prev_index + message.entries.size();
  if (message.leader_commit > commit_index_) {
    commit_index_ = std::max(commit_index_, std::min(message.leader_commit, last_matching));
  }
  reply(true, last_matching, 0);
}

void RaftNode::handle(NodeId from, const AppendReply& message) {
  observeTerm(message.term);
  if (role_ != Role::kLeader || message.term != term_) {
    return;
  }
  if (message.success) {
    Index& match = match_index_[from];
    match = std::max(match, message.match_index);
    next_index_[from] = std::max(next_index_[from], match + 1);
    advanceCommit();
    if (next_index_[from] <= lastIndex()) {
      sendAppend(from);
    }
    return;
  }
  // Late failures must not move this below what is already known to match.
  next_index_[from] = std::max({message.retry_from, Index{1}, match_index_[from] + 1});
  sendAppend(from);
}

std::vector<Message> RaftNode::takeOutbox() { return std::exchange(outbox_, {}); }

std::vector<std::pair<Index, LogEntry>> RaftNode::takeCommitted() {
  std::vector<std::pair<Index, LogEntry>> committed;
  while (delivered_up_to_ < commit_index_) {
    ++delivered_up_to_;
    committed.emplace_back(delivered_up_to_, log_[delivered_up_to_ - 1]);
  }
  return committed;
}

}  // namespace consensus
