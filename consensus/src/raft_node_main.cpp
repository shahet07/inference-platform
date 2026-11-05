// A replicated log served over HTTP, as one node of a Raft cluster.
//   POST /append       The body is one command, as UTF-8 text. 200 once it is committed on a
//   majority.
//                      409 with X-Leader when this node is not the leader. 503 if the outcome is
//                      unknown, so a client retrying may add the command twice.
//   GET  /log?from=N   Committed entries from index N, in order
//   GET  /status       Role, term, leader, and commit index
#include <httplib.h>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "consensus/file_storage.h"
#include "consensus/raft.h"
#include "obs/startup.h"

namespace {

using consensus::AppendEntries;
using consensus::AppendReply;
using consensus::LogEntry;
using consensus::Message;
using consensus::NodeId;
using consensus::RequestVote;
using consensus::VoteReply;
using nlohmann::json;

constexpr std::size_t kMaxSendQueue = 4096;

json toJson(const Message& message) {
  json out{{"from", message.from}, {"to", message.to}};
  if (const auto* m = std::get_if<RequestVote>(&message.body)) {
    out["type"] = "vote";
    out["term"] = m->term;
    out["last_index"] = m->last_log_index;
    out["last_term"] = m->last_log_term;
  } else if (const auto* m = std::get_if<VoteReply>(&message.body)) {
    out["type"] = "vote_reply";
    out["term"] = m->term;
    out["granted"] = m->granted;
  } else if (const auto* m = std::get_if<AppendEntries>(&message.body)) {
    out["type"] = "append";
    out["term"] = m->term;
    out["prev_index"] = m->prev_index;
    out["prev_term"] = m->prev_term;
    out["commit"] = m->leader_commit;
    json entries = json::array();
    for (const LogEntry& entry : m->entries) {
      entries.push_back({entry.term, entry.command});
    }
    out["entries"] = std::move(entries);
  } else if (const auto* m = std::get_if<AppendReply>(&message.body)) {
    out["type"] = "append_reply";
    out["term"] = m->term;
    out["success"] = m->success;
    out["match"] = m->match_index;
    out["retry"] = m->retry_from;
  }
  return out;
}

Message fromJson(const json& in) {
  Message message{.from = in.at("from").get<NodeId>(), .to = in.at("to").get<NodeId>(), .body = {}};
  const auto type = in.at("type").get<std::string>();
  const auto term = in.at("term").get<consensus::Term>();
  if (type == "vote") {
    message.body = RequestVote{.term = term,
                               .last_log_index = in.at("last_index").get<consensus::Index>(),
                               .last_log_term = in.at("last_term").get<consensus::Term>()};
  } else if (type == "vote_reply") {
    message.body = VoteReply{.term = term, .granted = in.at("granted").get<bool>()};
  } else if (type == "append") {
    AppendEntries body{.term = term,
                       .prev_index = in.at("prev_index").get<consensus::Index>(),
                       .prev_term = in.at("prev_term").get<consensus::Term>(),
                       .entries = {},
                       .leader_commit = in.at("commit").get<consensus::Index>()};
    for (const auto& entry : in.at("entries")) {
      body.entries.push_back(
          {.term = entry.at(0).get<consensus::Term>(), .command = entry.at(1).get<std::string>()});
    }
    message.body = std::move(body);
  } else if (type == "append_reply") {
    message.body = AppendReply{.term = term,
                               .success = in.at("success").get<bool>(),
                               .match_index = in.at("match").get<consensus::Index>(),
                               .retry_from = in.at("retry").get<consensus::Index>()};
  } else {
    throw std::invalid_argument("unknown message type: " + type);
  }
  return message;
}

// Sends to one peer on its own thread, so a dead peer cannot hold up the others. A full queue drops
// the oldest messages, which is safe because Raft resends what is lost.
class PeerSender {
 public:
  explicit PeerSender(const std::string& address) : client_("http://" + address) {
    client_.set_keep_alive(true);
    client_.set_tcp_nodelay(true);
    client_.set_connection_timeout(std::chrono::milliseconds(100));
    client_.set_read_timeout(std::chrono::milliseconds(500));
    client_.set_write_timeout(std::chrono::milliseconds(500));
    thread_ = std::thread([this] { run(); });
  }
  ~PeerSender() {
    {
      const std::scoped_lock lock(mutex_);
      stopping_ = true;
    }
    has_messages_.notify_all();
    thread_.join();
  }
  PeerSender(const PeerSender&) = delete;
  PeerSender& operator=(const PeerSender&) = delete;
  PeerSender(PeerSender&&) = delete;
  PeerSender& operator=(PeerSender&&) = delete;

  void send(const Message& message) {
    {
      const std::scoped_lock lock(mutex_);
      queue_.push_back(toJson(message));
      if (queue_.size() > kMaxSendQueue) {
        queue_.pop_front();
      }
    }
    has_messages_.notify_one();
  }

 private:
  void run() {
    std::unique_lock lock(mutex_);
    while (true) {
      has_messages_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (stopping_) {
        return;
      }
      json batch = json::array();
      while (!queue_.empty()) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
      lock.unlock();
      (void)client_.Post("/raft", batch.dump(), "application/json");
      lock.lock();
    }
  }

  httplib::Client client_;
  std::mutex mutex_;
  std::condition_variable has_messages_;
  std::deque<json> queue_;
  bool stopping_ = false;
  std::thread thread_;
};

enum class AppendOutcome : std::uint8_t { kCommitted, kNotLeader, kUnknown };

struct AppendResult {
  AppendOutcome outcome = AppendOutcome::kUnknown;
  consensus::Index index = 0;
};

struct PendingAppend {
  std::string command;
  std::promise<AppendResult> done;
};

class RaftServer {
 public:
  RaftServer(NodeId id, const std::map<NodeId, std::string>& cluster,
             consensus::FileStorage& storage, std::chrono::milliseconds tick)
      : tick_(tick),
        cluster_(cluster),
        node_(id, peerIds(cluster), storage, consensus::RaftConfig{},
              std::random_device{}() ^ std::uint64_t{id}) {
    for (const auto& [peer, address] : cluster) {
      if (peer != id) {
        senders_.emplace(peer, std::make_unique<PeerSender>(address));
      }
    }
    routes();
    driver_ = std::thread([this] { drive(); });
  }

  ~RaftServer() {
    {
      const std::scoped_lock lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    driver_.join();
  }

  RaftServer(const RaftServer&) = delete;
  RaftServer& operator=(const RaftServer&) = delete;
  RaftServer(RaftServer&&) = delete;
  RaftServer& operator=(RaftServer&&) = delete;

  httplib::Server& http() { return http_; }

 private:
  static std::vector<NodeId> peerIds(const std::map<NodeId, std::string>& cluster) {
    std::vector<NodeId> ids;
    ids.reserve(cluster.size());
    for (const auto& [id, address] : cluster) {
      ids.push_back(id);
    }
    return ids;
  }

  void routes() {
    http_.set_tcp_nodelay(true);
    http_.set_keep_alive_max_count(1'000'000);
    http_.new_task_queue = [] { return new httplib::ThreadPool(128); };
    http_.Post("/raft", [this](const httplib::Request& req, httplib::Response& res) {
      try {
        const json batch = json::parse(req.body);
        std::vector<Message> messages;
        messages.reserve(batch.size());
        for (const auto& item : batch) {
          messages.push_back(fromJson(item));
        }
        {
          const std::scoped_lock lock(mutex_);
          for (Message& message : messages) {
            inbox_.push_back(std::move(message));
          }
        }
        wake_.notify_one();
      } catch (const std::exception&) {
        res.status = 400;
      }
    });
    http_.Post("/append", [this](const httplib::Request& req, httplib::Response& res) {
      handleAppend(req, res);
    });
    http_.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
      const std::scoped_lock lock(mutex_);
      res.set_content(status_.dump(), "application/json");
    });
    http_.Get("/log", [this](const httplib::Request& req, httplib::Response& res) {
      const std::size_t from = req.has_param("from") ? std::stoull(req.get_param_value("from")) : 1;
      json entries = json::array();
      {
        const std::scoped_lock lock(mutex_);
        for (std::size_t i = std::max<std::size_t>(from, 1); i <= applied_.size(); ++i) {
          entries.push_back({i, applied_[i - 1].term, applied_[i - 1].command});
        }
      }
      res.set_content(json{{"entries", entries}}.dump(), "application/json");
    });
    http_.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
      res.set_content("ok\n", "text/plain");
    });
  }

  void handleAppend(const httplib::Request& req, httplib::Response& res) {
    // Checked here because a command that cannot be encoded as JSON could not be replicated.
    try {
      (void)json(req.body).dump();
    } catch (const json::exception&) {
      res.status = 400;
      res.set_content("command must be valid UTF-8\n", "text/plain");
      return;
    }
    std::future<AppendResult> result;
    {
      const std::scoped_lock lock(mutex_);
      auto pending = std::make_unique<PendingAppend>();
      pending->command = req.body;
      result = pending->done.get_future();
      pending_.push_back(std::move(pending));
    }
    wake_.notify_one();
    if (result.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      res.status = 503;
      res.set_content("outcome unknown\n", "text/plain");
      return;
    }
    const AppendResult outcome = result.get();
    if (outcome.outcome == AppendOutcome::kCommitted) {
      res.set_content(json{{"index", outcome.index}}.dump(), "application/json");
    } else if (outcome.outcome == AppendOutcome::kNotLeader) {
      res.status = 409;
      const std::scoped_lock lock(mutex_);
      const auto leader = status_.at("leader").get<NodeId>();
      const auto it = cluster_.find(leader);
      if (it != cluster_.end()) {
        res.set_header("X-Leader", it->second);
      }
    } else {
      res.status = 503;
      res.set_content("outcome unknown\n", "text/plain");
    }
  }

  // All Raft work happens on this one thread, so the Raft node itself needs no lock.
  void drive() {
    auto next_tick = std::chrono::steady_clock::now() + tick_;
    while (true) {
      std::vector<Message> inbox;
      std::vector<std::unique_ptr<PendingAppend>> pending;
      {
        std::unique_lock lock(mutex_);
        wake_.wait_until(lock, next_tick,
                         [this] { return stopping_ || !inbox_.empty() || !pending_.empty(); });
        if (stopping_) {
          failWaiters(AppendOutcome::kUnknown);
          return;
        }
        inbox.swap(inbox_);
        pending.swap(pending_);
      }
      for (const Message& message : inbox) {
        node_.receive(message);
      }
      proposeAll(pending);
      if (std::chrono::steady_clock::now() >= next_tick) {
        node_.tick();
        next_tick += tick_;
        if (next_tick < std::chrono::steady_clock::now()) {
          next_tick = std::chrono::steady_clock::now() + tick_;
        }
      }
      flush();
    }
  }

  // All appends that arrived together share one write and one sync, which is the group commit.
  void proposeAll(std::vector<std::unique_ptr<PendingAppend>>& pending) {
    if (pending.empty()) {
      return;
    }
    std::vector<std::string> commands;
    commands.reserve(pending.size());
    for (const auto& item : pending) {
      commands.push_back(item->command);
    }
    const auto proposal = node_.propose(commands);
    for (std::size_t i = 0; i < pending.size(); ++i) {
      if (!proposal) {
        pending[i]->done.set_value({.outcome = AppendOutcome::kNotLeader, .index = 0});
      } else {
        waiters_.emplace(proposal->index + i,
                         Waiter{.term = proposal->term, .append = std::move(pending[i])});
      }
    }
  }

  void flush() {
    for (Message& message : node_.takeOutbox()) {
      const auto it = senders_.find(message.to);
      if (it != senders_.end()) {
        it->second->send(message);
      }
    }
    const auto committed = node_.takeCommitted();
    {
      const std::scoped_lock lock(mutex_);
      for (const auto& [index, entry] : committed) {
        applied_.push_back(entry);
      }
      status_ = {{"id", node_.id()},
                 {"role", node_.role() == consensus::Role::kLeader      ? "leader"
                          : node_.role() == consensus::Role::kCandidate ? "candidate"
                                                                        : "follower"},
                 {"term", node_.term()},
                 {"leader", node_.leaderHint()},
                 {"commit", node_.commitIndex()},
                 {"last", node_.lastIndex()}};
    }
    for (const auto& [index, entry] : committed) {
      const auto it = waiters_.find(index);
      if (it != waiters_.end()) {
        // A different term at this index means another leader's entry replaced this one.
        it->second.append->done.set_value({.outcome = it->second.term == entry.term
                                                          ? AppendOutcome::kCommitted
                                                          : AppendOutcome::kUnknown,
                                           .index = index});
        waiters_.erase(it);
      }
    }
    if (node_.role() != consensus::Role::kLeader) {
      failWaiters(AppendOutcome::kUnknown);
    }
  }

  void failWaiters(AppendOutcome outcome) {
    for (auto& [index, waiter] : waiters_) {
      waiter.append->done.set_value({.outcome = outcome, .index = 0});
    }
    waiters_.clear();
  }

  struct Waiter {
    consensus::Term term = 0;
    std::unique_ptr<PendingAppend> append;
  };

  std::chrono::milliseconds tick_;
  std::map<NodeId, std::string> cluster_;
  consensus::RaftNode node_;
  std::map<NodeId, std::unique_ptr<PeerSender>> senders_;
  httplib::Server http_;
  std::map<consensus::Index, Waiter> waiters_;  // Used only by the driver thread.

  std::mutex mutex_;
  std::condition_variable wake_;
  std::vector<Message> inbox_;
  std::vector<std::unique_ptr<PendingAppend>> pending_;
  std::vector<LogEntry> applied_;
  json status_ = {{"role", "follower"}, {"term", 0}, {"leader", 0}, {"commit", 0}, {"last", 0}};
  bool stopping_ = false;
  std::thread driver_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    obs::primeLocaleCaches();
    std::map<std::string, std::string> flags;
    const std::span args(argv, static_cast<std::size_t>(argc));
    for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
      flags[std::string(args[i]).substr(2)] = args[i + 1];
    }
    if (!flags.contains("id") || !flags.contains("cluster") || !flags.contains("dir")) {
      std::cerr
          << "usage: raft_node --id N --cluster 1=host:port,2=host:port,3=host:port --dir DIR\n"
             "  [--fsync 0|1] [--tick-ms N]\n";
      return EXIT_FAILURE;
    }
    std::signal(SIGPIPE, SIG_IGN);
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);

    std::map<NodeId, std::string> cluster;
    std::stringstream list(flags["cluster"]);
    for (std::string item; std::getline(list, item, ',');) {
      const auto equals = item.find('=');
      cluster[static_cast<NodeId>(std::stoul(item.substr(0, equals)))] = item.substr(equals + 1);
    }
    const auto id = static_cast<NodeId>(std::stoul(flags["id"]));
    const std::string address = cluster.at(id);
    const auto colon = address.rfind(':');

    consensus::FileStorage storage(flags["dir"], !flags.contains("fsync") || flags["fsync"] != "0");
    RaftServer server(
        id, cluster, storage,
        std::chrono::milliseconds(flags.contains("tick-ms") ? std::stoll(flags["tick-ms"]) : 10));
    if (!server.http().bind_to_port(address.substr(0, colon),
                                    std::stoi(address.substr(colon + 1)))) {
      std::cerr << "failed to bind " << address << '\n';
      return EXIT_FAILURE;
    }
    std::thread serving([&server] { server.http().listen_after_bind(); });
    int received = 0;
    sigwait(&signals, &received);
    server.http().stop();
    serving.join();
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
