#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "cluster/model_registry.h"

namespace cluster {

// Periodically pulls each peer's full registry and merges it. Pulling means a node that was down
// catches up on its own, with no peer needing to remember what it missed. A cluster of N nodes
// converges in about one interval when every node lists every peer.
class Replicator {
 public:
  // Returns a peer's registry as JSON, or nullopt if it cannot be reached.
  using Fetch = std::function<std::optional<std::string>(const std::string& peer)>;

  Replicator(ModelRegistry& registry, std::vector<std::string> peers, Fetch fetch,
             std::chrono::milliseconds interval);
  ~Replicator();

  Replicator(const Replicator&) = delete;
  Replicator& operator=(const Replicator&) = delete;
  Replicator(Replicator&&) = delete;
  Replicator& operator=(Replicator&&) = delete;

  // Fetches over HTTP from "host:port", sending `api_key` when peers require one.
  [[nodiscard]] static Fetch httpFetch(const std::string& api_key = "");

 private:
  void run();
  void pullOnce();

  ModelRegistry& registry_;
  std::vector<std::string> peers_;
  Fetch fetch_;
  std::chrono::milliseconds interval_;
  std::mutex mutex_;
  std::condition_variable stop_requested_;
  bool stopping_ = false;
  std::thread thread_;
};

}  // namespace cluster
