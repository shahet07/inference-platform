#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cluster {

struct HealthConfig {
  std::chrono::milliseconds interval{250};
  // Consecutive failed probes before a node is taken out of rotation.
  std::size_t failure_threshold = 3;
  // Consecutive good probes before it returns. Above 1, so a flapping node stays out.
  std::size_t success_threshold = 2;
  // Failed live requests, with no good probe between, before a node is taken out. Above 1, because
  // a single failure can be one lost connection. Lower than failure_threshold, because a real
  // request failing is stronger evidence than a missed probe.
  std::size_t live_failure_threshold = 2;
};

// Probes nodes on a background thread. Nodes start healthy.
class HealthMonitor {
 public:
  using Probe = std::function<bool(const std::string& node)>;
  // Called in order, with a lock held, so it must be quick and must not call back into this class.
  using OnChange = std::function<void(const std::string& node, bool healthy)>;

  HealthMonitor(const std::vector<std::string>& nodes, Probe probe, OnChange on_change,
                const HealthConfig& config);
  ~HealthMonitor();

  HealthMonitor(const HealthMonitor&) = delete;
  HealthMonitor& operator=(const HealthMonitor&) = delete;
  HealthMonitor(HealthMonitor&&) = delete;
  HealthMonitor& operator=(HealthMonitor&&) = delete;

  // Records a failed live request, which takes the node out after live_failure_threshold in a row.
  void reportFailure(const std::string& node);

  [[nodiscard]] bool isHealthy(const std::string& node) const;

  // A new node starts healthy. Unknown nodes are ignored by the methods above, and report
  // unhealthy.
  void addNode(const std::string& node);
  void removeNode(const std::string& node);
  [[nodiscard]] std::vector<std::string> nodes() const;

 private:
  struct NodeState {
    bool healthy = true;
    std::size_t consecutive_failures = 0;
    std::size_t consecutive_successes = 0;
    std::size_t live_failures = 0;
  };

  void run();
  void record(const std::string& node, bool success);

  Probe probe_;
  OnChange on_change_;
  HealthConfig config_;
  mutable std::mutex mutex_;
  std::condition_variable stop_requested_;
  std::map<std::string, NodeState> states_;
  bool stopping_ = false;
  std::thread thread_;
};

}  // namespace cluster
