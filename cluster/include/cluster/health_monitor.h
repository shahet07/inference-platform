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
};

// Probes nodes on a background thread. Nodes start healthy.
class HealthMonitor {
 public:
  using Probe = std::function<bool(const std::string& node)>;
  // Called in order, with a lock held, so it must be quick and must not call back into this class.
  using OnChange = std::function<void(const std::string& node, bool healthy)>;

  HealthMonitor(std::vector<std::string> nodes, Probe probe, OnChange on_change,
                const HealthConfig& config);
  ~HealthMonitor();

  HealthMonitor(const HealthMonitor&) = delete;
  HealthMonitor& operator=(const HealthMonitor&) = delete;
  HealthMonitor(HealthMonitor&&) = delete;
  HealthMonitor& operator=(HealthMonitor&&) = delete;

  // Takes a node out immediately. A failed live request is stronger evidence than a missed
  // probe, and waiting for the threshold would fail more requests.
  void reportFailure(const std::string& node);

  [[nodiscard]] bool isHealthy(const std::string& node) const;

 private:
  struct NodeState {
    bool healthy = true;
    std::size_t consecutive_failures = 0;
    std::size_t consecutive_successes = 0;
  };

  void run();
  void record(const std::string& node, bool success);

  std::vector<std::string> nodes_;
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
