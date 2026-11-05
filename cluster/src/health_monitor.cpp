#include "cluster/health_monitor.h"

#include <utility>

namespace cluster {

HealthMonitor::HealthMonitor(std::vector<std::string> nodes, Probe probe, OnChange on_change,
                             const HealthConfig& config)
    : nodes_(std::move(nodes)),
      probe_(std::move(probe)),
      on_change_(std::move(on_change)),
      config_(config) {
  for (const std::string& node : nodes_) {
    states_[node] = NodeState{};
  }
  thread_ = std::thread([this] { run(); });
}

HealthMonitor::~HealthMonitor() {
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
  }
  stop_requested_.notify_all();
  thread_.join();
}

void HealthMonitor::run() {
  std::unique_lock lock(mutex_);
  while (!stopping_) {
    lock.unlock();
    for (const std::string& node : nodes_) {
      const bool success = probe_(node);  // Probes can be slow, so no lock is held.
      record(node, success);
    }
    lock.lock();
    stop_requested_.wait_for(lock, config_.interval, [this] { return stopping_; });
  }
}

void HealthMonitor::record(const std::string& node, bool success) {
  const std::scoped_lock lock(mutex_);
  NodeState& state = states_.at(node);
  if (success) {
    state.consecutive_failures = 0;
    ++state.consecutive_successes;
    if (!state.healthy && state.consecutive_successes >= config_.success_threshold) {
      state.healthy = true;
      on_change_(node, true);
    }
  } else {
    state.consecutive_successes = 0;
    ++state.consecutive_failures;
    if (state.healthy && state.consecutive_failures >= config_.failure_threshold) {
      state.healthy = false;
      on_change_(node, false);
    }
  }
}

void HealthMonitor::reportFailure(const std::string& node) {
  const std::scoped_lock lock(mutex_);
  NodeState& state = states_.at(node);
  state.consecutive_successes = 0;
  if (state.healthy) {
    state.healthy = false;
    on_change_(node, false);
  }
}

bool HealthMonitor::isHealthy(const std::string& node) const {
  const std::scoped_lock lock(mutex_);
  return states_.at(node).healthy;
}

}  // namespace cluster
