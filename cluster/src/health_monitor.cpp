#include "cluster/health_monitor.h"

#include <utility>

namespace cluster {

HealthMonitor::HealthMonitor(const std::vector<std::string>& nodes, Probe probe, OnChange on_change,
                             const HealthConfig& config)
    : probe_(std::move(probe)), on_change_(std::move(on_change)), config_(config) {
  for (const std::string& node : nodes) {
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
    std::vector<std::string> snapshot;
    snapshot.reserve(states_.size());
    for (const auto& [node, state] : states_) {
      snapshot.push_back(node);
    }
    lock.unlock();
    for (const std::string& node : snapshot) {
      const bool success = probe_(node);  // Probes can be slow, so no lock is held.
      record(node, success);
    }
    lock.lock();
    stop_requested_.wait_for(lock, config_.interval, [this] { return stopping_; });
  }
}

void HealthMonitor::record(const std::string& node, bool success) {
  const std::scoped_lock lock(mutex_);
  const auto found = states_.find(node);
  if (found == states_.end()) {
    return;  // Removed while its probe was running.
  }
  NodeState& state = found->second;
  if (success) {
    state.consecutive_failures = 0;
    state.live_failures = 0;
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
  const auto found = states_.find(node);
  if (found == states_.end()) {
    return;
  }
  NodeState& state = found->second;
  state.consecutive_successes = 0;
  ++state.live_failures;
  if (state.healthy && state.live_failures >= config_.live_failure_threshold) {
    state.healthy = false;
    on_change_(node, false);
  }
}

bool HealthMonitor::isHealthy(const std::string& node) const {
  const std::scoped_lock lock(mutex_);
  const auto found = states_.find(node);
  return found != states_.end() && found->second.healthy;
}

void HealthMonitor::addNode(const std::string& node) {
  const std::scoped_lock lock(mutex_);
  states_.try_emplace(node);
}

void HealthMonitor::removeNode(const std::string& node) {
  const std::scoped_lock lock(mutex_);
  states_.erase(node);
}

std::vector<std::string> HealthMonitor::nodes() const {
  const std::scoped_lock lock(mutex_);
  std::vector<std::string> out;
  out.reserve(states_.size());
  for (const auto& [node, state] : states_) {
    out.push_back(node);
  }
  return out;
}

}  // namespace cluster
