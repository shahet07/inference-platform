#include "cluster/health_monitor.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using cluster::HealthConfig;
using cluster::HealthMonitor;

constexpr HealthConfig kFast{.interval = 1ms, .failure_threshold = 3, .success_threshold = 2};

template <typename Condition>
bool eventually(Condition condition) {
  for (int attempt = 0; attempt < 2000; ++attempt) {
    if (condition()) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return false;
}

class Changes {
 public:
  void record(const std::string& node, bool healthy) {
    const std::scoped_lock lock(mutex_);
    changes_.emplace_back(node, healthy);
  }
  std::vector<std::pair<std::string, bool>> get() const {
    const std::scoped_lock lock(mutex_);
    return changes_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::pair<std::string, bool>> changes_;
};

TEST(HealthMonitorTest, StaysHealthyWhileProbesSucceed) {
  Changes changes;
  std::atomic<int> probes{0};
  HealthMonitor monitor(
      {"a"},
      [&](const std::string&) {
        ++probes;
        return true;
      },
      [&](const std::string& n, bool h) { changes.record(n, h); }, kFast);
  ASSERT_TRUE(eventually([&] { return probes > 10; }));
  EXPECT_TRUE(monitor.isHealthy("a"));
  EXPECT_TRUE(changes.get().empty());
}

TEST(HealthMonitorTest, MarksNodeDownOnlyAfterThreshold) {
  std::atomic<int> failures_remaining{2};
  std::atomic<int> probes{0};
  Changes changes;
  HealthMonitor monitor(
      {"a"},
      [&](const std::string&) {
        ++probes;
        return failures_remaining.fetch_sub(1) <= 0;  // Two failures, then success.
      },
      [&](const std::string& n, bool h) { changes.record(n, h); }, kFast);
  ASSERT_TRUE(eventually([&] { return probes > 10; }));
  EXPECT_TRUE(monitor.isHealthy("a"));  // Two failures are below the threshold of three.
  EXPECT_TRUE(changes.get().empty());
}

TEST(HealthMonitorTest, NodeGoesDownAndComesBack) {
  std::atomic<bool> up{false};
  Changes changes;
  HealthMonitor monitor(
      {"a"}, [&](const std::string&) { return up.load(); },
      [&](const std::string& n, bool h) { changes.record(n, h); }, kFast);

  ASSERT_TRUE(eventually([&] { return !monitor.isHealthy("a"); }));
  up = true;
  ASSERT_TRUE(eventually([&] { return monitor.isHealthy("a"); }));

  const auto seen = changes.get();
  ASSERT_EQ(seen.size(), 2U);
  EXPECT_EQ(seen[0], std::make_pair(std::string("a"), false));
  EXPECT_EQ(seen[1], std::make_pair(std::string("a"), true));
}

TEST(HealthMonitorTest, ReportFailureTakesEffectImmediately) {
  std::atomic<bool> up{false};
  HealthMonitor monitor(
      {"a"}, [&](const std::string&) { return up.load(); }, [](const std::string&, bool) {},
      HealthConfig{.interval = 1h, .failure_threshold = 100, .success_threshold = 1});
  monitor.reportFailure("a");
  EXPECT_FALSE(monitor.isHealthy("a"));
}

TEST(HealthMonitorTest, OneFailedNodeDoesNotAffectAnother) {
  HealthMonitor monitor(
      {"up", "down"}, [](const std::string& node) { return node == "up"; },
      [](const std::string&, bool) {}, kFast);
  ASSERT_TRUE(eventually([&] { return !monitor.isHealthy("down"); }));
  EXPECT_TRUE(monitor.isHealthy("up"));
}

}  // namespace
