#include "server/model_manager.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using server::EngineConfig;
using server::LoadedBackend;
using server::ModelManager;

// A model that multiplies by a fixed number, so outputs show which version answered.
LoadedBackend scaling(float factor) {
  LoadedBackend backend;
  backend.row_shape = {1};
  backend.run = [factor](std::span<const float> input, std::span<const std::int64_t>) {
    std::vector<float> output(input.begin(), input.end());
    for (float& value : output) {
      value *= factor;
    }
    return output;
  };
  return backend;
}

ModelManager::Loader scalingFromPath() {
  return [](const std::filesystem::path& path) {
    if (path.filename() == "bad") {
      throw std::runtime_error("cannot load");
    }
    return scaling(std::stof(path.filename().string()));
  };
}

float infer(ModelManager& manager, const std::string& name, float input) {
  auto model = manager.get(name);
  auto result = model->engine().submit({input});
  if (!result) {
    throw std::runtime_error("rejected");
  }
  return result->get().at(0);
}

template <typename Condition>
bool eventually(Condition condition) {
  for (int attempt = 0; attempt < 5000; ++attempt) {
    if (condition()) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return false;
}

TEST(ModelManagerTest, ServesNamedModelsSeparately) {
  obs::Registry registry;
  ModelManager manager(scalingFromPath(), EngineConfig{}, registry);
  manager.install("a", scaling(2), 1);
  manager.install("b", scaling(3), 1);
  EXPECT_EQ(infer(manager, "a", 1), 2);
  EXPECT_EQ(infer(manager, "b", 1), 3);
  EXPECT_EQ(manager.get("missing"), nullptr);
}

TEST(ModelManagerTest, SwapReplacesTheModel) {
  obs::Registry registry;
  ModelManager manager(scalingFromPath(), EngineConfig{}, registry);
  manager.install("m", scaling(2), 1);
  manager.install("m", scaling(5), 2);
  EXPECT_EQ(infer(manager, "m", 1), 5);
  EXPECT_EQ(manager.versions().at("m"), 2U);
}

TEST(ModelManagerTest, FailedLoadKeepsCurrentModel) {
  obs::Registry registry;
  ModelManager manager(scalingFromPath(), EngineConfig{}, registry);
  manager.install("m", scaling(2), 1);
  EXPECT_THROW(manager.load("m", "bad", 2), std::runtime_error);
  EXPECT_EQ(infer(manager, "m", 1), 2);
  EXPECT_EQ(manager.versions().at("m"), 1U);
}

TEST(ModelManagerTest, ModelThatCannotRunIsNotSwappedIn) {
  obs::Registry registry;
  ModelManager manager(scalingFromPath(), EngineConfig{}, registry);
  manager.install("m", scaling(2), 1);
  LoadedBackend broken;
  broken.row_shape = {1};
  broken.run = [](std::span<const float>, std::span<const std::int64_t>) -> std::vector<float> {
    throw std::runtime_error("cannot run");
  };
  EXPECT_THROW(manager.install("m", std::move(broken), 2), std::runtime_error);
  EXPECT_EQ(infer(manager, "m", 1), 2);
}

TEST(ModelManagerTest, InFlightRequestFinishesOnTheOldModel) {
  obs::Registry registry;
  // Declared before the manager, because its worker threads use these until it stops them.
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();
  std::promise<void> running;
  std::atomic<int> calls{0};
  ModelManager manager(scalingFromPath(), EngineConfig{}, registry);
  LoadedBackend slow;
  slow.row_shape = {1};
  slow.run = [&](std::span<const float> input, std::span<const std::int64_t>) {
    // The first call is the check before swapping in, which must not wait.
    if (calls.fetch_add(1) == 1) {
      running.set_value();
      released.wait();
    }
    return std::vector<float>(input.begin(), input.end());
  };
  manager.install("m", std::move(slow), 1);

  auto holder = manager.get("m");
  auto in_flight = holder->engine().submit({7.0F});
  ASSERT_TRUE(in_flight.has_value());
  running.get_future().wait();

  manager.install("m", scaling(100), 2);  // Must not wait for, or disturb, the old model.
  EXPECT_EQ(infer(manager, "m", 1), 100);

  release.set_value();
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): checked above.
  EXPECT_EQ(in_flight->get().at(0), 7.0F);
}

TEST(ModelManagerTest, BusyModelDoesNotBlockAnother) {
  obs::Registry registry;
  // Declared before the manager, because its worker threads use these until it stops them.
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();
  std::atomic<int> calls{0};
  ModelManager manager(scalingFromPath(), EngineConfig{.workers = 1}, registry);
  LoadedBackend stuck;
  stuck.row_shape = {1};
  stuck.run = [&](std::span<const float> input, std::span<const std::int64_t>) {
    if (calls.fetch_add(1) > 0) {
      released.wait();
    }
    return std::vector<float>(input.begin(), input.end());
  };
  manager.install("stuck", std::move(stuck), 1);
  manager.install("fast", scaling(2), 1);

  auto holder = manager.get("stuck");
  auto blocked = holder->engine().submit({1.0F});

  EXPECT_EQ(infer(manager, "fast", 4), 8);
  release.set_value();
}

TEST(ModelManagerTest, PerModelMetricsAreLabeled) {
  obs::Registry registry;
  ModelManager manager(scalingFromPath(), EngineConfig{}, registry);
  manager.install("default", scaling(1), 1);
  manager.install("a", scaling(1), 1);
  (void)infer(manager, "default", 1);
  (void)infer(manager, "a", 1);
  const std::string text = registry.renderPrometheus();
  EXPECT_THAT(text, testing::HasSubstr("inference_requests_total 1\n"));
  EXPECT_THAT(text, testing::HasSubstr("inference_requests_total{model=\"a\"} 1\n"));
}

TEST(ModelManagerTest, CountersContinueAcrossSwaps) {
  obs::Registry registry;
  ModelManager manager(scalingFromPath(), EngineConfig{}, registry);
  manager.install("a", scaling(1), 1);
  (void)infer(manager, "a", 1);
  manager.install("a", scaling(1), 2);
  (void)infer(manager, "a", 1);
  EXPECT_THAT(registry.renderPrometheus(),
              testing::HasSubstr("inference_requests_total{model=\"a\"} 2\n"));
}

TEST(ModelManagerTest, FollowsRegistryVersions) {
  obs::Registry registry;
  cluster::ModelRegistry metadata("node");
  ModelManager manager(scalingFromPath(), EngineConfig{}, registry);
  manager.followRegistry(metadata, "/models", 1ms);

  metadata.put("m", "2");
  ASSERT_TRUE(eventually([&] { return manager.get("m") != nullptr; }));
  EXPECT_EQ(infer(manager, "m", 1), 2);

  metadata.put("m", "9");
  ASSERT_TRUE(eventually([&] { return manager.versions().at("m") == 2; }));
  EXPECT_EQ(infer(manager, "m", 1), 9);
}

TEST(ModelManagerTest, FollowerKeepsServingWhenAVersionFails) {
  obs::Registry registry;
  cluster::ModelRegistry metadata("node");
  ModelManager manager(scalingFromPath(), EngineConfig{}, registry);
  manager.followRegistry(metadata, "/models", 1ms);
  metadata.put("m", "2");
  ASSERT_TRUE(eventually([&] { return manager.get("m") != nullptr; }));

  metadata.put("m", "bad");
  ASSERT_TRUE(
      eventually([&] { return registry.counter("model_load_failures_total", "").value() == 1; }));
  EXPECT_EQ(infer(manager, "m", 1), 2);

  metadata.put("m", "3");  // A newer version is tried, and recovers.
  ASSERT_TRUE(eventually([&] { return manager.versions().at("m") == 3; }));
  EXPECT_EQ(infer(manager, "m", 1), 3);
}

TEST(ModelManagerTest, FollowerRefusesPathsOutsideTheModelDirectory) {
  obs::Registry registry;
  cluster::ModelRegistry metadata("node");
  std::atomic<int> loads{0};
  ModelManager manager(
      [&](const std::filesystem::path&) {
        ++loads;
        return scaling(1);
      },
      EngineConfig{}, registry);
  manager.followRegistry(metadata, "/models", 1ms);
  metadata.put("a", "../../etc/passwd");
  metadata.put("b", "/etc/passwd");
  metadata.put("c", "..");
  metadata.put("d", "");
  ASSERT_TRUE(
      eventually([&] { return registry.counter("model_load_failures_total", "").value() == 4; }));
  EXPECT_EQ(loads.load(), 0);
  EXPECT_TRUE(manager.versions().empty());
}

TEST(IsPlainFileNameTest, AcceptsOnlyPlainNames) {
  EXPECT_TRUE(server::isPlainFileName("model.onnx"));
  EXPECT_TRUE(server::isPlainFileName("a..b.onnx"));
  for (const char* bad : {"", ".", "..", "a/b", "../a", "/a", "a\\b"}) {
    EXPECT_FALSE(server::isPlainFileName(bad)) << bad;
  }
}

}  // namespace
