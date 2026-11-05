#include "server/batching_engine.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using server::BatchingEngine;
using server::EngineConfig;

// Doubles every input, which keeps each output traceable to its request.
std::vector<float> doubleAll(std::span<const float> input,
                             std::span<const std::int64_t> /*shape*/) {
  std::vector<float> output(input.begin(), input.end());
  for (float& value : output) {
    value *= 2.0F;
  }
  return output;
}

std::future<std::vector<float>> submitOrThrow(BatchingEngine& engine, std::vector<float> features) {
  auto result = engine.submit(std::move(features));
  if (!result) {
    throw std::runtime_error("request was rejected");
  }
  return std::move(*result);
}

TEST(BatchingEngineTest, FullBatchRunsWithoutWaitingForDeadline) {
  obs::Registry registry;
  std::vector<std::int64_t> batch_rows;
  {
    BatchingEngine engine(
        [&](std::span<const float> input, std::span<const std::int64_t> shape) {
          batch_rows.push_back(shape[0]);
          return doubleAll(input, shape);
        },
        1, EngineConfig{.workers = 1, .max_batch_size = 3, .max_wait = 60s, .queue_capacity = 8},
        registry);

    auto a = submitOrThrow(engine, {1.0F});
    auto b = submitOrThrow(engine, {2.0F});
    auto c = submitOrThrow(engine, {3.0F});

    ASSERT_EQ(c.wait_for(10s), std::future_status::ready);
    EXPECT_THAT(a.get(), testing::ElementsAre(2.0F));
    EXPECT_THAT(b.get(), testing::ElementsAre(4.0F));
    EXPECT_THAT(c.get(), testing::ElementsAre(6.0F));
  }
  EXPECT_THAT(batch_rows, testing::ElementsAre(3));
}

TEST(BatchingEngineTest, PartialBatchRunsAtDeadline) {
  obs::Registry registry;
  BatchingEngine engine(
      doubleAll, 1,
      EngineConfig{.workers = 1, .max_batch_size = 100, .max_wait = 1ms, .queue_capacity = 8},
      registry);

  auto result = submitOrThrow(engine, {5.0F});

  ASSERT_EQ(result.wait_for(10s), std::future_status::ready);
  EXPECT_THAT(result.get(), testing::ElementsAre(10.0F));
}

TEST(BatchingEngineTest, SplitsMultiColumnOutputsByRow) {
  obs::Registry registry;
  BatchingEngine engine(
      doubleAll, 2,
      EngineConfig{.workers = 1, .max_batch_size = 2, .max_wait = 60s, .queue_capacity = 8},
      registry);

  auto a = submitOrThrow(engine, {1.0F, 2.0F});
  auto b = submitOrThrow(engine, {3.0F, 4.0F});

  EXPECT_THAT(a.get(), testing::ElementsAre(2.0F, 4.0F));
  EXPECT_THAT(b.get(), testing::ElementsAre(6.0F, 8.0F));
}

TEST(BatchingEngineTest, RejectsWhenQueueIsFull) {
  obs::Registry registry;
  std::promise<void> release;
  std::promise<void> running;
  std::shared_future<void> released = release.get_future().share();
  bool announced = false;
  BatchingEngine engine(
      [&](std::span<const float> input, std::span<const std::int64_t> shape) {
        if (!announced) {
          announced = true;
          running.set_value();
        }
        released.wait();
        return doubleAll(input, shape);
      },
      1, EngineConfig{.workers = 1, .max_batch_size = 1, .max_wait = 0us, .queue_capacity = 1},
      registry);

  auto in_flight = submitOrThrow(engine, {1.0F});  // Occupies the only worker.
  running.get_future().wait();
  auto queued = submitOrThrow(engine, {2.0F});  // Fills the queue.
  auto rejected = engine.submit({3.0F});

  EXPECT_FALSE(rejected.has_value());
  release.set_value();
  EXPECT_THAT(in_flight.get(), testing::ElementsAre(2.0F));
  EXPECT_THAT(queued.get(), testing::ElementsAre(4.0F));
}

TEST(BatchingEngineTest, RejectsWrongFeatureCount) {
  obs::Registry registry;
  BatchingEngine engine(doubleAll, 2, EngineConfig{}, registry);
  EXPECT_THROW((void)engine.submit({1.0F}), std::invalid_argument);
}

TEST(BatchingEngineTest, ModelFailureReachesEveryRequestInBatch) {
  obs::Registry registry;
  BatchingEngine engine(
      [](std::span<const float>, std::span<const std::int64_t>) -> std::vector<float> {
        throw std::runtime_error("model failed");
      },
      1, EngineConfig{.workers = 1, .max_batch_size = 2, .max_wait = 60s, .queue_capacity = 8},
      registry);

  auto a = submitOrThrow(engine, {1.0F});
  auto b = submitOrThrow(engine, {2.0F});

  EXPECT_THROW((void)a.get(), std::runtime_error);
  EXPECT_THROW((void)b.get(), std::runtime_error);
}

TEST(BatchingEngineTest, DestructorFinishesAcceptedRequests) {
  obs::Registry registry;
  std::vector<std::future<std::vector<float>>> futures;
  {
    BatchingEngine engine(
        doubleAll, 1,
        EngineConfig{.workers = 2, .max_batch_size = 4, .max_wait = 1ms, .queue_capacity = 64},
        registry);
    for (int i = 0; i < 32; ++i) {
      futures.push_back(submitOrThrow(engine, {static_cast<float>(i)}));
    }
  }
  for (int i = 0; i < 32; ++i) {
    ASSERT_EQ(futures[static_cast<std::size_t>(i)].wait_for(0s), std::future_status::ready);
    EXPECT_THAT(futures[static_cast<std::size_t>(i)].get(),
                testing::ElementsAre(static_cast<float>(i) * 2.0F));
  }
}

}  // namespace
