#include "llm/scheduler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "llm/fake_backend.h"

namespace {

using namespace std::chrono_literals;
using llm::BatchPolicy;
using llm::FakeBackend;
using llm::FinishReason;
using llm::Scheduler;
using llm::SchedulerConfig;

// Records one request's tokens and result. The scheduler calls into these until it stops, so tests
// must declare them before it.
class Result {
 public:
  llm::GenerateRequest request(const std::string& prompt, std::int32_t max_tokens) {
    return {.prompt = prompt,
            .max_tokens = max_tokens,
            .on_token =
                [this](std::string_view text) {
                  const std::scoped_lock lock(mutex_);
                  tokens_.emplace_back(text);
                  changed_.notify_all();
                },
            .on_finish =
                [this](FinishReason reason) {
                  const std::scoped_lock lock(mutex_);
                  finished_ = true;
                  reason_ = reason;
                  changed_.notify_all();
                },
            .handle = handle};
  }

  FinishReason wait() {
    std::unique_lock lock(mutex_);
    EXPECT_TRUE(changed_.wait_for(lock, 30s, [this] { return finished_; }));
    return reason_;
  }

  void waitForTokens(std::size_t count) {
    std::unique_lock lock(mutex_);
    EXPECT_TRUE(changed_.wait_for(lock, 30s, [&] { return tokens_.size() >= count; }));
  }

  std::vector<std::string> tokens() {
    const std::scoped_lock lock(mutex_);
    return tokens_;
  }

  std::shared_ptr<llm::GenerateHandle> handle = std::make_shared<llm::GenerateHandle>();

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  std::vector<std::string> tokens_;
  bool finished_ = false;
  FinishReason reason_ = FinishReason::kError;
};

TEST(SchedulerTest, GeneratesUpToMaxTokens) {
  obs::Registry registry;
  FakeBackend backend;
  Result result;
  Scheduler scheduler(backend, SchedulerConfig{}, registry);
  ASSERT_TRUE(scheduler.submit(result.request("a b c", 4)));
  EXPECT_EQ(result.wait(), FinishReason::kMaxTokens);
  // A 3-word prompt, then one token per step, continuing the position.
  EXPECT_THAT(result.tokens(), testing::ElementsAre("t3 ", "t4 ", "t5 ", "t6 "));
}

TEST(SchedulerTest, EndTokenFinishesWithoutBeingEmitted) {
  obs::Registry registry;
  FakeBackend backend;
  backend.endAt(5);
  Result result;
  Scheduler scheduler(backend, SchedulerConfig{}, registry);
  ASSERT_TRUE(scheduler.submit(result.request("a b c", 100)));
  EXPECT_EQ(result.wait(), FinishReason::kEnd);
  EXPECT_THAT(result.tokens(), testing::ElementsAre("t3 ", "t4 "));
}

TEST(SchedulerTest, SequenceLimitStopsGeneration) {
  obs::Registry registry;
  FakeBackend backend;
  Result result;
  Scheduler scheduler(backend, {.max_sequence_tokens = 6}, registry);
  ASSERT_TRUE(scheduler.submit(result.request("a b c", 100)));
  EXPECT_EQ(result.wait(), FinishReason::kMaxTokens);
  EXPECT_LE(result.tokens().size(), 3U);
}

TEST(SchedulerTest, PromptThatDoesNotFitIsAnError) {
  obs::Registry registry;
  FakeBackend backend;
  Result too_long;
  Result empty;
  Scheduler scheduler(backend, {.max_sequence_tokens = 2}, registry);
  ASSERT_TRUE(scheduler.submit(too_long.request("a b c", 1)));
  ASSERT_TRUE(scheduler.submit(empty.request("", 1)));
  EXPECT_EQ(too_long.wait(), FinishReason::kError);
  EXPECT_EQ(empty.wait(), FinishReason::kError);
}

TEST(SchedulerTest, ContinuousBatchingAdmitsNewRequestsMidFlight) {
  obs::Registry registry;
  FakeBackend backend(100us);
  {
    Scheduler scheduler(backend, {.max_slots = 2, .policy = BatchPolicy::kContinuous}, registry);
    Result long_running;
    Result arrives_later;
    ASSERT_TRUE(scheduler.submit(long_running.request("a", 200)));
    long_running.waitForTokens(10);
    ASSERT_TRUE(scheduler.submit(arrives_later.request("a", 3)));
    EXPECT_EQ(arrives_later.wait(), FinishReason::kMaxTokens);
    // It finished long before the first request could have.
    EXPECT_LT(long_running.tokens().size(), 200U);
    EXPECT_EQ(long_running.wait(), FinishReason::kMaxTokens);
  }
  EXPECT_THAT(backend.last_step_sizes, testing::Contains(2));
}

TEST(SchedulerTest, StaticBatchingMakesNewRequestsWait) {
  obs::Registry registry;
  FakeBackend backend(100us);
  Result long_running;
  Result arrives_later;
  Scheduler scheduler(backend, {.max_slots = 2, .policy = BatchPolicy::kStatic}, registry);
  ASSERT_TRUE(scheduler.submit(long_running.request("a", 100)));
  long_running.waitForTokens(10);
  ASSERT_TRUE(scheduler.submit(arrives_later.request("a", 3)));
  EXPECT_EQ(arrives_later.wait(), FinishReason::kMaxTokens);
  EXPECT_EQ(long_running.tokens().size(), 100U);  // It had to finish first.
}

TEST(SchedulerTest, FinishedSlotIsReusedAtOnce) {
  obs::Registry registry;
  FakeBackend backend;
  std::vector<std::unique_ptr<Result>> results;
  results.reserve(20);
  Scheduler scheduler(backend, {.max_slots = 1}, registry);
  for (int i = 0; i < 20; ++i) {
    results.push_back(std::make_unique<Result>());
    ASSERT_TRUE(scheduler.submit(results.back()->request("a b", 3)));
  }
  for (const auto& result : results) {
    EXPECT_EQ(result->wait(), FinishReason::kMaxTokens);
    // The fake backend throws if a reused sequence was not released, and this is the same every
    // time.
    EXPECT_THAT(result->tokens(), testing::ElementsAre("t2 ", "t3 ", "t4 "));
  }
}

TEST(SchedulerTest, QueueLimitRejects) {
  obs::Registry registry;
  FakeBackend backend(1ms);
  Result running;
  Result queued;
  Result rejected;
  Scheduler scheduler(backend, {.max_slots = 1, .queue_capacity = 1}, registry);
  ASSERT_TRUE(scheduler.submit(running.request("a", 1000)));
  running.waitForTokens(1);
  EXPECT_TRUE(scheduler.submit(queued.request("a", 1)));
  EXPECT_FALSE(scheduler.submit(rejected.request("a", 1)));
  running.handle->cancelled = true;
}

TEST(SchedulerTest, CancellingStopsAndFreesTheSlot) {
  obs::Registry registry;
  FakeBackend backend(100us);
  Result cancelled;
  Result waiting;
  Scheduler scheduler(backend, {.max_slots = 1}, registry);
  ASSERT_TRUE(scheduler.submit(cancelled.request("a", 1000000)));
  cancelled.waitForTokens(1);
  ASSERT_TRUE(scheduler.submit(waiting.request("a", 2)));
  cancelled.handle->cancelled = true;
  EXPECT_EQ(cancelled.wait(), FinishReason::kCancelled);
  EXPECT_EQ(waiting.wait(), FinishReason::kMaxTokens);
}

TEST(SchedulerTest, RequestCancelledWhileQueuedNeverStarts) {
  obs::Registry registry;
  FakeBackend backend(100us);
  Result running;
  Result cancelled;
  Scheduler scheduler(backend, {.max_slots = 1}, registry);
  ASSERT_TRUE(scheduler.submit(running.request("a", 50)));
  running.waitForTokens(1);
  cancelled.handle->cancelled = true;
  ASSERT_TRUE(scheduler.submit(cancelled.request("a", 5)));
  EXPECT_EQ(cancelled.wait(), FinishReason::kCancelled);
  EXPECT_TRUE(cancelled.tokens().empty());
}

TEST(SchedulerTest, ShutdownFinishesEverythingAsCancelled) {
  obs::Registry registry;
  FakeBackend backend(1ms);
  Result running;
  Result queued;
  {
    Scheduler scheduler(backend, {.max_slots = 1}, registry);
    ASSERT_TRUE(scheduler.submit(running.request("a", 1000000)));
    running.waitForTokens(1);
    ASSERT_TRUE(scheduler.submit(queued.request("a", 1)));
  }
  EXPECT_EQ(running.wait(), FinishReason::kCancelled);
  EXPECT_EQ(queued.wait(), FinishReason::kCancelled);
}

TEST(SchedulerTest, ManyConcurrentRequestsAllGetTheirOwnTokens) {
  obs::Registry registry;
  FakeBackend backend;
  std::vector<std::unique_ptr<Result>> results;
  results.reserve(200);
  Scheduler scheduler(backend, {.max_slots = 8, .queue_capacity = 1000}, registry);
  for (int i = 0; i < 200; ++i) {
    results.push_back(std::make_unique<Result>());
    ASSERT_TRUE(scheduler.submit(results.back()->request("a", (i % 7) + 1)));
  }
  for (int i = 0; i < 200; ++i) {
    const auto& result = results[static_cast<std::size_t>(i)];
    EXPECT_EQ(result->wait(), FinishReason::kMaxTokens);
    const auto tokens = result->tokens();
    ASSERT_EQ(tokens.size(), static_cast<std::size_t>((i % 7) + 1));
    for (std::size_t t = 0; t < tokens.size(); ++t) {
      EXPECT_EQ(tokens[t], "t" + std::to_string(t + 1) + " ");
    }
  }
}

}  // namespace
