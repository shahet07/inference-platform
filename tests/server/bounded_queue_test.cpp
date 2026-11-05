#include "server/bounded_queue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace {

using server::BoundedQueue;

TEST(BoundedQueueTest, TryPushFailsWhenFullAndKeepsValue) {
  BoundedQueue<std::unique_ptr<int>> queue(1);
  auto first = std::make_unique<int>(1);
  auto second = std::make_unique<int>(2);

  EXPECT_TRUE(queue.tryPush(first));
  EXPECT_FALSE(queue.tryPush(second));
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(*second, 2);
}

TEST(BoundedQueueTest, PopsInFifoOrder) {
  BoundedQueue<int> queue(3);
  queue.push(1);
  queue.push(2);
  EXPECT_EQ(queue.pop(), 1);
  EXPECT_EQ(queue.pop(), 2);
}

TEST(BoundedQueueTest, PopUntilTimesOutWhenEmpty) {
  BoundedQueue<int> queue(1);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
  EXPECT_EQ(queue.popUntil(deadline), std::nullopt);
  EXPECT_GE(std::chrono::steady_clock::now(), deadline);
}

TEST(BoundedQueueTest, CloseDrainsRemainingItemsThenEnds) {
  BoundedQueue<int> queue(2);
  queue.push(1);
  queue.close();

  EXPECT_FALSE(queue.push(2));
  EXPECT_EQ(queue.pop(), 1);
  EXPECT_EQ(queue.pop(), std::nullopt);
}

TEST(BoundedQueueTest, CloseWakesBlockedProducer) {
  BoundedQueue<int> queue(1);
  queue.push(1);
  std::atomic<bool> pushed{true};
  std::thread producer([&] { pushed = queue.push(2); });
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  queue.close();
  producer.join();
  EXPECT_FALSE(pushed);
}

TEST(BoundedQueueTest, ManyProducersAndConsumersLoseNothing) {
  constexpr int kPerProducer = 2000;
  BoundedQueue<int> queue(8);
  std::atomic<long> sum{0};
  std::vector<std::thread> consumers;
  consumers.reserve(4);
  for (int c = 0; c < 4; ++c) {
    consumers.emplace_back([&] {
      while (auto item = queue.pop()) {
        sum += *item;
      }
    });
  }
  std::vector<std::thread> producers;
  producers.reserve(4);
  for (int p = 0; p < 4; ++p) {
    producers.emplace_back([&] {
      for (int i = 1; i <= kPerProducer; ++i) {
        queue.push(i);
      }
    });
  }
  for (std::thread& t : producers) {
    t.join();
  }
  queue.close();
  for (std::thread& t : consumers) {
    t.join();
  }
  EXPECT_EQ(sum, 4L * kPerProducer * (kPerProducer + 1) / 2);
}

}  // namespace
