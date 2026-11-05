#include "server/mpmc_queue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using server::MpmcQueue;

TEST(MpmcQueueTest, PopsInOrderOnOneThread) {
  MpmcQueue<int> queue(4);
  for (int i = 0; i < 4; ++i) {
    int value = i;
    EXPECT_TRUE(queue.tryPush(value));
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(queue.tryPop(), i);
  }
}

TEST(MpmcQueueTest, EmptyAndFullAreReported) {
  MpmcQueue<int> queue(2);
  EXPECT_EQ(queue.tryPop(), std::nullopt);
  int value = 1;
  EXPECT_TRUE(queue.tryPush(value));
  EXPECT_TRUE(queue.tryPush(value));
  EXPECT_FALSE(queue.tryPush(value));
}

TEST(MpmcQueueTest, FailedPushLeavesTheValue) {
  MpmcQueue<std::unique_ptr<int>> queue(2);
  auto first = std::make_unique<int>(1);
  auto second = std::make_unique<int>(2);
  auto third = std::make_unique<int>(3);
  EXPECT_TRUE(queue.tryPush(first));
  EXPECT_TRUE(queue.tryPush(second));
  EXPECT_FALSE(queue.tryPush(third));
  EXPECT_NE(third, nullptr);
}

TEST(MpmcQueueTest, CapacityOfOneDoesNotOverwrite) {
  MpmcQueue<int> queue(1);  // Becomes 2, because one slot cannot tell full from free.
  int a = 1;
  int b = 2;
  EXPECT_TRUE(queue.tryPush(a));
  EXPECT_TRUE(queue.tryPush(b));
  EXPECT_EQ(queue.tryPop(), 1);
  EXPECT_EQ(queue.tryPop(), 2);
}

TEST(MpmcQueueTest, CapacityRoundsUpAndWrapsAround) {
  MpmcQueue<int> queue(3);  // Becomes 4.
  for (int round = 0; round < 100; ++round) {
    for (int i = 0; i < 4; ++i) {
      int value = (round * 10) + i;
      ASSERT_TRUE(queue.tryPush(value));
    }
    for (int i = 0; i < 4; ++i) {
      ASSERT_EQ(queue.tryPop(), (round * 10) + i);
    }
  }
}

TEST(MpmcQueueTest, ZeroCapacityIsRejected) {
  EXPECT_THROW(MpmcQueue<int> queue(0), std::invalid_argument);
}

TEST(MpmcQueueTest, ManyThreadsDeliverEachItemExactlyOnce) {
  constexpr int kThreads = 4;
  constexpr int kPerProducer = 50000;
  MpmcQueue<int> queue(64);
  constexpr std::size_t kTotal = 200000;  // Threads times items per producer.
  std::vector<std::atomic<int>> seen(kTotal);
  std::atomic<int> consumed{0};
  std::vector<std::thread> threads;
  threads.reserve(std::size_t{kThreads} * 2);
  for (int p = 0; p < kThreads; ++p) {
    threads.emplace_back([&, p] {
      for (int i = 0; i < kPerProducer; ++i) {
        int value = (p * kPerProducer) + i;
        while (!queue.tryPush(value)) {
          std::this_thread::yield();
        }
      }
    });
    threads.emplace_back([&] {
      while (consumed.load() < kThreads * kPerProducer) {
        if (const auto value = queue.tryPop()) {
          seen[static_cast<std::size_t>(*value)].fetch_add(1);
          consumed.fetch_add(1);
        } else {
          std::this_thread::yield();
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (const auto& count : seen) {
    ASSERT_EQ(count.load(), 1);
  }
}

}  // namespace
