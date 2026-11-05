#include "stream/consumer_group.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using stream::BackpressurePolicy;
using stream::ConsumerGroup;
using stream::Event;
using stream::ShardedLog;
using testing::ElementsAre;

stream::LogConfig config(std::size_t shards, std::size_t capacity = 16) {
  return {.shards = shards, .shard_capacity = capacity, .policy = BackpressurePolicy::kDropNewest};
}

Event eventFor(double value) { return {.key = "k", .value = value, .timestamp_ms = 0}; }

TEST(ConsumerGroupTest, SharesShardsEvenlyBetweenMembers) {
  ShardedLog log(config(5));
  ConsumerGroup group(log);
  const auto a = group.join();
  const auto b = group.join();

  EXPECT_THAT(group.assignment(a), ElementsAre(0, 2, 4));
  EXPECT_THAT(group.assignment(b), ElementsAre(1, 3));
}

TEST(ConsumerGroupTest, RemainingMemberTakesOverAfterLeave) {
  ShardedLog log(config(3));
  ConsumerGroup group(log);
  const auto a = group.join();
  const auto b = group.join();

  group.leave(a);

  EXPECT_THAT(group.assignment(b), ElementsAre(0, 1, 2));
  EXPECT_TRUE(group.assignment(a).empty());
}

TEST(ConsumerGroupTest, PollAdvancesPosition) {
  ShardedLog log(config(1));
  ConsumerGroup group(log);
  const auto member = group.join();
  (void)log.append(eventFor(1));
  (void)log.append(eventFor(2));

  EXPECT_EQ(group.poll(member, 1).at(0).value, 1);
  EXPECT_EQ(group.poll(member, 1).at(0).value, 2);
  EXPECT_TRUE(group.poll(member, 1).empty());
}

TEST(ConsumerGroupTest, UncommittedEventsAreRedeliveredAfterRebalance) {
  ShardedLog log(config(1));
  ConsumerGroup group(log);
  const auto a = group.join();
  (void)log.append(eventFor(1));
  (void)log.append(eventFor(2));
  EXPECT_EQ(group.poll(a, 1).size(), 1U);
  group.commit(a);
  EXPECT_EQ(group.poll(a, 1).size(), 1U);  // Polled, never committed.

  const auto b = group.join();
  group.leave(a);

  const auto events = group.poll(b, 10);
  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(events[0].value, 2);
}

TEST(ConsumerGroupTest, TwoGroupsReadIndependently) {
  ShardedLog log(config(1));
  ConsumerGroup first(log);
  ConsumerGroup second(log);
  const auto a = first.join();
  const auto b = second.join();
  (void)log.append(eventFor(1));

  EXPECT_EQ(first.poll(a, 10).size(), 1U);
  first.commit(a);
  EXPECT_EQ(second.poll(b, 10).size(), 1U);
}

}  // namespace
