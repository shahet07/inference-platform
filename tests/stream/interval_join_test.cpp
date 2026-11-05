#include "stream/interval_join.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

using stream::IntervalJoin;
using stream::JoinSide;

TEST(IntervalJoinTest, MatchesWithinTheGapInEitherArrivalOrder) {
  IntervalJoin left_first({.max_gap_ms = 100});
  EXPECT_TRUE(left_first.add(JoinSide::kLeft, "k", 1.0, 1000).empty());
  const auto pairs = left_first.add(JoinSide::kRight, "k", 2.0, 1050);
  ASSERT_EQ(pairs.size(), 1U);
  EXPECT_DOUBLE_EQ(pairs[0].left_value, 1.0);
  EXPECT_DOUBLE_EQ(pairs[0].right_value, 2.0);

  IntervalJoin right_first({.max_gap_ms = 100});
  EXPECT_TRUE(right_first.add(JoinSide::kRight, "k", 2.0, 1050).empty());
  const auto reversed = right_first.add(JoinSide::kLeft, "k", 1.0, 1000);
  ASSERT_EQ(reversed.size(), 1U);
  EXPECT_DOUBLE_EQ(reversed[0].left_value, 1.0);
  EXPECT_DOUBLE_EQ(reversed[0].right_value, 2.0);
}

TEST(IntervalJoinTest, TheGapIsInclusive) {
  IntervalJoin join({.max_gap_ms = 100});
  (void)join.add(JoinSide::kLeft, "k", 1.0, 1000);
  EXPECT_EQ(join.add(JoinSide::kRight, "k", 1.0, 1100).size(), 1U);
  EXPECT_TRUE(join.add(JoinSide::kRight, "k", 1.0, 1101).empty());
}

TEST(IntervalJoinTest, DifferentKeysDoNotMatch) {
  IntervalJoin join({.max_gap_ms = 100});
  (void)join.add(JoinSide::kLeft, "a", 1.0, 1000);
  EXPECT_TRUE(join.add(JoinSide::kRight, "b", 1.0, 1000).empty());
}

TEST(IntervalJoinTest, SameSideDoesNotMatchItself) {
  IntervalJoin join({.max_gap_ms = 100});
  (void)join.add(JoinSide::kLeft, "k", 1.0, 1000);
  EXPECT_TRUE(join.add(JoinSide::kLeft, "k", 1.0, 1000).empty());
}

TEST(IntervalJoinTest, OneEventCanMatchSeveral) {
  IntervalJoin join({.max_gap_ms = 100});
  (void)join.add(JoinSide::kLeft, "k", 1.0, 1000);
  (void)join.add(JoinSide::kLeft, "k", 2.0, 1010);
  EXPECT_EQ(join.add(JoinSide::kRight, "k", 3.0, 1005).size(), 2U);
}

TEST(IntervalJoinTest, MatchesABruteForceJoin) {
  // Both sides are in time order with jitter, as real streams are, and compared with the definition.
  // The seed is fixed so that a failure can be reproduced.
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(99);
  struct Item {
    JoinSide side;
    std::string key;
    std::int64_t time;
    double id;
  };
  std::vector<Item> items;
  items.reserve(5000);
  for (int i = 0; i < 5000; ++i) {
    items.push_back({.side = (rng() % 2 == 0) ? JoinSide::kLeft : JoinSide::kRight,
                     .key = "k" + std::to_string(rng() % 5),
                     .time = (std::int64_t{i} * 10) + static_cast<std::int64_t>(rng() % 50),
                     .id = static_cast<double>(i)});
  }
  IntervalJoin join({.max_gap_ms = 200});
  std::multiset<std::tuple<double, double>> actual;
  for (const Item& item : items) {
    for (const auto& pair : join.add(item.side, item.key, item.id, item.time)) {
      actual.emplace(pair.left_value, pair.right_value);
    }
  }
  std::multiset<std::tuple<double, double>> expected;
  for (const Item& a : items) {
    for (const Item& b : items) {
      if (a.side == JoinSide::kLeft && b.side == JoinSide::kRight && a.key == b.key &&
          std::abs(a.time - b.time) <= 200) {
        expected.emplace(a.id, b.id);
      }
    }
  }
  EXPECT_EQ(join.lateEvents(), 0U);
  EXPECT_EQ(actual, expected);
}

TEST(IntervalJoinTest, BufferedEventsAreEvictedAsTimeMovesOn) {
  IntervalJoin join({.max_gap_ms = 10});
  for (int i = 0; i < 100000; ++i) {
    (void)join.add((i % 2 == 0) ? JoinSide::kLeft : JoinSide::kRight, "k", 1.0, i);
  }
  EXPECT_LT(join.bufferedEvents(), 3000U);
}

TEST(IntervalJoinTest, StalledSideIsBoundedByTheKeyLimit) {
  IntervalJoin join({.max_gap_ms = 10, .max_buffered_per_key = 100});
  for (int i = 0; i < 10000; ++i) {
    (void)join.add(JoinSide::kLeft, "k", 1.0, i);
  }
  EXPECT_LE(join.bufferedEvents(), 100U);
  EXPECT_EQ(join.droppedEvents(), 9900U);
}

TEST(IntervalJoinTest, EventTooFarBehindTheOtherSideIsLate) {
  IntervalJoin join({.max_gap_ms = 10});
  (void)join.add(JoinSide::kRight, "k", 1.0, 1000);
  EXPECT_TRUE(join.add(JoinSide::kLeft, "k", 1.0, 100).empty());
  EXPECT_EQ(join.lateEvents(), 1U);
}

TEST(IntervalJoinTest, ExtremeTimestampsDoNotOverflow) {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  IntervalJoin join({.max_gap_ms = 10});
  (void)join.add(JoinSide::kLeft, "k", 1.0, kMin);
  EXPECT_TRUE(join.add(JoinSide::kRight, "k", 1.0, kMax).empty());
  EXPECT_EQ(join.add(JoinSide::kLeft, "k", 1.0, kMax).size(), 1U);
}

}  // namespace
