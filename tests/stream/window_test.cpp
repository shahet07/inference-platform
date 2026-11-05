#include "stream/window.h"

#include <gtest/gtest.h>

namespace {

using stream::WindowAggregator;

TEST(WindowTest, AggregatesPerKeyWithinAWindow) {
  WindowAggregator aggregator({.window_ms = 100});
  aggregator.add("a", 1.0, 10);
  aggregator.add("a", 3.0, 20);
  aggregator.add("b", 7.0, 30);
  aggregator.add("a", 0.0, 100);  // Moves the watermark to the end of the first window.

  auto closed = aggregator.takeClosed();

  ASSERT_EQ(closed.size(), 2U);
  const auto& a = closed[0].key == "a" ? closed[0] : closed[1];
  const auto& b = closed[0].key == "a" ? closed[1] : closed[0];
  EXPECT_EQ(a.count, 2U);
  EXPECT_DOUBLE_EQ(a.sum, 4.0);
  EXPECT_DOUBLE_EQ(a.min, 1.0);
  EXPECT_DOUBLE_EQ(a.max, 3.0);
  EXPECT_EQ(a.window_start_ms, 0);
  EXPECT_EQ(a.window_end_ms, 100);
  EXPECT_EQ(b.count, 1U);
}

TEST(WindowTest, WindowStaysOpenUntilWatermarkReachesItsEnd) {
  WindowAggregator aggregator({.window_ms = 100});
  aggregator.add("a", 1.0, 10);
  aggregator.add("a", 1.0, 99);
  EXPECT_TRUE(aggregator.takeClosed().empty());
}

TEST(WindowTest, LateEventsAreDroppedAndCounted) {
  WindowAggregator aggregator({.window_ms = 100});
  aggregator.add("a", 1.0, 10);
  aggregator.add("a", 1.0, 150);
  EXPECT_EQ(aggregator.takeClosed().size(), 1U);

  aggregator.add("a", 1.0, 20);

  EXPECT_EQ(aggregator.lateEvents(), 1U);
  EXPECT_EQ(aggregator.flush().at(0).count, 1U);
}

TEST(WindowTest, AllowedLatenessKeepsWindowOpen) {
  WindowAggregator aggregator({.window_ms = 100, .allowed_lateness_ms = 50});
  aggregator.add("a", 1.0, 10);
  aggregator.add("a", 1.0, 120);
  EXPECT_TRUE(aggregator.takeClosed().empty());

  aggregator.add("a", 1.0, 20);  // Late, but within the allowance.
  aggregator.add("a", 1.0, 150);
  const auto closed = aggregator.takeClosed();

  ASSERT_EQ(closed.size(), 1U);
  EXPECT_EQ(closed[0].count, 2U);
  EXPECT_EQ(aggregator.lateEvents(), 0U);
}

TEST(WindowTest, NegativeTimestampsRoundDown) {
  WindowAggregator aggregator({.window_ms = 100});
  aggregator.add("a", 1.0, -1);
  const auto results = aggregator.flush();
  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(results[0].window_start_ms, -100);
}

TEST(WindowTest, KeyLimitBoundsMemoryAndCountsOverflow) {
  WindowAggregator aggregator({.window_ms = 100, .max_keys_per_window = 2});
  aggregator.add("a", 1.0, 0);
  aggregator.add("b", 1.0, 0);
  aggregator.add("c", 1.0, 0);
  aggregator.add("a", 1.0, 0);  // An existing key still counts.

  EXPECT_EQ(aggregator.overflowEvents(), 1U);
  EXPECT_EQ(aggregator.flush().size(), 2U);
}

TEST(WindowTest, AdvanceClosesWindowsWithoutEvents) {
  WindowAggregator aggregator({.window_ms = 100});
  aggregator.add("a", 1.0, 10);
  aggregator.advanceTo(100);
  EXPECT_EQ(aggregator.takeClosed().size(), 1U);
}

TEST(WindowTest, FlushClosesEverythingOldestFirst) {
  WindowAggregator aggregator({.window_ms = 100});
  aggregator.add("a", 1.0, 250);
  aggregator.add("a", 1.0, 10);
  const auto results = aggregator.flush();
  ASSERT_EQ(results.size(), 2U);
  EXPECT_LT(results[0].window_start_ms, results[1].window_start_ms);
}

}  // namespace
