#include "stream/sliding_window.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

using stream::SlidingWindowAggregator;
using stream::WindowResult;

constexpr stream::SlidingWindowConfig kThreeBySecond{.window_ms = 3000, .slide_ms = 1000};

std::map<std::int64_t, std::uint64_t> countsByEnd(const std::vector<WindowResult>& results) {
  std::map<std::int64_t, std::uint64_t> counts;
  for (const WindowResult& result : results) {
    counts[result.window_end_ms] += result.count;
  }
  return counts;
}

TEST(SlidingWindowTest, EventFallsInEveryOverlappingWindow) {
  SlidingWindowAggregator aggregator(kThreeBySecond);
  aggregator.add("a", 5.0, 1500);
  aggregator.advanceTo(10000);

  const auto counts = countsByEnd(aggregator.takeClosed());

  // Windows ending at 2000, 3000, and 4000 hold 1500.
  EXPECT_EQ(counts, (std::map<std::int64_t, std::uint64_t>{{2000, 1}, {3000, 1}, {4000, 1}}));
}

TEST(SlidingWindowTest, MergesPanesWithCorrectStatistics) {
  SlidingWindowAggregator aggregator(kThreeBySecond);
  aggregator.add("a", 1.0, 500);
  aggregator.add("a", 3.0, 1500);
  aggregator.add("a", 8.0, 2500);
  aggregator.advanceTo(3000);

  std::vector<WindowResult> results = aggregator.takeClosed();

  ASSERT_FALSE(results.empty());
  const WindowResult& last = results.back();
  EXPECT_EQ(last.window_start_ms, 0);
  EXPECT_EQ(last.window_end_ms, 3000);
  EXPECT_EQ(last.count, 3U);
  EXPECT_DOUBLE_EQ(last.sum, 12.0);
  EXPECT_DOUBLE_EQ(last.min, 1.0);
  EXPECT_DOUBLE_EQ(last.max, 8.0);
}

TEST(SlidingWindowTest, MatchesABruteForceCalculation) {
  // Compared with computing every window directly, which is the definition.
  SlidingWindowAggregator aggregator(kThreeBySecond);
  std::vector<std::int64_t> times;
  for (std::int64_t t = 0; t < 20000; t += 137) {
    times.push_back(t);
    aggregator.add("k", 1.0, t);
  }
  aggregator.advanceTo(100000);
  const auto counts = countsByEnd(aggregator.takeClosed());

  std::map<std::int64_t, std::uint64_t> expected;
  for (const std::int64_t t : times) {
    const std::int64_t pane = t - (t % 1000);
    for (std::int64_t end = pane + 1000; end <= pane + 3000; end += 1000) {
      ++expected[end];
    }
  }
  EXPECT_EQ(counts, expected);
}

TEST(SlidingWindowTest, KeysStaySeparate) {
  SlidingWindowAggregator aggregator({.window_ms = 1000, .slide_ms = 1000});
  aggregator.add("a", 1.0, 100);
  aggregator.add("b", 2.0, 100);
  aggregator.advanceTo(1000);
  EXPECT_EQ(aggregator.takeClosed().size(), 2U);
}

TEST(SlidingWindowTest, LateEventsReachOnlyOpenWindows) {
  SlidingWindowAggregator aggregator(kThreeBySecond);
  aggregator.add("a", 1.0, 500);
  aggregator.advanceTo(1000);
  ASSERT_EQ(aggregator.takeClosed().size(), 1U);  // The window ending at 1000 has closed.

  aggregator.add("a", 1.0, 900);  // In a closed window, and in two still open.
  aggregator.advanceTo(100000);

  EXPECT_EQ(countsByEnd(aggregator.takeClosed()),
            (std::map<std::int64_t, std::uint64_t>{{2000, 2}, {3000, 2}}));
  EXPECT_EQ(aggregator.lateEvents(), 0U);
}

TEST(SlidingWindowTest, EventTooOldForAnyWindowIsCounted) {
  SlidingWindowAggregator aggregator(kThreeBySecond);
  aggregator.add("a", 1.0, 100);
  aggregator.advanceTo(100000);
  (void)aggregator.takeClosed();
  aggregator.add("a", 1.0, 200);
  EXPECT_EQ(aggregator.lateEvents(), 1U);
}

TEST(SlidingWindowTest, LongIdleGapDoesNotStepThroughEmptyWindows) {
  SlidingWindowAggregator aggregator({.window_ms = 2, .slide_ms = 1});
  aggregator.add("a", 1.0, 0);
  constexpr std::int64_t kFarAway = 1'000'000'000'000'000;  // Stepping there would never finish.
  aggregator.add("a", 1.0, kFarAway);
  aggregator.advanceTo(kFarAway + 10);
  EXPECT_EQ(aggregator.takeClosed().size(), 4U);
}

TEST(SlidingWindowTest, ExtremeTimestampsDoNotOverflow) {
  SlidingWindowAggregator aggregator(kThreeBySecond);
  aggregator.add("low", 1.0, std::numeric_limits<std::int64_t>::min());
  aggregator.add("high", 1.0, std::numeric_limits<std::int64_t>::max());
  (void)aggregator.takeClosed();
}

TEST(SlidingWindowTest, InvalidSizesAreRejected) {
  EXPECT_THROW(SlidingWindowAggregator({.window_ms = 1000, .slide_ms = 300}),
               std::invalid_argument);
  EXPECT_THROW(SlidingWindowAggregator({.window_ms = 1000, .slide_ms = 0}), std::invalid_argument);
  EXPECT_THROW(SlidingWindowAggregator({.window_ms = -1000, .slide_ms = 1000}),
               std::invalid_argument);
}

}  // namespace
