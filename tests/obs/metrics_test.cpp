#include "obs/metrics.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <thread>
#include <vector>

namespace {

using testing::HasSubstr;

TEST(HistogramTest, QuantileReturnsUpperBoundOfBucket) {
  obs::Histogram histogram({1.0, 2.0, 4.0});
  for (int i = 0; i < 99; ++i) {
    histogram.observe(0.5);
  }
  histogram.observe(3.0);

  EXPECT_EQ(histogram.count(), 100U);
  EXPECT_DOUBLE_EQ(histogram.quantile(0.5), 1.0);
  EXPECT_DOUBLE_EQ(histogram.quantile(0.995), 4.0);
}

TEST(HistogramTest, EmptyQuantileIsZero) {
  const obs::Histogram histogram({1.0});
  EXPECT_DOUBLE_EQ(histogram.quantile(0.5), 0.0);
}

TEST(RegistryTest, RendersCumulativePrometheusBuckets) {
  obs::Registry registry;
  registry.counter("requests_total", "Requests.").add(3);
  registry.gauge("depth", "Depth.").set(7);
  obs::Histogram& histogram = registry.histogram("latency", "Latency.", {1.0, 2.0});
  histogram.observe(0.5);
  histogram.observe(1.5);
  histogram.observe(9.0);

  const std::string text = registry.renderPrometheus();

  EXPECT_THAT(text, HasSubstr("# TYPE requests_total counter\nrequests_total 3\n"));
  EXPECT_THAT(text, HasSubstr("depth 7\n"));
  EXPECT_THAT(text, HasSubstr("latency_bucket{le=\"1\"} 1\n"));
  EXPECT_THAT(text, HasSubstr("latency_bucket{le=\"2\"} 2\n"));
  EXPECT_THAT(text, HasSubstr("latency_bucket{le=\"+Inf\"} 3\n"));
  EXPECT_THAT(text, HasSubstr("latency_count 3\n"));
  EXPECT_THAT(text, HasSubstr("process_cpu_seconds_total"));
}

TEST(RegistryTest, SameNameAndLabelsReturnSameMetric) {
  obs::Registry registry;
  registry.counter("c", "Help.", "model=\"a\"").add(1);
  registry.counter("c", "Help.", "model=\"a\"").add(2);
  EXPECT_EQ(registry.counter("c", "Help.", "model=\"a\"").value(), 3U);
  EXPECT_EQ(registry.counter("c", "Help.", "model=\"b\"").value(), 0U);
}

TEST(RegistryTest, RendersLabelsWithOneHeaderPerMetric) {
  obs::Registry registry;
  registry.counter("c", "Help.", "model=\"a\"").add(1);
  registry.gauge("g", "Help.");
  registry.counter("c", "Help.", "model=\"b\"").add(2);
  registry.histogram("h", "Help.", {1.0}, "model=\"a\"").observe(0.5);

  const std::string text = registry.renderPrometheus();

  EXPECT_THAT(text, HasSubstr("# TYPE c counter\nc{model=\"a\"} 1\nc{model=\"b\"} 2\n"));
  EXPECT_THAT(text, HasSubstr("h_bucket{model=\"a\",le=\"1\"} 1\n"));
  EXPECT_THAT(text, HasSubstr("h_count{model=\"a\"} 1\n"));
}

TEST(CounterTest, ConcurrentAddsAreNotLost) {
  obs::Counter counter;
  std::vector<std::thread> threads;
  threads.reserve(4);
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&counter] {
      for (int i = 0; i < 10000; ++i) {
        counter.add();
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(counter.value(), 40000U);
}

}  // namespace
