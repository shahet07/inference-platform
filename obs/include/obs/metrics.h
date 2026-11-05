#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace obs {

class Counter {
 public:
  void add(std::uint64_t n = 1) { value_.fetch_add(n, std::memory_order_relaxed); }
  [[nodiscard]] std::uint64_t value() const { return value_.load(std::memory_order_relaxed); }

 private:
  std::atomic<std::uint64_t> value_{0};
};

class Gauge {
 public:
  void set(std::int64_t v) { value_.store(v, std::memory_order_relaxed); }
  void add(std::int64_t n) { value_.fetch_add(n, std::memory_order_relaxed); }
  [[nodiscard]] std::int64_t value() const { return value_.load(std::memory_order_relaxed); }

 private:
  std::atomic<std::int64_t> value_{0};
};

// Fixed-bucket histogram. Observing takes no lock, so it is safe on the request path.
class Histogram {
 public:
  explicit Histogram(std::vector<double> upper_bounds);

  void observe(double value);

  [[nodiscard]] std::uint64_t count() const { return count_.load(std::memory_order_relaxed); }
  [[nodiscard]] double sum() const { return sum_.load(std::memory_order_relaxed); }
  // Estimated as the upper bound of the bucket holding the quantile; 0 when empty.
  [[nodiscard]] double quantile(double q) const;
  [[nodiscard]] const std::vector<double>& upperBounds() const { return upper_bounds_; }
  // One entry per bound plus a final +Inf entry. Not cumulative.
  [[nodiscard]] std::uint64_t bucketCount(std::size_t index) const;

 private:
  std::vector<double> upper_bounds_;
  std::deque<std::atomic<std::uint64_t>> buckets_;
  std::atomic<std::uint64_t> count_{0};
  std::atomic<double> sum_{0.0};
};

// Latency buckets from 50 microseconds to about 13 seconds, in seconds.
std::vector<double> latencyBuckets();

// Owns metrics and renders them in the Prometheus text format. Asking again for the same name and
// labels returns the same metric, so counts carry on when, say, a model is replaced. Returned
// references stay valid for the registry's lifetime. `labels` looks like: model="resnet"
class Registry {
 public:
  Counter& counter(const std::string& name, const std::string& help,
                   const std::string& labels = "");
  Gauge& gauge(const std::string& name, const std::string& help, const std::string& labels = "");
  Histogram& histogram(const std::string& name, const std::string& help,
                       std::vector<double> upper_bounds, const std::string& labels = "");

  // Also includes process_cpu_seconds_total, read when rendering.
  [[nodiscard]] std::string renderPrometheus() const;

 private:
  template <typename T>
  struct Entry {
    template <typename... Args>
    Entry(std::string n, std::string h, std::string l, Args&&... args)
        : name(std::move(n)),
          help(std::move(h)),
          labels(std::move(l)),
          metric(std::forward<Args>(args)...) {}
    std::string name;
    std::string help;
    std::string labels;
    T metric;
  };

  mutable std::mutex mutex_;
  std::deque<Entry<Counter>> counters_;
  std::deque<Entry<Gauge>> gauges_;
  std::deque<Entry<Histogram>> histograms_;
};

}  // namespace obs
