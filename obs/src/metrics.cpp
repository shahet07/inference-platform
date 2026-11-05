#include "obs/metrics.h"

#include <sys/resource.h>

#include <algorithm>
#include <sstream>
#include <utility>

namespace obs {
namespace {

double processCpuSeconds() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  const auto seconds = [](const timeval& t) {
    return static_cast<double>(t.tv_sec) + (static_cast<double>(t.tv_usec) / 1e6);
  };
  return seconds(usage.ru_utime) + seconds(usage.ru_stime);
}

}  // namespace

Histogram::Histogram(std::vector<double> upper_bounds)
    : upper_bounds_(std::move(upper_bounds)), buckets_(upper_bounds_.size() + 1) {
  std::ranges::sort(upper_bounds_);
}

void Histogram::observe(double value) {
  const auto it = std::ranges::lower_bound(upper_bounds_, value);
  buckets_[static_cast<std::size_t>(it - upper_bounds_.begin())].fetch_add(
      1, std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
  double current = sum_.load(std::memory_order_relaxed);
  while (!sum_.compare_exchange_weak(current, current + value, std::memory_order_relaxed)) {
  }
}

double Histogram::quantile(double q) const {
  const std::uint64_t total = count();
  if (total == 0 || upper_bounds_.empty()) {
    return 0.0;
  }
  const auto target = static_cast<std::uint64_t>(q * static_cast<double>(total));
  std::uint64_t seen = 0;
  for (std::size_t i = 0; i < upper_bounds_.size(); ++i) {
    seen += bucketCount(i);
    if (seen > target) {
      return upper_bounds_[i];
    }
  }
  return upper_bounds_.back();
}

std::uint64_t Histogram::bucketCount(std::size_t index) const {
  return buckets_[index].load(std::memory_order_relaxed);
}

std::vector<double> latencyBuckets() {
  constexpr int kBuckets = 19;
  std::vector<double> bounds;
  bounds.reserve(kBuckets);
  double bound = 50e-6;
  for (int i = 0; i < kBuckets; ++i) {
    bounds.push_back(bound);
    bound *= 2.0;
  }
  return bounds;
}

Counter& Registry::counter(const std::string& name, const std::string& help) {
  const std::scoped_lock lock(mutex_);
  return counters_.emplace_back(name, help).metric;
}

Gauge& Registry::gauge(const std::string& name, const std::string& help) {
  const std::scoped_lock lock(mutex_);
  return gauges_.emplace_back(name, help).metric;
}

Histogram& Registry::histogram(const std::string& name, const std::string& help,
                               std::vector<double> upper_bounds) {
  const std::scoped_lock lock(mutex_);
  return histograms_.emplace_back(name, help, std::move(upper_bounds)).metric;
}

std::string Registry::renderPrometheus() const {
  const std::scoped_lock lock(mutex_);
  std::ostringstream out;
  const auto header = [&out](const std::string& name, const std::string& help, const char* type) {
    out << "# HELP " << name << ' ' << help << "\n# TYPE " << name << ' ' << type << '\n';
  };
  for (const auto& e : counters_) {
    header(e.name, e.help, "counter");
    out << e.name << ' ' << e.metric.value() << '\n';
  }
  for (const auto& e : gauges_) {
    header(e.name, e.help, "gauge");
    out << e.name << ' ' << e.metric.value() << '\n';
  }
  for (const auto& e : histograms_) {
    header(e.name, e.help, "histogram");
    std::uint64_t cumulative = 0;
    const auto& bounds = e.metric.upperBounds();
    for (std::size_t i = 0; i < bounds.size(); ++i) {
      cumulative += e.metric.bucketCount(i);
      out << e.name << "_bucket{le=\"" << bounds[i] << "\"} " << cumulative << '\n';
    }
    cumulative += e.metric.bucketCount(bounds.size());
    out << e.name << "_bucket{le=\"+Inf\"} " << cumulative << '\n';
    out << e.name << "_sum " << e.metric.sum() << '\n';
    out << e.name << "_count " << e.metric.count() << '\n';
  }
  header("process_cpu_seconds_total", "User and system CPU time used by this process.", "counter");
  out << "process_cpu_seconds_total " << processCpuSeconds() << '\n';
  return out.str();
}

}  // namespace obs
