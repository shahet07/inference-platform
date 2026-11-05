#include "obs/trace.h"

#include <algorithm>
#include <random>
#include <stdexcept>

#include "obs/log.h"

namespace obs {
namespace {

bool isLowerHex(std::string_view text) {
  return std::ranges::all_of(
      text, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

bool isAllZero(std::string_view text) {
  return std::ranges::all_of(text, [](char c) { return c == '0'; });
}

std::string randomHex(std::size_t digits) {
  thread_local std::mt19937_64 rng{std::random_device{}()};
  constexpr std::string_view kHex = "0123456789abcdef";
  std::string out;
  out.reserve(digits);
  std::uint64_t bits = 0;
  for (std::size_t i = 0; i < digits; ++i) {
    if (i % 16 == 0) {
      bits = rng();
    }
    out += kHex[bits & 0xFU];
    bits >>= 4U;
  }
  return out;
}

}  // namespace

std::optional<TraceContext> parseTraceparent(std::string_view header) {
  // version "-" trace-id "-" parent-id "-" flags: 2 + 1 + 32 + 1 + 16 + 1 + 2 = 55 characters.
  constexpr std::size_t kLength = 55;
  if (header.size() != kLength || header[2] != '-' || header[35] != '-' || header[52] != '-') {
    return std::nullopt;
  }
  const std::string_view version = header.substr(0, 2);
  const std::string_view trace_id = header.substr(3, 32);
  const std::string_view span_id = header.substr(36, 16);
  const std::string_view flags = header.substr(53, 2);
  if (!isLowerHex(version) || version == "ff" || !isLowerHex(trace_id) || !isLowerHex(span_id) ||
      !isLowerHex(flags) || isAllZero(trace_id) || isAllZero(span_id)) {
    return std::nullopt;
  }
  const auto flag_bits = static_cast<unsigned>(std::stoi(std::string(flags), nullptr, 16));
  return TraceContext{.trace_id = std::string(trace_id),
                      .span_id = std::string(span_id),
                      .sampled = (flag_bits & 1U) != 0};
}

std::string formatTraceparent(const TraceContext& context) {
  return "00-" + context.trace_id + "-" + context.span_id + (context.sampled ? "-01" : "-00");
}

std::string newTraceId() { return randomHex(32); }

std::string newSpanId() { return randomHex(16); }

Tracer::Tracer(const std::string& path, std::string service, std::uint64_t sample_every,
               std::size_t queue_capacity)
    : file_(path, std::ios::app),
      service_(std::move(service)),
      sample_every_(sample_every),
      capacity_(queue_capacity) {
  if (!file_) {
    throw std::runtime_error("cannot open trace file: " + path);
  }
  writer_ = std::thread([this] { writeLoop(); });
}

Tracer::~Tracer() {
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
  }
  has_spans_.notify_all();
  writer_.join();
}

TraceContext Tracer::startOrContinue(const std::optional<TraceContext>& incoming) {
  if (incoming) {
    return {.trace_id = incoming->trace_id, .span_id = newSpanId(), .sampled = incoming->sampled};
  }
  const std::uint64_t seen = requests_seen_.fetch_add(1, std::memory_order_relaxed);
  const bool sampled = sample_every_ > 0 && seen % sample_every_ == 0;
  // Unsampled traces get ids too. Callees follow the flag in the header they are sent, so one
  // decision covers the whole trace, instead of each service sampling a different part of it.
  return {.trace_id = newTraceId(), .span_id = newSpanId(), .sampled = sampled};
}

void Tracer::record(Span span) {
  {
    const std::scoped_lock lock(mutex_);
    if (queue_.size() >= capacity_) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    queue_.push_back(std::move(span));
  }
  has_spans_.notify_one();
}

void Tracer::writeLoop() {
  std::deque<Span> batch;
  while (true) {
    {
      std::unique_lock lock(mutex_);
      has_spans_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) {
        return;  // Only reachable when stopping, with everything written.
      }
      batch.swap(queue_);
    }
    for (const Span& span : batch) {
      file_ << R"({"trace_id":")" << span.trace_id << R"(","span_id":")" << span.span_id
            << R"(","parent_span_id":")" << span.parent_span_id << R"(","service":")"
            << jsonEscape(service_) << R"(","name":")" << jsonEscape(span.name)
            << R"(","start_us":)"
            << std::chrono::duration_cast<std::chrono::microseconds>(span.start.time_since_epoch())
                   .count()
            << R"(,"duration_us":)" << span.duration.count() << R"(,"attributes":{)";
      bool first = true;
      for (const auto& [key, value] : span.attributes) {
        file_ << (first ? "" : ",") << '"' << jsonEscape(key) << "\":\"" << jsonEscape(value)
              << '"';
        first = false;
      }
      file_ << "}}\n";
    }
    file_.flush();
    batch.clear();
  }
}

}  // namespace obs
