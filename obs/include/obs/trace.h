#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace obs {

// The W3C Trace Context fields carried between services in the `traceparent` header.
struct TraceContext {
  std::string trace_id;  // 32 lowercase hex digits.
  std::string span_id;   // 16 lowercase hex digits: the caller's span.
  bool sampled = false;

  [[nodiscard]] bool valid() const { return !trace_id.empty(); }
};

// Returns nullopt for anything malformed, so a bad header starts a new trace, not an error.
[[nodiscard]] std::optional<TraceContext> parseTraceparent(std::string_view header);
[[nodiscard]] std::string formatTraceparent(const TraceContext& context);
[[nodiscard]] std::string newTraceId();
[[nodiscard]] std::string newSpanId();

struct Span {
  std::string trace_id;
  std::string span_id;
  std::string parent_span_id;  // Empty for a trace's first span.
  std::string name;
  std::chrono::system_clock::time_point start;
  std::chrono::microseconds duration{0};
  std::vector<std::pair<std::string, std::string>> attributes;
};

// Writes finished spans to a file as JSON lines from a background thread. A full queue drops the
// span and counts it, so tracing can never slow down or block a request.
class Tracer {
 public:
  // Starts one trace in every `sample_every` requests that arrive without a trace.
  Tracer(const std::string& path, std::string service, std::uint64_t sample_every,
         std::size_t queue_capacity = 65536);
  // Writes everything still queued.
  ~Tracer();

  Tracer(const Tracer&) = delete;
  Tracer& operator=(const Tracer&) = delete;
  Tracer(Tracer&&) = delete;
  Tracer& operator=(Tracer&&) = delete;

  // Continues `incoming` if it is valid, following its sampling decision, or else starts a trace.
  // The result's span_id is new, for this service's span.
  [[nodiscard]] TraceContext startOrContinue(const std::optional<TraceContext>& incoming);

  void record(Span span);

  [[nodiscard]] std::uint64_t droppedSpans() const {
    return dropped_.load(std::memory_order_relaxed);
  }

 private:
  void writeLoop();

  std::ofstream file_;
  std::string service_;
  std::uint64_t sample_every_;
  std::size_t capacity_;
  std::atomic<std::uint64_t> requests_seen_{0};
  std::atomic<std::uint64_t> dropped_{0};
  std::mutex mutex_;
  std::condition_variable has_spans_;
  std::deque<Span> queue_;
  bool stopping_ = false;
  std::thread writer_;
};

}  // namespace obs
