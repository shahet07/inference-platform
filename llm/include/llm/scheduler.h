#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "obs/metrics.h"

namespace llm {

using Token = std::int32_t;

// One sequence's input for one generation step.
struct SeqStep {
  std::int32_t seq = 0;
  std::span<const Token> tokens;  // The whole prompt on the first step, then the previous token.
  std::int32_t position = 0;      // Where in the sequence `tokens` start.
};

// A model that generates one token per sequence per step, keeping each sequence's state between
// steps. Used from one thread, so implementations need no locks.
class Backend {
 public:
  Backend() = default;
  virtual ~Backend() = default;
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;
  Backend(Backend&&) = delete;
  Backend& operator=(Backend&&) = delete;

  [[nodiscard]] virtual std::vector<Token> tokenize(std::string_view text) = 0;
  // Returns each sequence's next token, in order.
  [[nodiscard]] virtual std::vector<Token> step(std::span<const SeqStep> steps) = 0;
  // Forgets a finished sequence, so that its id can be reused.
  virtual void release(std::int32_t seq) = 0;
  [[nodiscard]] virtual bool isEnd(Token token) const = 0;
  [[nodiscard]] virtual std::string text(Token token) const = 0;
};

enum class BatchPolicy : std::uint8_t {
  // New requests join the running batch at the next step, and a finished one's slot is reused at
  // once.
  kContinuous,
  // New requests wait until everything in the running batch has finished. The simple approach, kept
  // for
  // comparison.
  kStatic,
};

struct SchedulerConfig {
  std::size_t max_slots = 8;  // Sequences generating at once.
  std::size_t queue_capacity = 256;
  std::int32_t max_sequence_tokens =
      256;  // The limit on prompt plus output, set by the model's context.
  BatchPolicy policy = BatchPolicy::kContinuous;
};

enum class FinishReason : std::uint8_t {
  kEnd,        // The model produced its end token.
  kMaxTokens,  // Reached the request's limit, or the sequence limit.
  kCancelled,
  kError,  // The prompt did not fit, or the model failed.
};

// Shared between the caller and the scheduler, so that a caller can stop a request at any time.
struct GenerateHandle {
  std::atomic<bool> cancelled{false};
};

struct GenerateRequest {
  std::string prompt;
  std::int32_t max_tokens = 64;
  // Called on the scheduler's thread for each token, so these must be quick.
  std::function<void(std::string_view text)> on_token;
  std::function<void(FinishReason reason)> on_finish;
  std::shared_ptr<GenerateHandle> handle;  // Optional.
};

// Runs requests through a Backend in shared batches, one token per step, and streams tokens as they
// come.
class Scheduler {
 public:
  Scheduler(Backend& backend, const SchedulerConfig& config, obs::Registry& registry);
  // Finishes every request as cancelled that has not already finished.
  ~Scheduler();

  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  // Returns false if the queue is full, in which case the callbacks are never called.
  [[nodiscard]] bool submit(GenerateRequest request);

 private:
  struct Slot {
    std::int32_t seq = 0;
    GenerateRequest request;
    std::vector<Token> pending;  // What goes in on the next step.
    std::int32_t position = 0;
    std::int32_t generated = 0;
    std::chrono::steady_clock::time_point submitted;
  };
  struct Queued {
    GenerateRequest request;
    std::chrono::steady_clock::time_point submitted;
  };

  void run();
  void admit();
  void stepOnce();
  void finish(std::size_t index, FinishReason reason);

  Backend& backend_;
  SchedulerConfig config_;
  obs::Counter& tokens_total_;
  obs::Counter& requests_total_;
  obs::Counter& rejected_total_;
  obs::Counter& cancelled_total_;
  obs::Gauge& active_slots_;
  obs::Gauge& queue_depth_;
  obs::Histogram& time_to_first_token_;
  obs::Histogram& batch_tokens_;

  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Queued> queue_;
  bool stopping_ = false;

  std::vector<Slot> slots_;  // Used only by the scheduler's thread.
  std::vector<std::int32_t> free_seqs_;
  std::thread thread_;
};

}  // namespace llm
