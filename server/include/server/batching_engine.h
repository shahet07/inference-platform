#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "obs/metrics.h"
#include "obs/trace.h"
#include "perf/arena.h"
#include "server/bounded_queue.h"

namespace server {

struct EngineConfig {
  std::size_t workers = 1;
  std::size_t max_batch_size = 8;
  // 0 batches only what queued while the workers were busy, so an idle server never delays a
  // request. Raise it for models where batching saves more time than the wait costs. See
  // docs/benchmarks.md.
  std::chrono::microseconds max_wait{0};
  std::size_t queue_capacity = 1024;
  // Assembles each batch's input in a per-worker arena instead of a new vector.
  bool use_arena = false;
  // When set, sampled requests record queue-wait and model-run spans. Must outlive the engine.
  obs::Tracer* tracer = nullptr;
  // Gives each worker its own core, on Linux. Elsewhere this does nothing, and macOS has no such
  // control. Can help when other work on the machine would otherwise move workers between cores.
  bool pin_workers = false;
  // Added to this engine's metrics, like: model="resnet"
  std::string metric_labels;
};

// Set on a request's future when its deadline passed before it reached the model.
class DeadlineExceeded : public std::runtime_error {
 public:
  DeadlineExceeded() : std::runtime_error("deadline exceeded") {}
};

// Runs a model on a flattened batch and returns the flattened outputs.
using BatchRunner =
    std::function<std::vector<float>(std::span<const float>, std::span<const std::int64_t>)>;

// Groups single requests into batches. A batch is sent to the model when it reaches
// max_batch_size or when its oldest request has waited max_wait, whichever comes first.
class BatchingEngine {
 public:
  // `row_shape` is one request's dimensions: {64} for a flat model, {3, 224, 224} for an image
  // model.
  BatchingEngine(BatchRunner runner, std::vector<std::int64_t> row_shape,
                 const EngineConfig& config, obs::Registry& registry);
  BatchingEngine(BatchRunner runner, std::size_t features_per_row, const EngineConfig& config,
                 obs::Registry& registry);
  // Finishes every request already accepted, then stops the workers.
  ~BatchingEngine();

  BatchingEngine(const BatchingEngine&) = delete;
  BatchingEngine& operator=(const BatchingEngine&) = delete;
  BatchingEngine(BatchingEngine&&) = delete;
  BatchingEngine& operator=(BatchingEngine&&) = delete;

  // Returns nullopt if the queue is full, so callers can shed load instead of queueing forever.
  // Throws std::invalid_argument if `features` has the wrong size.
  [[nodiscard]] std::optional<std::future<std::vector<float>>> submit(
      std::vector<float> features, obs::TraceContext trace = {},
      std::optional<std::chrono::steady_clock::time_point> deadline = std::nullopt);

  // Called on a worker thread when a request finishes, with its outputs or its error. Must be
  // quick.
  using Completion = std::function<void(std::vector<float>, std::exception_ptr)>;
  // Like submit, for callers that must not block on a future, such as an event loop. Returns false
  // if the queue is full, in which case `done` is not called.
  [[nodiscard]] bool submitAsync(
      std::vector<float> features, Completion done,
      std::optional<std::chrono::steady_clock::time_point> deadline = std::nullopt);

  [[nodiscard]] std::size_t featuresPerRow() const { return features_per_row_; }

 private:
  struct Request {
    std::vector<float> features;
    std::promise<std::vector<float>> result;
    std::chrono::steady_clock::time_point enqueued_at;
    obs::TraceContext trace;
    std::optional<std::chrono::steady_clock::time_point> deadline;
    Completion done;  // When set, this is used in place of `result`.

    void succeed(std::vector<float> outputs);
    void fail(std::exception_ptr error);
  };

  void workerLoop();
  void runBatch(std::vector<Request>& batch, perf::Arena& arena);
  void recordSpans(const std::vector<Request>& batch,
                   std::chrono::steady_clock::time_point run_start,
                   std::chrono::steady_clock::time_point run_end) const;

  BatchRunner runner_;
  std::vector<std::int64_t> row_shape_;
  std::size_t features_per_row_;
  EngineConfig config_;
  BoundedQueue<Request> queue_;
  obs::Counter& requests_total_;
  obs::Counter& rejected_total_;
  obs::Counter& batches_total_;
  obs::Counter& deadline_exceeded_total_;
  obs::Gauge& queue_depth_;
  obs::Histogram& batch_size_;
  obs::Histogram& queue_wait_seconds_;
  obs::Histogram& model_run_seconds_;
  std::vector<std::thread> workers_;
};

}  // namespace server
