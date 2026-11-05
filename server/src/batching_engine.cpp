#include "server/batching_engine.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace server {
namespace {

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<double> batchSizeBuckets() { return {1, 2, 4, 8, 16, 32, 64, 128, 256}; }

}  // namespace

BatchingEngine::BatchingEngine(BatchRunner runner, std::size_t features_per_row,
                               const EngineConfig& config, obs::Registry& registry)
    : BatchingEngine(std::move(runner),
                     std::vector<std::int64_t>{static_cast<std::int64_t>(features_per_row)}, config,
                     registry) {}

BatchingEngine::BatchingEngine(BatchRunner runner, std::vector<std::int64_t> row_shape,
                               const EngineConfig& config, obs::Registry& registry)
    : runner_(std::move(runner)),
      row_shape_(std::move(row_shape)),
      features_per_row_(static_cast<std::size_t>(std::accumulate(
          row_shape_.begin(), row_shape_.end(), std::int64_t{1}, std::multiplies<>()))),
      config_(config),
      queue_(config.queue_capacity),
      requests_total_(
          registry.counter("inference_requests_total", "Requests accepted.", config.metric_labels)),
      rejected_total_(registry.counter("inference_rejected_total", "Requests rejected: queue full.",
                                       config.metric_labels)),
      batches_total_(registry.counter("inference_batches_total", "Batches sent to the model.",
                                      config.metric_labels)),
      deadline_exceeded_total_(registry.counter("inference_deadline_exceeded_total",
                                                "Requests dropped: deadline passed before running.",
                                                config.metric_labels)),
      queue_depth_(registry.gauge("inference_queue_depth", "Requests waiting for a worker.",
                                  config.metric_labels)),
      batch_size_(registry.histogram("inference_batch_size", "Requests per batch.",
                                     batchSizeBuckets(), config.metric_labels)),
      queue_wait_seconds_(registry.histogram("inference_queue_wait_seconds",
                                             "Time from accepted until sent to the model.",
                                             obs::latencyBuckets(), config.metric_labels)),
      model_run_seconds_(registry.histogram("inference_model_run_seconds",
                                            "Model execution time per batch.",
                                            obs::latencyBuckets(), config.metric_labels)) {
  workers_.reserve(config.workers);
  for (std::size_t i = 0; i < config.workers; ++i) {
    workers_.emplace_back([this] { workerLoop(); });
  }
}

BatchingEngine::~BatchingEngine() {
  queue_.close();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

std::optional<std::future<std::vector<float>>> BatchingEngine::submit(
    std::vector<float> features, obs::TraceContext trace,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  if (features.size() != features_per_row_) {
    throw std::invalid_argument("expected " + std::to_string(features_per_row_) +
                                " features, got " + std::to_string(features.size()));
  }
  Request request{.features = std::move(features),
                  .result = {},
                  .enqueued_at = Clock::now(),
                  .trace = std::move(trace),
                  .deadline = deadline};
  std::future<std::vector<float>> future = request.result.get_future();
  if (!queue_.tryPush(request)) {
    rejected_total_.add();
    return std::nullopt;
  }
  requests_total_.add();
  queue_depth_.add(1);
  return future;
}

void BatchingEngine::workerLoop() {
  perf::Arena arena(std::size_t{1024} * 1024);
  std::vector<Request> batch;
  batch.reserve(config_.max_batch_size);
  while (std::optional<Request> first = queue_.pop()) {
    batch.push_back(std::move(*first));
    const Clock::time_point deadline = batch.front().enqueued_at + config_.max_wait;
    while (batch.size() < config_.max_batch_size) {
      std::optional<Request> next = queue_.popUntil(deadline);
      if (!next) {
        break;
      }
      batch.push_back(std::move(*next));
    }
    runBatch(batch, arena);
    batch.clear();
  }
}

void BatchingEngine::runBatch(std::vector<Request>& batch, perf::Arena& arena) {
  queue_depth_.add(-static_cast<std::int64_t>(batch.size()));
  // Work nobody is waiting for any more would only delay the requests behind it.
  const Clock::time_point now = Clock::now();
  const auto expired = std::ranges::remove_if(batch, [&](Request& request) {
    if (request.deadline && *request.deadline <= now) {
      request.result.set_exception(std::make_exception_ptr(DeadlineExceeded{}));
      deadline_exceeded_total_.add();
      return true;
    }
    return false;
  });
  batch.erase(expired.begin(), expired.end());
  if (batch.empty()) {
    return;
  }
  batches_total_.add();
  batch_size_.observe(static_cast<double>(batch.size()));
  for (const Request& request : batch) {
    queue_wait_seconds_.observe(secondsSince(request.enqueued_at));
  }

  const std::size_t input_size = batch.size() * features_per_row_;
  std::vector<float> heap_inputs;
  float* input_data = nullptr;
  if (config_.use_arena) {
    arena.reset();
    input_data = static_cast<float*>(arena.allocate(input_size * sizeof(float), alignof(float)));
  } else {
    heap_inputs.resize(input_size);
    input_data = heap_inputs.data();
  }
  const std::span<const float> inputs(input_data, input_size);
  for (std::size_t i = 0; i < batch.size(); ++i) {
    std::ranges::copy(batch[i].features, input_data + (i * features_per_row_));
  }
  std::vector<std::int64_t> shape;
  shape.reserve(row_shape_.size() + 1);
  shape.push_back(static_cast<std::int64_t>(batch.size()));
  shape.insert(shape.end(), row_shape_.begin(), row_shape_.end());
  try {
    const Clock::time_point started = Clock::now();
    const std::vector<float> outputs = runner_(inputs, shape);
    model_run_seconds_.observe(secondsSince(started));
    recordSpans(batch, started, Clock::now());
    if (outputs.size() % batch.size() != 0) {
      throw std::runtime_error("model output does not divide evenly across the batch");
    }
    const std::size_t per_row = outputs.size() / batch.size();
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const auto begin = outputs.begin() + static_cast<std::ptrdiff_t>(i * per_row);
      batch[i].result.set_value(
          std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(per_row)));
    }
  } catch (...) {
    for (Request& request : batch) {
      request.result.set_exception(std::current_exception());
    }
  }
}

void BatchingEngine::recordSpans(const std::vector<Request>& batch, Clock::time_point run_start,
                                 Clock::time_point run_end) const {
  if (config_.tracer == nullptr) {
    return;
  }
  // Steady-clock durations, anchored to wall time so that spans line up across processes.
  const auto wall_now = std::chrono::system_clock::now();
  const auto steady_now = Clock::now();
  const auto toWall = [&](Clock::time_point t) {
    return wall_now -
           std::chrono::duration_cast<std::chrono::system_clock::duration>(steady_now - t);
  };
  const auto micros = [](Clock::duration d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d);
  };
  for (const Request& request : batch) {
    if (!request.trace.sampled) {
      continue;
    }
    config_.tracer->record({.trace_id = request.trace.trace_id,
                            .span_id = obs::newSpanId(),
                            .parent_span_id = request.trace.span_id,
                            .name = "engine.queue_wait",
                            .start = toWall(request.enqueued_at),
                            .duration = micros(run_start - request.enqueued_at),
                            .attributes = {}});
    config_.tracer->record({.trace_id = request.trace.trace_id,
                            .span_id = obs::newSpanId(),
                            .parent_span_id = request.trace.span_id,
                            .name = "model.run",
                            .start = toWall(run_start),
                            .duration = micros(run_end - run_start),
                            .attributes = {{"batch_size", std::to_string(batch.size())}}});
  }
}

}  // namespace server
