#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "cluster/model_registry.h"
#include "obs/log.h"
#include "obs/metrics.h"
#include "obs/trace.h"
#include "server/batching_engine.h"
#include "stream/processor.h"
#include "stream/sharded_log.h"

namespace server {

// HTTP front end for a BatchingEngine.
//   POST /v1/infer  {"inputs":[...]} -> {"outputs":[...]}; 503 when the queue is full
//   GET  /v1/stats  Closed per-client latency windows, when an event stream is attached
//   GET  /v1/models        Model metadata, when a registry is attached
//   PUT  /v1/models/{name} {"uri":"..."} records a new version of a model's metadata
//   GET  /healthz   200 while serving
//   GET  /metrics   Prometheus text format
class HttpServer {
 public:
  // `connection_threads` caps concurrent connections, since each one holds a thread.
  // `tcp_nodelay` stops headers and body, which are written separately, from stalling on Nagle's
  // algorithm and delayed ACKs. `keep_alive_max_requests` is how many requests a connection
  // serves before it is closed; the library default of 100 forces frequent reconnects. See
  // docs/benchmarks.md for measurements of both.
  HttpServer(BatchingEngine& engine, obs::Registry& registry, std::size_t connection_threads,
             bool tcp_nodelay = true, std::size_t keep_alive_max_requests = 1'000'000);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;
  HttpServer(HttpServer&&) = delete;
  HttpServer& operator=(HttpServer&&) = delete;

  // Records one event per request, keyed by the X-Client-Id header, with latency in ms as its
  // value. Call before serve(). Both must outlive the server. Use a log with a drop policy, so
  // that analytics can never delay inference.
  void attachEventStream(stream::ShardedLog& log, const stream::StreamProcessor& processor);

  // Serves model metadata from `registry`, which must outlive the server. Call before serve().
  void attachModelRegistry(cluster::ModelRegistry& registry);

  // Logs errors, rate-limited so that overload cannot flood the log. Must outlive the server.
  void attachLogger(obs::Logger& logger);
  // Records a span per sampled request and passes the trace to the engine. Must outlive the server.
  void attachTracer(obs::Tracer& tracer);

  // Binds to `port`, or to any free port if it is 0. Returns the bound port, or -1 on failure.
  int bind(const std::string& host, int port);
  // Blocks until stop() is called.
  void serve();
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace server
