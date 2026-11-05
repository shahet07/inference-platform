#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "cluster/health_monitor.h"
#include "obs/log.h"
#include "obs/metrics.h"
#include "obs/trace.h"

namespace cluster {

struct RouterConfig {
  std::vector<std::string> backends;  // "host:port"
  std::size_t virtual_nodes = 128;
  std::size_t connection_threads = 256;
  // Nodes tried for one request. Inference has no side effects, so retrying is safe.
  std::size_t max_attempts = 2;
  HealthConfig health;
  // Optional, and must outlive the router.
  obs::Logger* logger = nullptr;
  obs::Tracer* tracer = nullptr;
};

// Sends each request to a backend chosen by consistent hashing of its X-Routing-Key header
// (falling back to X-Client-Id), so a key stays on one node while the cluster is stable.
//   POST /v1/infer    Forwarded; the response names the backend in X-Served-By
//   GET  /v1/cluster  Each backend's health
//   GET  /healthz     200 while at least one backend is healthy
//   GET  /metrics     Prometheus text format
class Router {
 public:
  Router(const RouterConfig& config, obs::Registry& registry);
  ~Router();

  Router(const Router&) = delete;
  Router& operator=(const Router&) = delete;
  Router(Router&&) = delete;
  Router& operator=(Router&&) = delete;

  // Binds to `port`, or to any free port if it is 0. Returns the bound port, or -1 on failure.
  int bind(const std::string& host, int port);
  // Blocks until stop() is called.
  void serve();
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cluster
