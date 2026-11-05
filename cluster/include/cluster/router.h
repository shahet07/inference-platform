#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cluster/health_monitor.h"
#include "obs/log.h"
#include "obs/metrics.h"
#include "obs/trace.h"
#include "security/api_keys.h"

namespace cluster {

enum class BalancePolicy : std::uint8_t {
  // A key stays on one node while the cluster is stable, which suits per-key state or caches. Load
  // follows the keys, so a few busy keys can overload a node. Requests with no key use
  // kLeastLoaded.
  kConsistentHash,
  // Two random backends are compared, and the one with fewer requests in flight is used. Ignores
  // keys.
  kLeastLoaded,
};

struct RouterConfig {
  std::vector<std::string> backends;  // "host:port"
  BalancePolicy balance = BalancePolicy::kConsistentHash;
  // How long a backend may take to answer before it counts as failed. A request's own deadline
  // takes precedence, and running past that does not count against the backend.
  std::chrono::milliseconds backend_timeout{30000};
  // Optional "name:port". The name is resolved periodically, and backends follow the addresses it
  // returns, which is how pods behind a Kubernetes headless Service are found.
  std::string discover_dns;
  std::chrono::milliseconds discover_interval{2000};
  std::size_t virtual_nodes = 128;
  std::size_t connection_threads = 256;
  // Nodes tried for one request. Inference has no side effects, so retrying is safe.
  std::size_t max_attempts = 2;
  HealthConfig health;
  // Serve HTTPS, which needs a build with IP_ENABLE_TLS. Throws otherwise.
  std::string tls_cert;
  std::string tls_key;
  // Talk to backends over HTTPS, checking their certificates against this CA file.
  std::string backend_ca;
  // Optional, and must outlive the router. With keys, requests need a valid key, and adding or
  // removing backends needs an admin one. The Authorization header is forwarded, so backends can
  // check it too.
  security::ApiKeys* api_keys = nullptr;
  obs::Logger* logger = nullptr;
  obs::Tracer* tracer = nullptr;
};

// Sends each request to a backend chosen by consistent hashing of its X-Routing-Key header
// (falling back to X-Client-Id), so a key stays on one node while the cluster is stable.
//   POST /v1/infer, /v1/models/{name}/infer    Forwarded; X-Served-By names the backend
//   GET  /v1/cluster  Each backend's health
//   POST /v1/backends {"backend":"host:port"}  Adds a backend
//   DELETE /v1/backends/{host:port}            Removes one; requests already on it still finish
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
