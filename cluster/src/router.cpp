#include "cluster/router.h"

#include <httplib.h>

#include <chrono>
#include <initializer_list>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <shared_mutex>
#include <string_view>
#include <utility>

#include "cluster/hash_ring.h"

namespace cluster {
namespace {

constexpr const char* kJson = "application/json";
using LogFields = std::initializer_list<obs::LogField>;

// Reuses keep-alive connections. An httplib::Client serializes its own requests, so each
// in-flight request takes a client of its own.
class ClientPool {
 public:
  explicit ClientPool(std::string backend) : backend_(std::move(backend)) {}

  std::unique_ptr<httplib::Client> acquire() {
    {
      const std::scoped_lock lock(mutex_);
      if (!idle_.empty()) {
        auto client = std::move(idle_.back());
        idle_.pop_back();
        return client;
      }
    }
    auto client = std::make_unique<httplib::Client>("http://" + backend_);
    client->set_keep_alive(true);
    client->set_tcp_nodelay(true);
    client->set_connection_timeout(std::chrono::milliseconds(200));
    client->set_read_timeout(std::chrono::seconds(30));
    return client;
  }

  void release(std::unique_ptr<httplib::Client> client) {
    const std::scoped_lock lock(mutex_);
    idle_.push_back(std::move(client));
  }

  // Connections to a failed node are dead, and reusing them would fail again.
  void clear() {
    const std::scoped_lock lock(mutex_);
    idle_.clear();
  }

 private:
  std::string backend_;
  std::mutex mutex_;
  std::vector<std::unique_ptr<httplib::Client>> idle_;
};

bool probeHealthz(const std::string& backend) {
  httplib::Client client("http://" + backend);
  client.set_connection_timeout(std::chrono::milliseconds(200));
  client.set_read_timeout(std::chrono::milliseconds(500));
  const auto response = client.Get("/healthz");
  return response && response->status == 200;
}

}  // namespace

struct Router::Impl {
  Impl(const RouterConfig& c, obs::Registry& r)
      : config(c),
        registry(r),
        ring(c.virtual_nodes),
        requests_total(r.counter("router_requests_total", "Requests received.")),
        failovers_total(
            r.counter("router_failovers_total", "Requests retried on another backend.")),
        unavailable_total(
            r.counter("router_unavailable_total", "Requests failed: no backend could serve them.")),
        healthy_backends(r.gauge("router_healthy_backends", "Backends in rotation.")),
        forward_seconds(r.histogram("router_forward_seconds", "Time forwarding to a backend.",
                                    obs::latencyBuckets())) {
    for (const std::string& backend : c.backends) {
      ring.add(backend);
      pools.emplace(backend, std::make_unique<ClientPool>(backend));
    }
    healthy_backends.set(static_cast<std::int64_t>(c.backends.size()));
  }

  // Called last in the router's constructor, because the monitor's thread can call back at once.
  void startMonitor() {
    // Building a client compiles a regex, and libstdc++ fills a shared locale cache the first time
    // without a lock. Doing it once here means later clients on other threads only read that cache.
    const httplib::Client prime("http://127.0.0.1:1");
    monitor = std::make_unique<HealthMonitor>(
        config.backends, probeHealthz,
        [this](const std::string& node, bool healthy) { onHealthChange(node, healthy); },
        config.health);
  }

  void onHealthChange(const std::string& node, bool healthy) {
    const std::unique_lock lock(ring_mutex);
    if (healthy) {
      ring.add(node);
    } else {
      ring.remove(node);
    }
    healthy_backends.set(static_cast<std::int64_t>(ring.nodeCount()));
    if (config.logger != nullptr) {
      const LogFields fields = {{"backend", node},
                                {"healthy_backends", static_cast<std::int64_t>(ring.nodeCount())}};
      if (healthy) {
        config.logger->info("backend back in rotation", fields);
      } else {
        config.logger->warn("backend removed from rotation", fields);
      }
    }
  }

  std::vector<std::string> candidates(std::string_view key) const {
    const std::shared_lock lock(ring_mutex);
    return ring.nodesFor(key, config.max_attempts);
  }

  void handleInfer(const httplib::Request& req, httplib::Response& res) {
    requests_total.add();
    std::string key = req.get_header_value("X-Routing-Key");
    if (key.empty()) {
      key = req.get_header_value("X-Client-Id");
    }
    httplib::Headers forwarded;
    for (const char* name : {"X-Client-Id", "X-Deadline-Ms"}) {
      if (req.has_header(name)) {
        forwarded.emplace(name, req.get_header_value(name));
      }
    }
    const std::string content_type = req.get_header_value("Content-Type", kJson);
    const auto wall_started = std::chrono::system_clock::now();
    const auto request_started = std::chrono::steady_clock::now();
    std::optional<obs::TraceContext> incoming;
    obs::TraceContext trace;
    if (config.tracer != nullptr) {
      incoming = obs::parseTraceparent(req.get_header_value("traceparent"));
      trace = config.tracer->startOrContinue(incoming);
      if (trace.valid()) {
        forwarded.emplace("traceparent", obs::formatTraceparent(trace));
      }
    } else if (req.has_header("traceparent")) {
      // Without a tracer here, the trace still reaches the backend intact.
      forwarded.emplace("traceparent", req.get_header_value("traceparent"));
    }
    const auto finishTrace = [&] {
      if (trace.sampled) {
        config.tracer->record({.trace_id = trace.trace_id,
                               .span_id = trace.span_id,
                               .parent_span_id = incoming ? incoming->span_id : "",
                               .name = "router.request",
                               .start = wall_started,
                               .duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - request_started),
                               .attributes = {{"status", std::to_string(res.status)},
                                              {"backend", res.get_header_value("X-Served-By")}}});
      }
    };

    bool retried = false;
    for (const std::string& backend : candidates(key)) {
      if (retried) {
        failovers_total.add();
      }
      retried = true;
      ClientPool& pool = *pools.at(backend);
      auto client = pool.acquire();
      const auto started = std::chrono::steady_clock::now();
      auto response = client->Post(req.path, forwarded, req.body, content_type);
      forward_seconds.observe(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
      if (!response) {
        if (config.logger != nullptr) {
          config.logger->warn("forward failed", {{"backend", backend},
                                                 {"error", httplib::to_string(response.error())}});
        }
        monitor->reportFailure(backend);
        pool.clear();
        continue;
      }
      pool.release(std::move(client));
      res.status = response->status;
      res.set_header("X-Served-By", backend);
      if (response->has_header("Retry-After")) {
        res.set_header("Retry-After", response->get_header_value("Retry-After"));
      }
      if (response->has_header("X-Model-Version")) {
        res.set_header("X-Model-Version", response->get_header_value("X-Model-Version"));
      }
      res.set_content(response->body, response->get_header_value("Content-Type", kJson));
      finishTrace();
      return;
    }
    unavailable_total.add();
    res.status = 503;
    res.set_content(nlohmann::json{{"error", "no backend available"}}.dump(), kJson);
    finishTrace();
  }

  RouterConfig config;
  obs::Registry& registry;
  mutable std::shared_mutex ring_mutex;
  HashRing ring;
  std::map<std::string, std::unique_ptr<ClientPool>> pools;
  obs::Counter& requests_total;
  obs::Counter& failovers_total;
  obs::Counter& unavailable_total;
  obs::Gauge& healthy_backends;
  obs::Histogram& forward_seconds;
  httplib::Server http;
  // Last member, so it stops before anything its callback touches is destroyed.
  std::unique_ptr<HealthMonitor> monitor;
};

Router::Router(const RouterConfig& config, obs::Registry& registry)
    : impl_(std::make_unique<Impl>(config, registry)) {
  const std::size_t threads = config.connection_threads;
  impl_->http.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };
  impl_->http.set_tcp_nodelay(true);
  impl_->http.set_keep_alive_max_count(1'000'000);
  const auto forward = [this](const httplib::Request& req, httplib::Response& res) {
    impl_->handleInfer(req, res);
  };
  impl_->http.Post("/v1/infer", forward);
  impl_->http.Post("/v1/models/:name/infer", forward);
  impl_->http.Get("/healthz", [this](const httplib::Request&, httplib::Response& res) {
    if (impl_->healthy_backends.value() > 0) {
      res.set_content("ok\n", "text/plain");
    } else {
      res.status = 503;
      res.set_content("no healthy backends\n", "text/plain");
    }
  });
  impl_->http.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(impl_->registry.renderPrometheus(), "text/plain; version=0.0.4");
  });
  impl_->http.Get("/v1/cluster", [this](const httplib::Request&, httplib::Response& res) {
    nlohmann::json backends = nlohmann::json::array();
    for (const std::string& backend : impl_->config.backends) {
      backends.push_back({{"backend", backend}, {"healthy", impl_->monitor->isHealthy(backend)}});
    }
    res.set_content(nlohmann::json{{"backends", backends}}.dump(), kJson);
  });
  impl_->startMonitor();
}

Router::~Router() = default;

int Router::bind(const std::string& host, int port) {
  if (port == 0) {
    return impl_->http.bind_to_any_port(host);
  }
  return impl_->http.bind_to_port(host, port) ? port : -1;
}

void Router::serve() { impl_->http.listen_after_bind(); }

void Router::stop() { impl_->http.stop(); }

}  // namespace cluster
