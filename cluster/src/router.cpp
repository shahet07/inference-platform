#include "cluster/router.h"

#include <arpa/inet.h>
#include <httplib.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "cluster/hash_ring.h"
#include "obs/startup.h"

namespace cluster {
namespace {

constexpr const char* kJson = "application/json";
using LogFields = std::initializer_list<obs::LogField>;

// Reuses keep-alive connections. An httplib::Client serializes its own requests, so each
// in-flight request takes a client of its own.
class ClientPool {
 public:
  explicit ClientPool(std::string backend) : backend_(std::move(backend)) {}

  std::atomic<std::int64_t> in_flight{0};

  // `reused` says whether this client has an existing connection, which may have since died.
  std::unique_ptr<httplib::Client> acquire(bool& reused) {
    {
      const std::scoped_lock lock(mutex_);
      if (!idle_.empty()) {
        auto client = std::move(idle_.back());
        idle_.pop_back();
        reused = true;
        return client;
      }
    }
    reused = false;
    auto client = std::make_unique<httplib::Client>("http://" + backend_);
    client->set_keep_alive(true);
    client->set_tcp_nodelay(true);
    client->set_connection_timeout(std::chrono::milliseconds(200));
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

// Returns "address:port" for each IPv4 address of `name`. Empty on failure, which callers must not
// read as "no backends": a DNS outage should not empty a working cluster.
std::set<std::string> resolveBackends(const std::string& name, const std::string& port) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* results = nullptr;
  std::set<std::string> backends;
  if (getaddrinfo(name.c_str(), port.c_str(), &hints, &results) != 0) {
    return backends;
  }
  for (const addrinfo* entry = results; entry != nullptr; entry = entry->ai_next) {
    std::array<char, INET_ADDRSTRLEN> text{};
    const auto* address = reinterpret_cast<const sockaddr_in*>(entry->ai_addr);
    if (inet_ntop(AF_INET, &address->sin_addr, text.data(), text.size()) != nullptr) {
      backends.insert(std::string(text.data()) + ":" + port);
    }
  }
  freeaddrinfo(results);
  return backends;
}

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
        deadline_exceeded_total(r.counter("router_deadline_exceeded_total",
                                          "Requests that ran out of their own deadline.")),
        healthy_backends(r.gauge("router_healthy_backends", "Backends in rotation.")),
        panic_mode(
            r.gauge("router_panic_mode", "1 when every backend looks down, so all are tried.")),
        forward_seconds(r.histogram("router_forward_seconds", "Time forwarding to a backend.",
                                    obs::latencyBuckets())) {
    for (const std::string& backend : c.backends) {
      ring.add(backend);
      pools.emplace(backend, std::make_shared<ClientPool>(backend));
      healthy.push_back(backend);
    }
    healthy_backends.set(static_cast<std::int64_t>(c.backends.size()));
  }

  // Called last in the router's constructor, because the monitor's thread can call back at once.
  void startMonitor() {
    obs::primeLocaleCaches();  // The monitor's thread is about to use the HTTP client.
    monitor = std::make_unique<HealthMonitor>(
        config.backends, probeHealthz,
        [this](const std::string& node, bool now_healthy) { onHealthChange(node, now_healthy); },
        config.health);
  }

  void startDiscovery() {
    if (config.discover_dns.empty()) {
      return;
    }
    const auto colon = config.discover_dns.rfind(':');
    if (colon == std::string::npos) {
      throw std::invalid_argument("discover_dns must be name:port");
    }
    discovery = std::thread([this, name = config.discover_dns.substr(0, colon),
                             port = config.discover_dns.substr(colon + 1)] {
      std::unique_lock lock(discovery_mutex);
      while (!discovery_stopping) {
        lock.unlock();
        const std::set<std::string> found = resolveBackends(name, port);
        if (!found.empty()) {
          for (const std::string& backend : found) {
            addBackend(backend);
          }
          for (const std::string& backend : monitor->nodes()) {
            if (!found.contains(backend)) {
              removeBackend(backend);
            }
          }
        }
        lock.lock();
        discovery_stop.wait_for(lock, config.discover_interval,
                                [this] { return discovery_stopping; });
      }
    });
  }

  ~Impl() {
    {
      const std::scoped_lock lock(discovery_mutex);
      discovery_stopping = true;
    }
    discovery_stop.notify_all();
    if (discovery.joinable()) {
      discovery.join();
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  void onHealthChange(const std::string& node, bool now_healthy) {
    const std::unique_lock lock(ring_mutex);
    if (now_healthy) {
      down.erase(node);
    } else {
      down.insert(node);
      // Connections to a node that went down are not worth trying when it comes back.
      const auto pool = pools.find(node);
      if (pool != pools.end()) {
        pool->second->clear();
      }
    }
    rebuildRotation();
    if (config.logger != nullptr) {
      const LogFields fields = {
          {"backend", node},
          {"healthy_backends", static_cast<std::int64_t>(pools.size() - down.size())}};
      if (now_healthy) {
        config.logger->info("backend back in rotation", fields);
      } else {
        config.logger->warn("backend removed from rotation", fields);
      }
    }
  }

  // If every backend looks down, all of them stay in rotation. Health checks can be wrong, and an
  // overloaded cluster can fail its checks while still serving, so refusing everything would turn a
  // partial problem into a total outage. If they really are down, requests fail quickly anyway.
  void rebuildRotation() {
    ring = HashRing(config.virtual_nodes);
    healthy.clear();
    const bool panic = down.size() >= pools.size();
    for (const auto& [backend, pool] : pools) {
      if (panic || !down.contains(backend)) {
        ring.add(backend);
        healthy.push_back(backend);
      }
    }
    healthy_backends.set(static_cast<std::int64_t>(pools.size() - down.size()));
    panic_mode.set(panic && !pools.empty() ? 1 : 0);
  }

  std::vector<std::string> candidates(std::string_view key) const {
    const std::shared_lock lock(ring_mutex);
    if (config.balance == BalancePolicy::kConsistentHash && !key.empty()) {
      return ring.nodesFor(key, config.max_attempts);
    }
    return leastLoaded();
  }

  // Comparing two at random comes close to the best choice without coordinating, and stops every
  // router piling onto whichever backend looks least loaded at the same moment.
  std::vector<std::string> leastLoaded() const {
    if (healthy.empty()) {
      return {};
    }
    thread_local std::minstd_rand rng{std::random_device{}()};
    const std::size_t first = rng() % healthy.size();
    if (healthy.size() == 1) {
      return {healthy[first]};
    }
    std::size_t second = rng() % (healthy.size() - 1);
    second += static_cast<std::size_t>(second >= first);
    const auto load = [this](std::size_t i) {
      return pools.at(healthy[i])->in_flight.load(std::memory_order_relaxed);
    };
    return load(first) <= load(second) ? std::vector<std::string>{healthy[first], healthy[second]}
                                       : std::vector<std::string>{healthy[second], healthy[first]};
  }

  void addBackend(const std::string& backend) {
    {
      const std::unique_lock lock(ring_mutex);
      if (pools.contains(backend)) {
        return;
      }
      pools.emplace(backend, std::make_shared<ClientPool>(backend));
      rebuildRotation();
    }
    monitor->addNode(backend);
    if (config.logger != nullptr) {
      config.logger->info("backend added", {{"backend", backend}});
    }
  }

  // Requests on it keep their own reference to its connections, so they still finish.
  bool removeBackend(const std::string& backend) {
    {
      const std::unique_lock lock(ring_mutex);
      if (pools.erase(backend) == 0) {
        return false;
      }
      down.erase(backend);
      rebuildRotation();
    }
    monitor->removeNode(backend);
    if (config.logger != nullptr) {
      config.logger->info("backend removed", {{"backend", backend}});
    }
    return true;
  }

  std::shared_ptr<ClientPool> poolFor(const std::string& backend) const {
    const std::shared_lock lock(ring_mutex);
    const auto it = pools.find(backend);
    return it == pools.end() ? nullptr : it->second;
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
    bool has_deadline = false;
    std::chrono::steady_clock::time_point deadline{};
    if (req.has_header("X-Deadline-Ms")) {
      const std::string text = req.get_header_value("X-Deadline-Ms");
      std::int64_t ms = 0;
      const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), ms);
      if (error != std::errc{} || end != text.data() + text.size() || ms < 0 ||
          ms > std::int64_t{86'400'000}) {
        res.status = 400;
        res.set_content(nlohmann::json{{"error", "invalid X-Deadline-Ms"}}.dump(), kJson);
        return;
      }
      has_deadline = true;
      deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    }
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

    const auto respondDeadlineExceeded = [&] {
      deadline_exceeded_total.add();
      res.status = 504;
      res.set_content(nlohmann::json{{"error", "deadline exceeded"}}.dump(), kJson);
      finishTrace();
    };

    bool retried = false;
    for (const std::string& backend : candidates(key)) {
      if (retried) {
        failovers_total.add();
      }
      retried = true;
      const std::shared_ptr<ClientPool> pool_ptr = poolFor(backend);
      if (pool_ptr == nullptr) {
        continue;  // Removed since the candidates were chosen.
      }
      ClientPool& pool = *pool_ptr;
      bool reused = false;
      auto client = pool.acquire(reused);
      const auto started = std::chrono::steady_clock::now();
      // What is left of the request's own deadline can be less than the backend's allowance.
      const auto remaining =
          has_deadline ? std::chrono::duration_cast<std::chrono::milliseconds>(deadline - started)
                       : config.backend_timeout;
      const bool deadline_limits = has_deadline && remaining < config.backend_timeout;
      if (deadline_limits && remaining.count() <= 0) {
        respondDeadlineExceeded();
        return;
      }
      client->set_read_timeout(deadline_limits ? remaining : config.backend_timeout);
      forwarded.erase("X-Deadline-Ms");
      if (has_deadline) {
        forwarded.emplace("X-Deadline-Ms",
                          std::to_string(std::max<std::int64_t>(remaining.count(), 0)));
      }
      pool.in_flight.fetch_add(1, std::memory_order_relaxed);
      auto response = client->Post(req.path, forwarded, req.body, content_type);
      // A kept-alive connection can die while idle, for instance across a restart, and then fails
      // at once. That says nothing about the backend, so it gets one try on a new connection,
      // instead of the request moving to another node and this one being marked as failing.
      if (!response && reused &&
          std::chrono::steady_clock::now() - started < std::chrono::milliseconds(100)) {
        pool.clear();
        bool ignored = false;
        client = pool.acquire(ignored);
        client->set_read_timeout(deadline_limits ? remaining : config.backend_timeout);
        response = client->Post(req.path, forwarded, req.body, content_type);
      }
      pool.in_flight.fetch_sub(1, std::memory_order_relaxed);
      forward_seconds.observe(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
      if (!response) {
        pool.clear();
        // Running out of the client's own time says nothing about the backend's health.
        // The clock is not checked here: a read timeout set from the time remaining can fire just
        // before the deadline itself passes.
        if (deadline_limits && response.error() == httplib::Error::Read) {
          respondDeadlineExceeded();
          return;
        }
        if (config.logger != nullptr) {
          config.logger->warn("forward failed", {{"backend", backend},
                                                 {"error", httplib::to_string(response.error())}});
        }
        monitor->reportFailure(backend);
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
  std::map<std::string, std::shared_ptr<ClientPool>> pools;
  std::vector<std::string> healthy;  // The backends in rotation.
  std::set<std::string> down;        // Backends the health monitor considers unhealthy.
  obs::Counter& requests_total;
  obs::Counter& failovers_total;
  obs::Counter& unavailable_total;
  obs::Counter& deadline_exceeded_total;
  obs::Gauge& healthy_backends;
  obs::Gauge& panic_mode;
  obs::Histogram& forward_seconds;
  httplib::Server http;
  std::mutex discovery_mutex;
  std::condition_variable discovery_stop;
  bool discovery_stopping = false;
  std::thread discovery;
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
  impl_->http.Post("/v1/backends", [this](const httplib::Request& req, httplib::Response& res) {
    try {
      impl_->addBackend(nlohmann::json::parse(req.body).at("backend").get<std::string>());
      res.set_content("{}", kJson);
    } catch (const nlohmann::json::exception& e) {
      res.status = 400;
      res.set_content(nlohmann::json{{"error", e.what()}}.dump(), kJson);
    }
  });
  impl_->http.Delete("/v1/backends/:backend",
                     [this](const httplib::Request& req, httplib::Response& res) {
                       if (!impl_->removeBackend(req.path_params.at("backend"))) {
                         res.status = 404;
                       }
                     });
  impl_->http.Get("/v1/cluster", [this](const httplib::Request&, httplib::Response& res) {
    nlohmann::json backends = nlohmann::json::array();
    for (const std::string& backend : impl_->monitor->nodes()) {
      backends.push_back({{"backend", backend}, {"healthy", impl_->monitor->isHealthy(backend)}});
    }
    res.set_content(nlohmann::json{{"backends", backends}}.dump(), kJson);
  });
  impl_->startMonitor();
  impl_->startDiscovery();
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
