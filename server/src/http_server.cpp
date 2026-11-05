#include "server/http_server.h"

#include <httplib.h>

#include <chrono>
#include <exception>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace server {
namespace {

constexpr const char* kJson = "application/json";

void sendError(httplib::Response& res, int status, const std::string& message) {
  res.status = status;
  res.set_content(nlohmann::json{{"error", message}}.dump(), kJson);
}

}  // namespace

struct HttpServer::Impl {
  Impl(BatchingEngine& e, obs::Registry& r)
      : engine(e),
        registry(r),
        request_seconds(r.histogram("http_request_duration_seconds",
                                    "Time spent handling POST /v1/infer.", obs::latencyBuckets())) {
  }

  void handleInfer(const httplib::Request& req, httplib::Response& res) {
    const auto started = std::chrono::steady_clock::now();
    const auto wall_started = std::chrono::system_clock::now();
    std::optional<obs::TraceContext> incoming;
    obs::TraceContext trace;
    if (tracer != nullptr) {
      incoming = obs::parseTraceparent(req.get_header_value("traceparent"));
      trace = tracer->startOrContinue(incoming);
    }
    std::vector<float> features;
    try {
      features = nlohmann::json::parse(req.body).at("inputs").get<std::vector<float>>();
    } catch (const nlohmann::json::exception& e) {
      sendError(res, 400, e.what());
      return;
    }
    try {
      auto result = engine.submit(std::move(features), trace);
      if (!result) {
        res.set_header("Retry-After", "1");
        sendError(res, 503, "queue full");
        if (logger != nullptr && reject_limiter.allow()) {
          logger->warn("shedding load: queue full");
        }
        return;
      }
      res.set_content(nlohmann::json{{"outputs", result->get()}}.dump(), kJson);
      res.status = 200;  // The library only fills this in after the handler, too late for the span.
      recordEvent(req, started);
    } catch (const std::invalid_argument& e) {
      sendError(res, 400, e.what());
    } catch (const std::exception& e) {
      sendError(res, 500, e.what());
      if (logger != nullptr && error_limiter.allow()) {
        logger->error("inference failed", {{"error", std::string(e.what())}});
      }
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    request_seconds.observe(std::chrono::duration<double>(elapsed).count());
    if (trace.sampled) {
      tracer->record({.trace_id = trace.trace_id,
                      .span_id = trace.span_id,
                      .parent_span_id = incoming ? incoming->span_id : "",
                      .name = "server.request",
                      .start = wall_started,
                      .duration = std::chrono::duration_cast<std::chrono::microseconds>(elapsed),
                      .attributes = {{"status", std::to_string(res.status)}}});
    }
  }

  void recordEvent(const httplib::Request& req, std::chrono::steady_clock::time_point started) {
    if (events == nullptr) {
      return;
    }
    const std::string client = req.get_header_value("X-Client-Id");
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    (void)events->appendNoWait(
        {.key = client.empty() ? "anonymous" : client,
         .value =
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                 .count(),
         .timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count()});
  }

  void handleStats(httplib::Response& res) const {
    nlohmann::json windows = nlohmann::json::array();
    if (processor != nullptr) {
      for (const stream::WindowResult& w : processor->closedWindows()) {
        windows.push_back({{"key", w.key},
                           {"start_ms", w.window_start_ms},
                           {"end_ms", w.window_end_ms},
                           {"count", w.count},
                           {"mean_latency_ms", w.sum / static_cast<double>(w.count)},
                           {"min_latency_ms", w.min},
                           {"max_latency_ms", w.max}});
      }
    }
    res.set_content(nlohmann::json{{"windows", windows}}.dump(), kJson);
  }

  obs::Logger* logger = nullptr;
  obs::Tracer* tracer = nullptr;
  obs::LogLimiter reject_limiter{1000};
  obs::LogLimiter error_limiter{1000};
  cluster::ModelRegistry* models = nullptr;
  stream::ShardedLog* events = nullptr;
  const stream::StreamProcessor* processor = nullptr;
  BatchingEngine& engine;
  obs::Registry& registry;
  obs::Histogram& request_seconds;
  httplib::Server http;
};

HttpServer::HttpServer(BatchingEngine& engine, obs::Registry& registry,
                       std::size_t connection_threads, bool tcp_nodelay,
                       std::size_t keep_alive_max_requests)
    : impl_(std::make_unique<Impl>(engine, registry)) {
  impl_->http.set_tcp_nodelay(tcp_nodelay);
  impl_->http.set_keep_alive_max_count(keep_alive_max_requests);
  impl_->http.new_task_queue = [connection_threads] {
    return new httplib::ThreadPool(connection_threads);
  };
  impl_->http.Post("/v1/infer", [this](const httplib::Request& req, httplib::Response& res) {
    impl_->handleInfer(req, res);
  });
  impl_->http.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("ok\n", "text/plain");
  });
  impl_->http.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
    std::string text = impl_->registry.renderPrometheus();
    if (impl_->events != nullptr) {
      text += impl_->events->renderPrometheus();
    }
    res.set_content(text, "text/plain; version=0.0.4");
  });
  impl_->http.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
    if (impl_->models == nullptr) {
      sendError(res, 404, "no model registry");
      return;
    }
    res.set_content(impl_->models->toJson(), kJson);
  });
  impl_->http.Put("/v1/models/:name", [this](const httplib::Request& req, httplib::Response& res) {
    if (impl_->models == nullptr) {
      sendError(res, 404, "no model registry");
      return;
    }
    try {
      const auto uri = nlohmann::json::parse(req.body).at("uri").get<std::string>();
      const cluster::ModelMeta meta = impl_->models->put(req.path_params.at("name"), uri);
      res.set_content(nlohmann::json{{"name", meta.name}, {"version", meta.version}}.dump(), kJson);
    } catch (const nlohmann::json::exception& e) {
      sendError(res, 400, e.what());
    }
  });
  impl_->http.Get("/v1/stats", [this](const httplib::Request&, httplib::Response& res) {
    impl_->handleStats(res);
  });
}

HttpServer::~HttpServer() = default;

void HttpServer::attachEventStream(stream::ShardedLog& log,
                                   const stream::StreamProcessor& processor) {
  impl_->events = &log;
  impl_->processor = &processor;
}

void HttpServer::attachLogger(obs::Logger& logger) { impl_->logger = &logger; }

void HttpServer::attachTracer(obs::Tracer& tracer) { impl_->tracer = &tracer; }

void HttpServer::attachModelRegistry(cluster::ModelRegistry& registry) {
  impl_->models = &registry;
}

int HttpServer::bind(const std::string& host, int port) {
  if (port == 0) {
    return impl_->http.bind_to_any_port(host);
  }
  return impl_->http.bind_to_port(host, port) ? port : -1;
}

void HttpServer::serve() { impl_->http.listen_after_bind(); }

void HttpServer::stop() { impl_->http.stop(); }

}  // namespace server
