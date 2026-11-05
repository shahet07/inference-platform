#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "obs/metrics.h"
#include "security/api_keys.h"
#include "server/model_manager.h"

namespace server {

// Serves inference on the event-driven server, where a few threads handle many connections. It
// offers:
//   POST /v1/infer, POST /v1/models/{name}/infer   JSON or raw floats, and X-Deadline-Ms
//   GET  /healthz, GET /metrics
// It does not have HttpServer's tracing, event recording, or model management endpoints.
class EventFrontend {
 public:
  EventFrontend(ModelManager& models, obs::Registry& registry, std::size_t threads);
  ~EventFrontend();

  EventFrontend(const EventFrontend&) = delete;
  EventFrontend& operator=(const EventFrontend&) = delete;
  EventFrontend(EventFrontend&&) = delete;
  EventFrontend& operator=(EventFrontend&&) = delete;

  // Requires an API key on everything except /healthz. `keys` must outlive this. Call before
  // serve().
  void attachApiKeys(security::ApiKeys& keys);

  // Binds to `port`, or any free port if it is 0. Returns the bound port, or -1 on failure.
  int bind(const std::string& host, int port);
  void serve();
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace server
