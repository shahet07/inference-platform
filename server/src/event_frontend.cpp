#include "server/event_frontend.h"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "net/event_server.h"

namespace server {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::string_view kJson = "application/json";
constexpr std::string_view kBinary = "application/octet-stream";
constexpr std::string_view kModelPrefix = "/v1/models/";
constexpr std::string_view kInferSuffix = "/infer";

void sendError(const net::Responder& responder, int status, const std::string& message) {
  responder.respond(status, kJson, nlohmann::json{{"error", message}}.dump());
}

std::vector<float> decodeFloats(std::string_view body) {
  if (body.size() % sizeof(float) != 0) {
    throw std::invalid_argument("binary body must be a whole number of 32-bit floats");
  }
  std::vector<float> values(body.size() / sizeof(float));
  std::memcpy(values.data(), body.data(), body.size());
  return values;
}

std::int64_t parseDeadlineMs(std::string_view text) {
  std::int64_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  constexpr std::int64_t kMaxMs = std::int64_t{1000} * 60 * 60 * 24;
  if (error != std::errc{} || end != text.data() + text.size() || value < 0 || value > kMaxMs) {
    throw std::invalid_argument(
        "X-Deadline-Ms must be a whole number of milliseconds, up to a day");
  }
  return value;
}

// The model named in the path, or nullopt if the path is not an inference path. Query strings are
// not used by this API, so one simply does not match.
std::optional<std::string> modelForPath(std::string_view path) {
  if (path == "/v1/infer") {
    return "default";
  }
  if (path.starts_with(kModelPrefix) && path.ends_with(kInferSuffix) &&
      path.size() > kModelPrefix.size() + kInferSuffix.size()) {
    const std::string_view name =
        path.substr(kModelPrefix.size(), path.size() - kModelPrefix.size() - kInferSuffix.size());
    if (name.find('/') == std::string_view::npos) {
      return std::string(name);
    }
  }
  return std::nullopt;
}

}  // namespace

struct EventFrontend::Impl {
  Impl(ModelManager& m, obs::Registry& r, std::size_t threads)
      : manager(m),
        registry(r),
        request_seconds(r.histogram("http_request_duration_seconds",
                                    "Time spent handling POST /v1/infer.", obs::latencyBuckets())),
        server({.threads = threads},
               [this](const net::HttpRequest& request, const net::Responder& responder) {
                 handle(request, responder);
               }) {}

  // Returns false if a response has been sent.
  [[nodiscard]] bool authorize(const net::HttpRequest& request,
                               const net::Responder& responder) const {
    if (api_keys == nullptr || request.path == "/healthz") {
      return true;
    }
    switch (api_keys->check(request.header("Authorization"), security::Role::kClient)) {
      case security::AuthResult::kAllowed:
        return true;
      case security::AuthResult::kUnauthenticated:
        responder.respond(401, kJson, R"({"error":"a valid API key is required"})",
                          {{"WWW-Authenticate", "Bearer"}});
        return false;
      case security::AuthResult::kForbidden:
        sendError(responder, 403, "this key cannot do that");
        return false;
      case security::AuthResult::kRateLimited:
        responder.respond(429, kJson, R"({"error":"rate limit exceeded"})", {{"Retry-After", "1"}});
        return false;
    }
    return false;
  }

  void handle(const net::HttpRequest& request, const net::Responder& responder) {
    if (!authorize(request, responder)) {
      return;
    }
    if (request.method == "GET") {
      if (request.path == "/healthz") {
        responder.respond(200, "text/plain", "ok\n");
      } else if (request.path == "/metrics") {
        responder.respond(200, "text/plain; version=0.0.4", registry.renderPrometheus());
      } else {
        sendError(responder, 404, "not found");
      }
      return;
    }
    const std::optional<std::string> name = modelForPath(request.path);
    if (request.method != "POST" || !name) {
      sendError(responder, 404, "not found");
      return;
    }
    infer(request, responder, *name);
  }

  void infer(const net::HttpRequest& request, const net::Responder& responder,
             const std::string& name) {
    const auto started = Clock::now();
    // Held by the completion below until the response is on its way, so a hot swap cannot remove
    // it.
    std::shared_ptr<ServingModel> model = manager.get(name);
    if (model == nullptr) {
      sendError(responder, 404, "no such model: " + name);
      return;
    }
    const bool binary = request.header("Content-Type") == kBinary;
    std::vector<float> features;
    std::optional<Clock::time_point> deadline;
    try {
      const std::string_view deadline_text = request.header("X-Deadline-Ms");
      if (!deadline_text.empty()) {
        deadline = started + std::chrono::milliseconds(parseDeadlineMs(deadline_text));
      }
      features = binary
                     ? decodeFloats(request.body)
                     : nlohmann::json::parse(request.body).at("inputs").get<std::vector<float>>();
    } catch (const nlohmann::json::exception& e) {
      sendError(responder, 400, e.what());
      return;
    } catch (const std::invalid_argument& e) {
      sendError(responder, 400, e.what());
      return;
    }

    BatchingEngine& engine = model->engine();
    auto done = [this, responder, binary, started, model = std::move(model)](
                    const std::vector<float>& outputs, const std::exception_ptr& error) {
      finish(responder, binary, started, *model, outputs, error);
    };
    try {
      if (!engine.submitAsync(std::move(features), std::move(done), deadline)) {
        responder.respond(503, kJson, R"({"error":"queue full"})", {{"Retry-After", "1"}});
      }
    } catch (const std::invalid_argument& e) {
      sendError(responder, 400, e.what());
    }
  }

  void finish(const net::Responder& responder, bool binary, Clock::time_point started,
              const ServingModel& model, const std::vector<float>& outputs,
              const std::exception_ptr& error) {
    if (error) {
      try {
        std::rethrow_exception(error);
      } catch (const DeadlineExceeded& e) {
        sendError(responder, 504, e.what());
      } catch (const std::exception& e) {
        sendError(responder, 500, e.what());
      }
    } else {
      const std::vector<std::pair<std::string, std::string>> headers{
          {"X-Model-Version", std::to_string(model.version())}};
      if (binary) {
        responder.respond(200, kBinary,
                          std::string_view(reinterpret_cast<const char*>(outputs.data()),
                                           outputs.size() * sizeof(float)),
                          headers);
      } else {
        responder.respond(200, kJson, nlohmann::json{{"outputs", outputs}}.dump(), headers);
      }
    }
    request_seconds.observe(std::chrono::duration<double>(Clock::now() - started).count());
  }

  security::ApiKeys* api_keys = nullptr;
  ModelManager& manager;
  obs::Registry& registry;
  obs::Histogram& request_seconds;
  net::EventServer server;
};

EventFrontend::EventFrontend(ModelManager& models, obs::Registry& registry, std::size_t threads)
    : impl_(std::make_unique<Impl>(models, registry, threads)) {}

EventFrontend::~EventFrontend() = default;

void EventFrontend::attachApiKeys(security::ApiKeys& keys) { impl_->api_keys = &keys; }

int EventFrontend::bind(const std::string& host, int port) {
  try {
    return impl_->server.bind(host, port);
  } catch (const std::exception&) {
    return -1;
  }
}

void EventFrontend::serve() { impl_->server.serve(); }

void EventFrontend::stop() { impl_->server.stop(); }

}  // namespace server
