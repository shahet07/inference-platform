// Streams generated text over HTTP.
//   POST /v1/generate  {"prompt":"...","max_tokens":64,"stream":true}
//     With stream, the response is server-sent events: one {"token":"..."} per token, then
//     {"done":true,"finish_reason":"..."}. Without it, one JSON object once finished.
//     A client that disconnects while streaming frees its slot at the next step.
//   GET  /metrics, GET /healthz
#include <httplib.h>
#include <pthread.h>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "llm/fake_backend.h"
#include "llm/scheduler.h"
#include "obs/metrics.h"
#include "obs/startup.h"
#ifdef IP_HAS_LLAMA
#include "llm/llama_backend.h"
#endif

namespace {

using nlohmann::json;

const char* reasonName(llm::FinishReason reason) {
  switch (reason) {
    case llm::FinishReason::kEnd:
      return "end";
    case llm::FinishReason::kMaxTokens:
      return "max_tokens";
    case llm::FinishReason::kCancelled:
      return "cancelled";
    case llm::FinishReason::kError:
      return "error";
  }
  return "unknown";
}

// Passes tokens from the scheduler's thread to the one writing this response.
struct Stream {
  std::mutex mutex;
  std::condition_variable has_data;
  std::deque<std::string> tokens;
  bool finished = false;
  llm::FinishReason reason = llm::FinishReason::kError;
  std::shared_ptr<llm::GenerateHandle> handle = std::make_shared<llm::GenerateHandle>();
};

// The model produces bytes, and one token can end in the middle of a multi-byte character, which
// JSON cannot represent, so invalid bytes are replaced.
std::string tokenJson(const std::string& token) {
  return json{{"token", token}}.dump(-1, ' ', false, json::error_handler_t::replace);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    obs::primeLocaleCaches();
    std::map<std::string, std::string> flags;
    const std::span args(argv, static_cast<std::size_t>(argc));
    for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
      flags[std::string(args[i]).substr(2)] = args[i + 1];
    }
    if (!flags.contains("model") && !flags.contains("fake-step-us")) {
      std::cerr << "usage: llm_server (--model GGUF | --fake-step-us N) [--host H] [--port P]\n"
                   "  [--slots N] [--context N] [--threads N] [--policy continuous|static]\n";
      return EXIT_FAILURE;
    }
    std::signal(SIGPIPE, SIG_IGN);
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);

    const auto number = [&flags](const std::string& name, std::int64_t fallback) {
      return flags.contains(name) ? std::stoll(flags[name]) : fallback;
    };
    const auto slots = static_cast<std::int32_t>(number("slots", 8));
    const auto context = static_cast<std::int32_t>(number("context", 256));

    std::unique_ptr<llm::Backend> backend;
    if (flags.contains("fake-step-us")) {
      backend =
          std::make_unique<llm::FakeBackend>(std::chrono::microseconds(number("fake-step-us", 0)));
    } else {
#ifdef IP_HAS_LLAMA
      backend = std::make_unique<llm::LlamaBackend>(
          flags["model"], slots, context, static_cast<std::int32_t>(number("threads", 4)));
#else
      throw std::runtime_error("this build has no llama.cpp (IP_ENABLE_LLAMA)");
#endif
    }

    obs::Registry registry;
    llm::Scheduler scheduler(*backend,
                             {.max_slots = static_cast<std::size_t>(slots),
                              .queue_capacity = static_cast<std::size_t>(number("queue", 256)),
                              .max_sequence_tokens = context,
                              .policy = flags.contains("policy") && flags["policy"] == "static"
                                            ? llm::BatchPolicy::kStatic
                                            : llm::BatchPolicy::kContinuous},
                             registry);

    httplib::Server http;
    http.set_tcp_nodelay(true);
    http.new_task_queue = [] { return new httplib::ThreadPool(512); };
    http.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
      res.set_content("ok\n", "text/plain");
    });
    http.Get("/metrics", [&registry](const httplib::Request&, httplib::Response& res) {
      res.set_content(registry.renderPrometheus(), "text/plain; version=0.0.4");
    });
    http.Post("/v1/generate", [&scheduler](const httplib::Request& req, httplib::Response& res) {
      std::string prompt;
      std::int32_t max_tokens = 64;
      bool wants_stream = false;
      try {
        const json body = json::parse(req.body);
        prompt = body.at("prompt").get<std::string>();
        max_tokens = body.value("max_tokens", 64);
        wants_stream = body.value("stream", false);
      } catch (const json::exception& e) {
        res.status = 400;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        return;
      }
      if (max_tokens <= 0) {
        res.status = 400;
        res.set_content(R"({"error":"max_tokens must be positive"})", "application/json");
        return;
      }
      auto stream = std::make_shared<Stream>();
      const bool accepted = scheduler.submit({.prompt = std::move(prompt),
                                              .max_tokens = max_tokens,
                                              .on_token =
                                                  [stream](std::string_view text) {
                                                    {
                                                      const std::scoped_lock lock(stream->mutex);
                                                      stream->tokens.emplace_back(text);
                                                    }
                                                    stream->has_data.notify_one();
                                                  },
                                              .on_finish =
                                                  [stream](llm::FinishReason reason) {
                                                    {
                                                      const std::scoped_lock lock(stream->mutex);
                                                      stream->finished = true;
                                                      stream->reason = reason;
                                                    }
                                                    stream->has_data.notify_one();
                                                  },
                                              .handle = stream->handle});
      if (!accepted) {
        res.status = 503;
        res.set_header("Retry-After", "1");
        res.set_content(R"({"error":"queue full"})", "application/json");
        return;
      }
      if (!wants_stream) {
        std::string text;
        std::unique_lock lock(stream->mutex);
        stream->has_data.wait(lock, [&] { return stream->finished; });
        for (const std::string& token : stream->tokens) {
          text += token;
        }
        res.status = stream->reason == llm::FinishReason::kError ? 400 : 200;
        res.set_content(json{{"text", text}, {"finish_reason", reasonName(stream->reason)}}.dump(
                            -1, ' ', false, json::error_handler_t::replace),
                        "application/json");
        return;
      }
      res.set_header("Cache-Control", "no-cache");
      res.set_chunked_content_provider(
          "text/event-stream",
          [stream](std::size_t, httplib::DataSink& sink) {
            std::unique_lock lock(stream->mutex);
            stream->has_data.wait(lock,
                                  [&] { return stream->finished || !stream->tokens.empty(); });
            while (!stream->tokens.empty()) {
              const std::string line = "data: " + tokenJson(stream->tokens.front()) + "\n\n";
              stream->tokens.pop_front();
              lock.unlock();
              if (!sink.write(line.data(), line.size())) {
                return false;  // The client went away. The release callback below cancels.
              }
              lock.lock();
            }
            if (stream->finished) {
              const std::string line =
                  "data: " +
                  json{{"done", true}, {"finish_reason", reasonName(stream->reason)}}.dump() +
                  "\n\n";
              lock.unlock();
              sink.write(line.data(), line.size());
              sink.done();
            }
            return true;
          },
          // Runs however the response ended. Without this, a client that vanished would keep its
          // slot generating tokens nobody will read.
          [stream](bool) { stream->handle->cancelled.store(true); });
    });

    const std::string host = flags.contains("host") ? flags["host"] : "127.0.0.1";
    const int port = static_cast<int>(number("port", 8090));
    if (!http.bind_to_port(host, port)) {
      std::cerr << "failed to bind " << host << ':' << port << '\n';
      return EXIT_FAILURE;
    }
    std::thread serving([&http] { http.listen_after_bind(); });
    int received = 0;
    sigwait(&signals, &received);
    http.stop();
    serving.join();
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
