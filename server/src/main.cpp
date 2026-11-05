#include <pthread.h>
#include <sys/resource.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cluster/model_registry.h"
#include "cluster/replicator.h"
#include "obs/log.h"
#include "obs/metrics.h"
#include "obs/startup.h"
#include "obs/trace.h"
#include "runtime/onnx_model.h"
#include "security/api_keys.h"
#include "server/batching_engine.h"
#include "server/event_frontend.h"
#include "server/http_server.h"
#include "server/model_manager.h"
#include "stream/processor.h"
#include "stream/sharded_log.h"

namespace {

// Parses "--key value" pairs.
std::map<std::string, std::string> parseFlags(std::span<char*> args) {
  std::map<std::string, std::string> flags;
  for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
    flags[std::string(args[i]).substr(2)] = args[i + 1];
  }
  return flags;
}

std::size_t sizeFlag(const std::map<std::string, std::string>& flags, const std::string& name,
                     std::size_t fallback) {
  const auto it = flags.find(name);
  return it == flags.end() ? fallback : std::stoul(it->second);
}

server::LoadedBackend loadOnnx(const std::filesystem::path& path,
                               runtime::OnnxModel::Provider provider) {
  auto model = std::make_shared<runtime::OnnxModel>(path, provider);
  server::LoadedBackend backend;
  backend.row_shape = model->inputRowShape();
  backend.run = [model = model.get()](std::span<const float> input,
                                      std::span<const std::int64_t> shape) {
    return model->run(input, shape);
  };
  backend.owner = std::move(model);
  return backend;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    obs::primeLocaleCaches();
    const auto flags = parseFlags(std::span(argv, static_cast<std::size_t>(argc)));
    if (!flags.contains("model")) {
      std::cerr
          << "usage: inference_server --model PATH [--host H] [--port P] [--workers N]\n"
             "  [--max-batch N] [--max-wait-us N] [--queue N] [--connections N]\n"
             "  [--tcp-nodelay 0|1] [--keep-alive-max N] [--events 0|1] [--window-ms N]\n"
             "  [--events-dir DIR] [--events-fsync always|interval|never] [--events-archive DIR]\n"
             "  [--api-keys FILE] [--tls-cert PEM --tls-key PEM] [--peer-key KEY]\n"
             "  [--provider cpu|coreml] [--frontend httplib|event] [--event-threads N] "
             "[--pin-workers 0|1] [--model-dir "
             "DIR] [--model-poll-ms N] [--arena 0|1] "
             "[--trace-file PATH] "
             "[--trace-sample N] [--node-id ID] [--peers "
             "host:port,...] "
             "[--replication-interval-ms N]\n";
      return EXIT_FAILURE;
    }

    // Every connection needs a descriptor, and the default limit can be as low as 256.
    rlimit limit{};
    if (getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur < limit.rlim_max) {
      limit.rlim_cur = limit.rlim_max == RLIM_INFINITY ? 1048576 : limit.rlim_max;
      (void)setrlimit(RLIMIT_NOFILE, &limit);
    }

    // A client that disconnects mid-response must fail that one write, not kill the process.
    std::signal(SIGPIPE, SIG_IGN);

    // Block SIGINT and SIGTERM before any thread starts, so only sigwait below receives them.
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);

    const std::string node_id = flags.contains("node-id") ? flags.at("node-id") : "standalone";
    obs::Logger logger(std::cerr, node_id);
    std::unique_ptr<obs::Tracer> tracer;
    if (flags.contains("trace-file")) {
      tracer = std::make_unique<obs::Tracer>(flags.at("trace-file"), node_id,
                                             sizeFlag(flags, "trace-sample", 100));
    }

    obs::Registry registry;
    const server::EngineConfig config{
        .workers = sizeFlag(flags, "workers", 4),
        .max_batch_size = sizeFlag(flags, "max-batch", 8),
        .max_wait = std::chrono::microseconds(sizeFlag(flags, "max-wait-us", 0)),
        .queue_capacity = sizeFlag(flags, "queue", 1024),
        .use_arena = sizeFlag(flags, "arena", 0) != 0,
        .tracer = tracer.get(),
        .pin_workers = sizeFlag(flags, "pin-workers", 0) != 0,
    };
    // Declared before the manager, whose thread reads it, so that it is destroyed after.
    cluster::ModelRegistry models(node_id);
    const auto provider = flags.contains("provider") && flags.at("provider") == "coreml"
                              ? runtime::OnnxModel::Provider::kCoreMl
                              : runtime::OnnxModel::Provider::kCpu;
    server::ModelManager manager(
        [provider](const std::filesystem::path& path) { return loadOnnx(path, provider); }, config,
        registry, &logger);
    manager.load("default", flags.at("model"), 0);

    server::HttpServer http(manager, registry, sizeFlag(flags, "connections", 256),
                            sizeFlag(flags, "tcp-nodelay", 1) != 0,
                            sizeFlag(flags, "keep-alive-max", 1000000),
                            flags.contains("tls-cert") ? flags.at("tls-cert") : "",
                            flags.contains("tls-key") ? flags.at("tls-key") : "");
    std::unique_ptr<security::ApiKeys> api_keys;
    if (flags.contains("api-keys")) {
      api_keys = std::make_unique<security::ApiKeys>(std::filesystem::path(flags.at("api-keys")));
      http.attachApiKeys(*api_keys);
      logger.info("API keys required", {{"keys", static_cast<std::int64_t>(api_keys->size())}});
    }

    // On by default. Requests are recorded on a drop-newest log, so a slow consumer loses
    // analytics events rather than delaying inference.
    std::unique_ptr<stream::ShardedLog> events;
    std::unique_ptr<stream::StreamProcessor> processor;
    if (sizeFlag(flags, "events", 1) != 0) {
      stream::LogConfig log_config{.policy = stream::BackpressurePolicy::kDropNewest};
      if (flags.contains("events-dir")) {
        // Sync on an interval by default: syncing per event would make requests wait on the disk.
        stream::WalConfig wal;
        wal.dir = flags.at("events-dir");
        const std::string fsync =
            flags.contains("events-fsync") ? flags.at("events-fsync") : "interval";
        wal.fsync = fsync == "always"  ? stream::FsyncPolicy::kEveryAppend
                    : fsync == "never" ? stream::FsyncPolicy::kNever
                                       : stream::FsyncPolicy::kInterval;
        if (flags.contains("events-archive")) {
          wal.archive = std::make_shared<stream::DirectorySegmentStore>(flags.at("events-archive"));
        }
        log_config.wal = wal;
      }
      events = std::make_unique<stream::ShardedLog>(log_config);
      processor = std::make_unique<stream::StreamProcessor>(
          *events,
          stream::ProcessorConfig{.window = {.window_ms = static_cast<std::int64_t>(
                                                 sizeFlag(flags, "window-ms", 1000))},
                                  .clock = [] {
                                    return std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                        .count();
                                  }});
      http.attachEventStream(*events, *processor);
    }

    http.attachLogger(logger);
    if (tracer) {
      http.attachTracer(*tracer);
    }
    http.attachModelRegistry(models);
    std::vector<std::string> peers;
    if (flags.contains("peers")) {
      std::stringstream list(flags.at("peers"));
      for (std::string peer; std::getline(list, peer, ',');) {
        peers.push_back(peer);
      }
    }
    // Metadata can only name files in this directory, so a registry write cannot load anything
    // else.
    if (flags.contains("model-dir")) {
      manager.followRegistry(models, flags.at("model-dir"),
                             std::chrono::milliseconds(sizeFlag(flags, "model-poll-ms", 200)));
    }
    std::unique_ptr<cluster::Replicator> replicator;
    if (!peers.empty()) {
      replicator = std::make_unique<cluster::Replicator>(
          models, peers,
          cluster::Replicator::httpFetch(flags.contains("peer-key") ? flags.at("peer-key") : ""),
          std::chrono::milliseconds(sizeFlag(flags, "replication-interval-ms", 1000)));
    }

    const std::string host = flags.contains("host") ? flags.at("host") : "127.0.0.1";
    // The event frontend serves requests and metrics only, with no tracing, event recording, or
    // model management endpoints, so those flags have no effect with it.
    std::unique_ptr<server::EventFrontend> frontend;
    if (flags.contains("frontend") && flags.at("frontend") == "event") {
      if (flags.contains("tls-cert")) {
        // Refusing is better than quietly serving plain text to someone who asked for TLS.
        throw std::runtime_error("the event frontend does not support TLS");
      }
      frontend = std::make_unique<server::EventFrontend>(manager, registry,
                                                         sizeFlag(flags, "event-threads", 2));
      if (api_keys) {
        frontend->attachApiKeys(*api_keys);
      }
    }
    const int requested_port = static_cast<int>(sizeFlag(flags, "port", 8080));
    const int port =
        frontend ? frontend->bind(host, requested_port) : http.bind(host, requested_port);
    if (port < 0) {
      logger.error("failed to bind", {{"host", host}});
      return EXIT_FAILURE;
    }
    std::cout << "listening on " << host << ':' << port << '\n' << std::flush;

    logger.info("serving", {{"host", host},
                            {"port", std::int64_t{port}},
                            {"workers", static_cast<std::int64_t>(config.workers)},
                            {"max_batch", static_cast<std::int64_t>(config.max_batch_size)}});

    std::thread serving([&] {
      if (frontend) {
        frontend->serve();
      } else {
        http.serve();
      }
    });
    int received = 0;
    sigwait(&signals, &received);
    logger.info("shutting down", {{"signal", std::int64_t{received}}});
    if (frontend) {
      frontend->stop();
    } else {
      http.stop();
    }
    serving.join();
    logger.info("stopped");
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
