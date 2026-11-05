#include <pthread.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "cluster/model_registry.h"
#include "cluster/replicator.h"
#include "obs/log.h"
#include "obs/metrics.h"
#include "obs/trace.h"
#include "runtime/onnx_model.h"
#include "server/batching_engine.h"
#include "server/http_server.h"
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto flags = parseFlags(std::span(argv, static_cast<std::size_t>(argc)));
    if (!flags.contains("model")) {
      std::cerr << "usage: inference_server --model PATH [--host H] [--port P] [--workers N]\n"
                   "  [--max-batch N] [--max-wait-us N] [--queue N] [--connections N]\n"
                   "  [--tcp-nodelay 0|1] [--keep-alive-max N] [--events 0|1] [--window-ms N]\n"
                   "  [--arena 0|1] [--trace-file PATH] [--trace-sample N] [--node-id ID] [--peers "
                   "host:port,...] "
                   "[--replication-interval-ms N]\n";
      return EXIT_FAILURE;
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

    const runtime::OnnxModel model(flags.at("model"));
    obs::Registry registry;
    const server::EngineConfig config{
        .workers = sizeFlag(flags, "workers", 4),
        .max_batch_size = sizeFlag(flags, "max-batch", 8),
        .max_wait = std::chrono::microseconds(sizeFlag(flags, "max-wait-us", 0)),
        .queue_capacity = sizeFlag(flags, "queue", 1024),
        .use_arena = sizeFlag(flags, "arena", 0) != 0,
        .tracer = tracer.get(),
    };
    server::BatchingEngine engine(
        [&model](std::span<const float> input, std::span<const std::int64_t> shape) {
          return model.run(input, shape);
        },
        static_cast<std::size_t>(model.inputFeatureCount()), config, registry);

    server::HttpServer http(engine, registry, sizeFlag(flags, "connections", 256),
                            sizeFlag(flags, "tcp-nodelay", 1) != 0,
                            sizeFlag(flags, "keep-alive-max", 1000000));

    // On by default. Requests are recorded on a drop-newest log, so a slow consumer loses
    // analytics events rather than delaying inference.
    std::unique_ptr<stream::ShardedLog> events;
    std::unique_ptr<stream::StreamProcessor> processor;
    if (sizeFlag(flags, "events", 1) != 0) {
      events = std::make_unique<stream::ShardedLog>(
          stream::LogConfig{.policy = stream::BackpressurePolicy::kDropNewest});
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

    cluster::ModelRegistry models(node_id);
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
    std::unique_ptr<cluster::Replicator> replicator;
    if (!peers.empty()) {
      replicator = std::make_unique<cluster::Replicator>(
          models, peers, cluster::Replicator::httpFetch(),
          std::chrono::milliseconds(sizeFlag(flags, "replication-interval-ms", 1000)));
    }

    const std::string host = flags.contains("host") ? flags.at("host") : "127.0.0.1";
    const int port = http.bind(host, static_cast<int>(sizeFlag(flags, "port", 8080)));
    if (port < 0) {
      logger.error("failed to bind", {{"host", host}});
      return EXIT_FAILURE;
    }
    std::cout << "listening on " << host << ':' << port << '\n' << std::flush;

    logger.info("serving", {{"host", host},
                            {"port", std::int64_t{port}},
                            {"workers", static_cast<std::int64_t>(config.workers)},
                            {"max_batch", static_cast<std::int64_t>(config.max_batch_size)}});

    std::thread serving([&http] { http.serve(); });
    int received = 0;
    sigwait(&signals, &received);
    logger.info("shutting down", {{"signal", std::int64_t{received}}});
    http.stop();
    serving.join();
    logger.info("stopped");
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
