#include <pthread.h>

#include <csignal>
#include <cstdint>
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

#include "cluster/router.h"
#include "obs/log.h"
#include "obs/metrics.h"
#include "obs/trace.h"

int main(int argc, char** argv) {
  try {
    std::map<std::string, std::string> flags;
    const std::span args(argv, static_cast<std::size_t>(argc));
    for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
      flags[std::string(args[i]).substr(2)] = args[i + 1];
    }
    if (!flags.contains("backends")) {
      std::cerr << "usage: inference_router --backends host:port,host:port [--host H] [--port P]\n"
                   "  [--trace-file PATH] [--trace-sample N]\n";
      return EXIT_FAILURE;
    }

    // A client or backend that disconnects mid-write must not kill the process.
    std::signal(SIGPIPE, SIG_IGN);
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);

    cluster::RouterConfig config;
    std::stringstream backends(flags["backends"]);
    for (std::string backend; std::getline(backends, backend, ',');) {
      config.backends.push_back(backend);
    }

    obs::Logger logger(std::cerr, "router");
    config.logger = &logger;
    std::unique_ptr<obs::Tracer> tracer;
    if (flags.contains("trace-file")) {
      const std::uint64_t sample =
          flags.contains("trace-sample") ? std::stoull(flags["trace-sample"]) : 100;
      tracer = std::make_unique<obs::Tracer>(flags["trace-file"], "router", sample);
      config.tracer = tracer.get();
    }

    obs::Registry registry;
    cluster::Router router(config, registry);
    const std::string host = flags.contains("host") ? flags["host"] : "127.0.0.1";
    const int port = router.bind(host, flags.contains("port") ? std::stoi(flags["port"]) : 8000);
    if (port < 0) {
      std::cerr << "failed to bind " << host << '\n';
      return EXIT_FAILURE;
    }
    std::cout << "routing on " << host << ':' << port << '\n' << std::flush;
    logger.info("routing", {{"host", host},
                            {"port", std::int64_t{port}},
                            {"backends", static_cast<std::int64_t>(config.backends.size())}});

    std::thread serving([&router] { router.serve(); });
    int received = 0;
    sigwait(&signals, &received);
    router.stop();
    serving.join();
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
