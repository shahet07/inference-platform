#include "cluster/replicator.h"

#include <httplib.h>

#include <exception>
#include <utility>

namespace cluster {

Replicator::Replicator(ModelRegistry& registry, std::vector<std::string> peers, Fetch fetch,
                       std::chrono::milliseconds interval)
    : registry_(registry), peers_(std::move(peers)), fetch_(std::move(fetch)), interval_(interval) {
  thread_ = std::thread([this] { run(); });
}

Replicator::~Replicator() {
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
  }
  stop_requested_.notify_all();
  thread_.join();
}

void Replicator::run() {
  std::unique_lock lock(mutex_);
  while (!stopping_) {
    lock.unlock();
    pullOnce();
    lock.lock();
    stop_requested_.wait_for(lock, interval_, [this] { return stopping_; });
  }
}

void Replicator::pullOnce() {
  for (const std::string& peer : peers_) {
    const std::optional<std::string> body = fetch_(peer);
    if (!body) {
      continue;  // The next round tries again.
    }
    try {
      registry_.mergeJson(*body);
    } catch (const std::exception&) {
      continue;  // One bad peer must not stop replication from the others.
    }
  }
}

Replicator::Fetch Replicator::httpFetch() {
  return [](const std::string& peer) -> std::optional<std::string> {
    httplib::Client client("http://" + peer);
    client.set_connection_timeout(std::chrono::milliseconds(200));
    client.set_read_timeout(std::chrono::seconds(1));
    const auto response = client.Get("/v1/models");
    if (!response || response->status != 200) {
      return std::nullopt;
    }
    return response->body;
  };
}

}  // namespace cluster
