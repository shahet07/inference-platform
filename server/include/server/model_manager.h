#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "cluster/model_registry.h"
#include "obs/log.h"
#include "obs/metrics.h"
#include "server/batching_engine.h"

namespace server {

// A model ready to wrap in an engine. `owner` keeps whatever `run` needs alive.
struct LoadedBackend {
  BatchRunner run;
  std::vector<std::int64_t> row_shape;
  std::shared_ptr<void> owner;
};

// One live model. Requests hold a shared_ptr while they run, so replacing a model cannot pull it
// out from under them: the old one finishes its queue and goes away with the last request.
class ServingModel {
 public:
  ServingModel(LoadedBackend backend, std::uint64_t version, const EngineConfig& config,
               obs::Registry& registry);

  [[nodiscard]] BatchingEngine& engine() { return engine_; }
  [[nodiscard]] std::uint64_t version() const { return version_; }

 private:
  std::shared_ptr<void> owner_;  // Declared first, so the engine stops before this is released.
  std::uint64_t version_;
  BatchingEngine engine_;
};

// Holds the models a node serves, and replaces them with no failed requests. Each model has its own
// queue and workers, so a slow or overloaded one cannot starve the others.
class ModelManager {
 public:
  using Loader = std::function<LoadedBackend(const std::filesystem::path&)>;

  // `config` applies to every model, with metric labels filled in per model. The model named
  // "default" gets no label, so that single-model setups keep plain metric names.
  ModelManager(Loader loader, EngineConfig config, obs::Registry& registry,
               obs::Logger* logger = nullptr);
  ~ModelManager();

  ModelManager(const ModelManager&) = delete;
  ModelManager& operator=(const ModelManager&) = delete;
  ModelManager(ModelManager&&) = delete;
  ModelManager& operator=(ModelManager&&) = delete;

  // Loads the file, checks that it can run, and only then swaps it in. On failure the current model
  // keeps serving, and this throws.
  void load(const std::string& name, const std::filesystem::path& path, std::uint64_t version);
  // The same, for a backend already in memory.
  void install(const std::string& name, LoadedBackend backend, std::uint64_t version);

  // Null if there is no such model.
  [[nodiscard]] std::shared_ptr<ServingModel> get(const std::string& name) const;
  [[nodiscard]] std::map<std::string, std::uint64_t> versions() const;

  // Follows `registry` on a background thread, loading each model whose metadata has a newer
  // version from `model_dir`. A uri must be a plain file name, since it can arrive over the
  // network.
  void followRegistry(cluster::ModelRegistry& registry, std::filesystem::path model_dir,
                      std::chrono::milliseconds interval);

 private:
  void reconcile(cluster::ModelRegistry& registry, const std::filesystem::path& model_dir);

  Loader loader_;
  EngineConfig config_;
  obs::Registry& registry_;
  obs::Logger* logger_;
  obs::Counter& swaps_total_;
  obs::Counter& load_failures_total_;

  mutable std::shared_mutex mutex_;
  std::map<std::string, std::shared_ptr<ServingModel>> models_;

  std::mutex follow_mutex_;
  std::condition_variable stop_following_;
  bool stopping_ = false;
  std::map<std::string, std::uint64_t> failed_versions_;  // Not retried until a newer one appears.
  std::thread follower_;
};

// Rejects anything that could escape the model directory.
[[nodiscard]] bool isPlainFileName(const std::string& name);

}  // namespace server
