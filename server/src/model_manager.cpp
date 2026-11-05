#include "server/model_manager.h"

#include <exception>
#include <functional>
#include <numeric>
#include <utility>

namespace server {

bool isPlainFileName(const std::string& name) {
  if (name.empty() || name == "." || name == "..") {
    return false;
  }
  return name.find('/') == std::string::npos && name.find('\\') == std::string::npos &&
         name.find('\0') == std::string::npos;
}

ServingModel::ServingModel(LoadedBackend backend, std::uint64_t version, const EngineConfig& config,
                           obs::Registry& registry)
    : owner_(std::move(backend.owner)),
      version_(version),
      engine_(std::move(backend.run), std::move(backend.row_shape), config, registry) {}

ModelManager::ModelManager(Loader loader, EngineConfig config, obs::Registry& registry,
                           obs::Logger* logger)
    : loader_(std::move(loader)),
      config_(std::move(config)),
      registry_(registry),
      logger_(logger),
      swaps_total_(registry.counter("model_swaps_total", "Models loaded and swapped in.")),
      load_failures_total_(
          registry.counter("model_load_failures_total", "Model loads that failed.")) {}

ModelManager::~ModelManager() {
  {
    const std::scoped_lock lock(follow_mutex_);
    stopping_ = true;
  }
  stop_following_.notify_all();
  if (follower_.joinable()) {
    follower_.join();
  }
}

void ModelManager::load(const std::string& name, const std::filesystem::path& path,
                        std::uint64_t version) {
  try {
    install(name, loader_(path), version);
  } catch (const std::exception& e) {
    load_failures_total_.add();
    if (logger_ != nullptr) {
      logger_->error("model load failed",
                     {{"model", name}, {"path", path.string()}, {"error", std::string(e.what())}});
    }
    throw;
  }
}

void ModelManager::install(const std::string& name, LoadedBackend backend, std::uint64_t version) {
  // Run it once before it takes traffic. That rejects a model that loads but cannot run, and means
  // the first real request does not pay for lazy setup.
  const auto features = static_cast<std::size_t>(std::accumulate(
      backend.row_shape.begin(), backend.row_shape.end(), std::int64_t{1}, std::multiplies<>()));
  std::vector<std::int64_t> shape{1};
  shape.insert(shape.end(), backend.row_shape.begin(), backend.row_shape.end());
  (void)backend.run(std::vector<float>(features, 0.0F), shape);

  EngineConfig config = config_;
  if (name != "default") {
    config.metric_labels = "model=\"" + name + "\"";
  }
  auto model = std::make_shared<ServingModel>(std::move(backend), version, config, registry_);
  std::shared_ptr<ServingModel> previous;
  {
    const std::unique_lock lock(mutex_);
    previous = std::exchange(models_[name], std::move(model));
  }
  swaps_total_.add();
  if (logger_ != nullptr) {
    logger_->info("model swapped in",
                  {{"model", name}, {"version", static_cast<std::int64_t>(version)}});
  }
  // `previous` finishes its queue when the last request holding it returns, possibly right here.
}

std::shared_ptr<ServingModel> ModelManager::get(const std::string& name) const {
  const std::shared_lock lock(mutex_);
  const auto it = models_.find(name);
  return it == models_.end() ? nullptr : it->second;
}

std::map<std::string, std::uint64_t> ModelManager::versions() const {
  const std::shared_lock lock(mutex_);
  std::map<std::string, std::uint64_t> out;
  for (const auto& [name, model] : models_) {
    out[name] = model->version();
  }
  return out;
}

void ModelManager::followRegistry(cluster::ModelRegistry& registry, std::filesystem::path model_dir,
                                  std::chrono::milliseconds interval) {
  follower_ = std::thread([this, &registry, dir = std::move(model_dir), interval] {
    std::unique_lock lock(follow_mutex_);
    while (!stopping_) {
      lock.unlock();
      reconcile(registry, dir);
      lock.lock();
      stop_following_.wait_for(lock, interval, [this] { return stopping_; });
    }
  });
}

void ModelManager::reconcile(cluster::ModelRegistry& registry,
                             const std::filesystem::path& model_dir) {
  const std::map<std::string, std::uint64_t> current = versions();
  for (const cluster::ModelMeta& meta : registry.list()) {
    const auto it = current.find(meta.name);
    if (it != current.end() && it->second >= meta.version) {
      continue;
    }
    const auto failed = failed_versions_.find(meta.name);
    if (failed != failed_versions_.end() && failed->second >= meta.version) {
      continue;
    }
    if (!isPlainFileName(meta.uri)) {
      failed_versions_[meta.name] = meta.version;
      load_failures_total_.add();
      if (logger_ != nullptr) {
        logger_->error("refused model uri: not a plain file name",
                       {{"model", meta.name}, {"uri", meta.uri}});
      }
      continue;
    }
    try {
      load(meta.name, model_dir / meta.uri, meta.version);
    } catch (const std::exception&) {
      failed_versions_[meta.name] = meta.version;
    }
  }
}

}  // namespace server
