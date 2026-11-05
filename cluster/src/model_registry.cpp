#include "cluster/model_registry.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <tuple>

namespace cluster {
namespace {

bool isNewer(const ModelMeta& a, const ModelMeta& b) {
  return std::tie(a.version, a.updated_by) > std::tie(b.version, b.updated_by);
}

}  // namespace

ModelMeta ModelRegistry::put(const std::string& name, const std::string& uri) {
  const std::scoped_lock lock(mutex_);
  ModelMeta& entry = models_[name];
  entry = {.name = name, .uri = uri, .version = entry.version + 1, .updated_by = node_id_};
  return entry;
}

std::size_t ModelRegistry::merge(const std::vector<ModelMeta>& entries) {
  const std::scoped_lock lock(mutex_);
  std::size_t changed = 0;
  for (const ModelMeta& incoming : entries) {
    const auto it = models_.find(incoming.name);
    if (it == models_.end()) {
      models_.emplace(incoming.name, incoming);
      ++changed;
    } else if (isNewer(incoming, it->second)) {
      it->second = incoming;
      ++changed;
    }
  }
  return changed;
}

std::vector<ModelMeta> ModelRegistry::list() const {
  const std::scoped_lock lock(mutex_);
  std::vector<ModelMeta> entries;
  entries.reserve(models_.size());
  for (const auto& [name, entry] : models_) {
    entries.push_back(entry);
  }
  return entries;
}

std::string ModelRegistry::toJson() const {
  nlohmann::json models = nlohmann::json::array();
  for (const ModelMeta& entry : list()) {
    models.push_back({{"name", entry.name},
                      {"uri", entry.uri},
                      {"version", entry.version},
                      {"updated_by", entry.updated_by}});
  }
  return nlohmann::json{{"models", models}}.dump();
}

std::size_t ModelRegistry::mergeJson(std::string_view json) {
  std::vector<ModelMeta> entries;
  try {
    const nlohmann::json parsed = nlohmann::json::parse(json);
    for (const auto& item : parsed.at("models")) {
      entries.push_back({.name = item.at("name").get<std::string>(),
                         .uri = item.at("uri").get<std::string>(),
                         .version = item.at("version").get<std::uint64_t>(),
                         .updated_by = item.at("updated_by").get<std::string>()});
    }
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(std::string("malformed model metadata: ") + e.what());
  }
  return merge(entries);
}

}  // namespace cluster
