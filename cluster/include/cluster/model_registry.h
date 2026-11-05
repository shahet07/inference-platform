#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cluster {

struct ModelMeta {
  std::string name;
  std::string uri;
  std::uint64_t version = 0;
  std::string updated_by;  // Breaks ties when two nodes write the same version.
};

// Model metadata replicated between nodes. Conflicts resolve by last-writer-wins on
// (version, updated_by). Merging is commutative, associative, and idempotent, so nodes converge
// whatever order updates arrive in, and however many times. This holds metadata only: it does
// not load models, and a concurrent update can lose to the other write without any error.
class ModelRegistry {
 public:
  explicit ModelRegistry(std::string node_id) : node_id_(std::move(node_id)) {}

  // Writes a new version above any this node has seen for `name`.
  ModelMeta put(const std::string& name, const std::string& uri);
  // Returns how many entries changed.
  std::size_t merge(const std::vector<ModelMeta>& entries);

  [[nodiscard]] std::vector<ModelMeta> list() const;
  [[nodiscard]] std::string toJson() const;
  // Throws std::runtime_error if the JSON is malformed.
  std::size_t mergeJson(std::string_view json);

 private:
  std::string node_id_;
  mutable std::mutex mutex_;
  std::map<std::string, ModelMeta> models_;
};

}  // namespace cluster
