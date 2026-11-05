#include "cluster/hash_ring.h"

#include <algorithm>
#include <string>

namespace cluster {

std::uint64_t stableHash(std::string_view text) {
  // FNV-1a, followed by MurmurHash3's finalizer: FNV alone clusters similar short strings such
  // as "node-1#0" and "node-1#1", which would unbalance the ring.
  std::uint64_t hash = 14695981039346656037ULL;
  for (const char c : text) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ULL;
  }
  hash ^= hash >> 33U;
  hash *= 0xff51afd7ed558ccdULL;
  hash ^= hash >> 33U;
  hash *= 0xc4ceb9fe1a85ec53ULL;
  hash ^= hash >> 33U;
  return hash;
}

void HashRing::add(const std::string& node) {
  bool added = false;
  for (std::size_t i = 0; i < virtual_nodes_; ++i) {
    added = ring_.try_emplace(stableHash(node + "#" + std::to_string(i)), node).second || added;
  }
  if (added) {
    ++node_count_;
  }
}

void HashRing::remove(const std::string& node) {
  const std::size_t removed =
      std::erase_if(ring_, [&node](const auto& entry) { return entry.second == node; });
  if (removed > 0) {
    --node_count_;
  }
}

std::vector<std::string> HashRing::nodesFor(std::string_view key, std::size_t count) const {
  std::vector<std::string> nodes;
  if (ring_.empty()) {
    return nodes;
  }
  const std::size_t wanted = std::min(count, node_count_);
  nodes.reserve(wanted);
  auto it = ring_.lower_bound(stableHash(key));
  // Stops after one full turn, so a shorter result can never loop forever.
  for (std::size_t step = 0; step < ring_.size() && nodes.size() < wanted; ++step, ++it) {
    if (it == ring_.end()) {
      it = ring_.begin();
    }
    if (std::ranges::find(nodes, it->second) == nodes.end()) {
      nodes.push_back(it->second);
    }
  }
  return nodes;
}

}  // namespace cluster
