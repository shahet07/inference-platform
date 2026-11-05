#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace cluster {

// A stable 64-bit hash, so every process and platform places keys on the same nodes.
[[nodiscard]] std::uint64_t stableHash(std::string_view text);

// Consistent hashing with virtual nodes. Adding or removing a node only moves the keys that
// belong on it, about 1/N of them, so a failure does not reshuffle the whole cluster.
class HashRing {
 public:
  explicit HashRing(std::size_t virtual_nodes = 128) : virtual_nodes_(virtual_nodes) {}

  void add(const std::string& node);
  void remove(const std::string& node);

  // The first `count` distinct nodes clockwise from `key`: its owner, then its failover order.
  [[nodiscard]] std::vector<std::string> nodesFor(std::string_view key, std::size_t count) const;
  [[nodiscard]] std::size_t nodeCount() const { return node_count_; }

 private:
  std::size_t virtual_nodes_;
  std::size_t node_count_ = 0;
  std::map<std::uint64_t, std::string> ring_;
};

}  // namespace cluster
