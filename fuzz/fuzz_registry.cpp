// Peers send this JSON. Bad input must throw only std::runtime_error, and any state it produces
// must replicate exactly to another node.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include "cluster/model_registry.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  cluster::ModelRegistry source("source");
  try {
    source.mergeJson(text);
  } catch (const std::runtime_error&) {
    return 0;
  }
  cluster::ModelRegistry replica("replica");
  replica.mergeJson(source.toJson());
  if (source.toJson() != replica.toJson()) {
    std::abort();
  }
  return 0;
}
