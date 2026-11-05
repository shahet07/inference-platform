#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "llm/scheduler.h"

namespace llm {

// Runs a GGUF model through llama.cpp, on the CPU, always picking the most likely next token, so a
// given prompt always gives the same output. Needs a build with IP_ENABLE_LLAMA.
class LlamaBackend final : public Backend {
 public:
  // `context_per_sequence` is how many tokens each of `sequences` can hold.
  LlamaBackend(const std::filesystem::path& model, std::int32_t sequences,
               std::int32_t context_per_sequence, std::int32_t threads);
  ~LlamaBackend() override;

  [[nodiscard]] std::vector<Token> tokenize(std::string_view text) override;
  [[nodiscard]] std::vector<Token> step(std::span<const SeqStep> steps) override;
  void release(std::int32_t seq) override;
  [[nodiscard]] bool isEnd(Token token) const override;
  [[nodiscard]] std::string text(Token token) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace llm
