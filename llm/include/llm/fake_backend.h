#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "llm/scheduler.h"

namespace llm {

// Follows the Backend contract with no model, for tests and for measuring the serving layers alone.
// Each word of the prompt is a token, and the output counts up from its length. It checks that it
// is used correctly, which is what catches scheduler bugs: positions must continue, and sequences
// must exist.
class FakeBackend final : public Backend {
 public:
  explicit FakeBackend(std::chrono::microseconds step_time = std::chrono::microseconds(0))
      : step_time_(step_time) {}

  static constexpr Token kEnd = -1;

  [[nodiscard]] std::vector<Token> tokenize(std::string_view text) override {
    std::vector<Token> tokens;
    bool in_word = false;
    for (const char c : text) {
      const bool is_word = c != ' ';
      if (is_word && !in_word) {
        tokens.push_back(static_cast<Token>(tokens.size()) + 1);
      }
      in_word = is_word;
    }
    return tokens;
  }

  [[nodiscard]] std::vector<Token> step(std::span<const SeqStep> steps) override {
    if (step_time_.count() > 0) {
      std::this_thread::sleep_for(step_time_);
    }
    std::vector<Token> next;
    next.reserve(steps.size());
    for (const SeqStep& step : steps) {
      if (step.seq >= static_cast<std::int32_t>(positions_.size())) {
        positions_.resize(static_cast<std::size_t>(step.seq) + 1, 0);
      }
      std::int32_t& position = positions_[static_cast<std::size_t>(step.seq)];
      if (position != step.position) {
        throw std::logic_error("step at the wrong position for its sequence");
      }
      position += static_cast<std::int32_t>(step.tokens.size());
      next.push_back(end_at_ > 0 && position >= end_at_ ? kEnd : position);
    }
    last_step_sizes.push_back(steps.size());
    return next;
  }

  void release(std::int32_t seq) override {
    if (seq < static_cast<std::int32_t>(positions_.size())) {
      positions_[static_cast<std::size_t>(seq)] = 0;
    }
  }

  [[nodiscard]] bool isEnd(Token token) const override { return token == kEnd; }
  [[nodiscard]] std::string text(Token token) const override {
    return "t" + std::to_string(token) + " ";
  }

  void endAt(std::int32_t position) { end_at_ = position; }

  std::vector<std::size_t> last_step_sizes;  // Only for reading after the scheduler has stopped.

 private:
  std::chrono::microseconds step_time_;
  std::int32_t end_at_ = 0;
  std::vector<std::int32_t> positions_;
};

}  // namespace llm
