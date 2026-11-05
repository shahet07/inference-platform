#include "llm/llama_backend.h"

#include <llama.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>

namespace llm {

struct LlamaBackend::Impl {
  llama_model* model = nullptr;
  llama_context* context = nullptr;
  const llama_vocab* vocab = nullptr;
  std::int32_t vocab_size = 0;
  llama_batch batch{};
};

LlamaBackend::LlamaBackend(const std::filesystem::path& model, std::int32_t sequences,
                           std::int32_t context_per_sequence, std::int32_t threads)
    : impl_(std::make_unique<Impl>()) {
  llama_backend_init();
  llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  impl_->model = llama_model_load_from_file(model.c_str(), model_params);
  if (impl_->model == nullptr) {
    throw std::runtime_error("cannot load model: " + model.string());
  }
  llama_context_params context_params = llama_context_default_params();
  const auto capacity = static_cast<std::uint32_t>(sequences * context_per_sequence);
  context_params.n_ctx = capacity;
  context_params.n_batch = capacity;
  context_params.n_ubatch = capacity;
  context_params.n_seq_max = static_cast<std::uint32_t>(sequences);
  context_params.n_threads = threads;
  context_params.n_threads_batch = threads;
  impl_->context = llama_init_from_model(impl_->model, context_params);
  if (impl_->context == nullptr) {
    llama_model_free(impl_->model);
    throw std::runtime_error("cannot create a context for: " + model.string());
  }
  impl_->vocab = llama_model_get_vocab(impl_->model);
  impl_->vocab_size = llama_vocab_n_tokens(impl_->vocab);
  impl_->batch = llama_batch_init(static_cast<std::int32_t>(capacity), 0, 1);
}

LlamaBackend::~LlamaBackend() {
  llama_batch_free(impl_->batch);
  llama_free(impl_->context);
  llama_model_free(impl_->model);
  llama_backend_free();
}

std::vector<Token> LlamaBackend::tokenize(std::string_view text) {
  const auto length = static_cast<std::int32_t>(text.size());
  std::vector<llama_token> tokens(text.size() + 8);
  const std::int32_t count = llama_tokenize(impl_->vocab, text.data(), length, tokens.data(),
                                            static_cast<std::int32_t>(tokens.size()), true, false);
  if (count < 0) {
    throw std::runtime_error("cannot tokenize prompt");
  }
  tokens.resize(static_cast<std::size_t>(count));
  return tokens;
}

std::vector<Token> LlamaBackend::step(std::span<const SeqStep> steps) {
  llama_batch& batch = impl_->batch;
  batch.n_tokens = 0;
  std::vector<std::int32_t> last_of_sequence;
  last_of_sequence.reserve(steps.size());
  for (const SeqStep& step : steps) {
    for (std::size_t i = 0; i < step.tokens.size(); ++i) {
      const auto at = static_cast<std::size_t>(batch.n_tokens);
      batch.token[at] = step.tokens[i];
      batch.pos[at] = step.position + static_cast<std::int32_t>(i);
      batch.n_seq_id[at] = 1;
      batch.seq_id[at][0] = step.seq;
      // Logits cost time, and are only needed for each sequence's last token.
      batch.logits[at] = i + 1 == step.tokens.size() ? 1 : 0;
      ++batch.n_tokens;
    }
    last_of_sequence.push_back(batch.n_tokens - 1);
  }
  if (llama_decode(impl_->context, batch) != 0) {
    throw std::runtime_error("llama_decode failed");
  }
  std::vector<Token> next;
  next.reserve(steps.size());
  for (const std::int32_t index : last_of_sequence) {
    const float* logits = llama_get_logits_ith(impl_->context, index);
    if (logits == nullptr) {
      throw std::runtime_error("no logits from llama_decode");
    }
    next.push_back(
        static_cast<Token>(std::max_element(logits, logits + impl_->vocab_size) - logits));
  }
  return next;
}

void LlamaBackend::release(std::int32_t seq) {
  (void)llama_memory_seq_rm(llama_get_memory(impl_->context), seq, -1, -1);
}

bool LlamaBackend::isEnd(Token token) const { return llama_vocab_is_eog(impl_->vocab, token); }

std::string LlamaBackend::text(Token token) const {
  std::array<char, 256> buffer{};
  const std::int32_t length = llama_token_to_piece(
      impl_->vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()), 0, false);
  return length > 0 ? std::string(buffer.data(), static_cast<std::size_t>(length)) : std::string();
}

}  // namespace llm
