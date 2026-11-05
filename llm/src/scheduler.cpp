#include "llm/scheduler.h"

#include <algorithm>
#include <exception>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace llm {
namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

Scheduler::Scheduler(Backend& backend, const SchedulerConfig& config, obs::Registry& registry)
    : backend_(backend),
      config_(config),
      tokens_total_(registry.counter("llm_tokens_total", "Tokens generated.")),
      requests_total_(registry.counter("llm_requests_total", "Requests accepted.")),
      rejected_total_(registry.counter("llm_rejected_total", "Requests rejected: queue full.")),
      cancelled_total_(
          registry.counter("llm_cancelled_total", "Requests cancelled by the client.")),
      active_slots_(registry.gauge("llm_active_slots", "Sequences generating now.")),
      queue_depth_(registry.gauge("llm_queue_depth", "Requests waiting for a slot.")),
      time_to_first_token_(registry.histogram("llm_time_to_first_token_seconds",
                                              "Time from accepted to first token.",
                                              obs::latencyBuckets())),
      batch_tokens_(registry.histogram("llm_batch_sequences", "Sequences in each step.",
                                       {1, 2, 4, 8, 16, 32, 64, 128})) {
  for (auto seq = static_cast<std::int32_t>(config.max_slots) - 1; seq >= 0; --seq) {
    free_seqs_.push_back(seq);
  }
  thread_ = std::thread([this] { run(); });
}

Scheduler::~Scheduler() {
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  thread_.join();
}

bool Scheduler::submit(GenerateRequest request) {
  {
    const std::scoped_lock lock(mutex_);
    if (stopping_ || queue_.size() >= config_.queue_capacity) {
      rejected_total_.add();
      return false;
    }
    queue_.push_back({.request = std::move(request), .submitted = Clock::now()});
    queue_depth_.set(static_cast<std::int64_t>(queue_.size()));
  }
  requests_total_.add();
  wake_.notify_one();
  return true;
}

void Scheduler::run() {
  while (true) {
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || !queue_.empty() || !slots_.empty(); });
      if (stopping_) {
        break;
      }
    }
    admit();
    stepOnce();
  }
  for (std::size_t i = slots_.size(); i > 0; --i) {
    finish(i - 1, FinishReason::kCancelled);
  }
  const std::scoped_lock lock(mutex_);
  for (Queued& queued : queue_) {
    queued.request.on_finish(FinishReason::kCancelled);
  }
  queue_.clear();
}

void Scheduler::admit() {
  // Static batching takes new work only once the running batch is empty.
  if (config_.policy == BatchPolicy::kStatic && !slots_.empty()) {
    return;
  }
  while (!free_seqs_.empty()) {
    Queued queued;
    {
      const std::scoped_lock lock(mutex_);
      if (queue_.empty()) {
        return;
      }
      queued = std::move(queue_.front());
      queue_.pop_front();
      queue_depth_.set(static_cast<std::int64_t>(queue_.size()));
    }
    if (queued.request.handle && queued.request.handle->cancelled.load()) {
      cancelled_total_.add();
      queued.request.on_finish(FinishReason::kCancelled);
      continue;
    }
    std::vector<Token> prompt;
    try {
      prompt = backend_.tokenize(queued.request.prompt);
    } catch (const std::exception&) {
      queued.request.on_finish(FinishReason::kError);
      continue;
    }
    // The model must have room for at least one new token.
    if (prompt.empty() || std::cmp_greater_equal(prompt.size(), config_.max_sequence_tokens)) {
      queued.request.on_finish(FinishReason::kError);
      continue;
    }
    const std::int32_t seq = free_seqs_.back();
    free_seqs_.pop_back();
    Slot slot;
    slot.seq = seq;
    slot.request = std::move(queued.request);
    slot.pending = std::move(prompt);
    slot.submitted = queued.submitted;
    slots_.push_back(std::move(slot));
  }
}

void Scheduler::finish(std::size_t index, FinishReason reason) {
  Slot slot = std::move(slots_[index]);
  slots_.erase(slots_.begin() + static_cast<std::ptrdiff_t>(index));
  backend_.release(slot.seq);
  free_seqs_.push_back(slot.seq);
  if (reason == FinishReason::kCancelled) {
    cancelled_total_.add();
  }
  slot.request.on_finish(reason);
}

void Scheduler::stepOnce() {
  // Cancelled requests stop here, freeing their slots for this step.
  for (std::size_t i = slots_.size(); i > 0; --i) {
    const auto& handle = slots_[i - 1].request.handle;
    if (handle && handle->cancelled.load()) {
      finish(i - 1, FinishReason::kCancelled);
    }
  }
  active_slots_.set(static_cast<std::int64_t>(slots_.size()));
  if (slots_.empty()) {
    return;
  }

  std::vector<SeqStep> steps;
  steps.reserve(slots_.size());
  for (const Slot& slot : slots_) {
    steps.push_back({.seq = slot.seq, .tokens = slot.pending, .position = slot.position});
  }
  batch_tokens_.observe(static_cast<double>(steps.size()));

  std::vector<Token> next;
  try {
    next = backend_.step(steps);
    if (next.size() != slots_.size()) {
      throw std::runtime_error("the backend returned the wrong number of tokens");
    }
  } catch (const std::exception&) {
    // The state of every sequence in the batch is now unknown, so they all end.
    for (std::size_t i = slots_.size(); i > 0; --i) {
      finish(i - 1, FinishReason::kError);
    }
    return;
  }

  std::vector<std::pair<std::size_t, FinishReason>> finished;
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    Slot& slot = slots_[i];
    slot.position += static_cast<std::int32_t>(slot.pending.size());
    if (slot.generated == 0) {
      time_to_first_token_.observe(
          std::chrono::duration<double>(Clock::now() - slot.submitted).count());
    }
    const Token token = next[i];
    if (backend_.isEnd(token)) {
      finished.emplace_back(i, FinishReason::kEnd);
      continue;
    }
    slot.request.on_token(backend_.text(token));
    tokens_total_.add();
    ++slot.generated;
    slot.pending.assign(1, token);
    if (slot.generated >= slot.request.max_tokens ||
        slot.position + 1 >= config_.max_sequence_tokens) {
      finished.emplace_back(i, FinishReason::kMaxTokens);
    }
  }
  // Highest index first, so that removing one does not move the others still to be removed.
  for (const auto& [index, reason] : std::views::reverse(finished)) {
    finish(index, reason);
  }
  active_slots_.set(static_cast<std::int64_t>(slots_.size()));
}

}  // namespace llm
