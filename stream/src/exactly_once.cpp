#include "stream/exactly_once.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace stream {
namespace {

namespace fs = std::filesystem;

constexpr const char* kTrailer = "end";

void syncPath(const fs::path& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::system_error(errno, std::generic_category(), "open " + path.string());
  }
  const int result = fsync(fd);
  close(fd);
  if (result != 0) {
    throw std::system_error(errno, std::generic_category(), "fsync " + path.string());
  }
}

}  // namespace

bool ExactlyOnceAggregator::parseCheckpoint(const std::string& text, std::uint64_t& offset,
                                            std::map<std::string, KeyTotals>& totals) {
  std::istringstream in(text);
  std::size_t keys = 0;
  std::uint64_t parsed_offset = 0;
  if (!(in >> parsed_offset >> keys) || keys > text.size()) {
    return false;
  }
  std::map<std::string, KeyTotals> parsed;
  for (std::size_t i = 0; i < keys; ++i) {
    std::string key;
    KeyTotals key_totals;
    if (!(in >> std::quoted(key) >> key_totals.count >> key_totals.sum)) {
      return false;
    }
    parsed[key] = key_totals;
  }
  std::string trailer;
  if (!(in >> trailer) || trailer != kTrailer) {
    return false;
  }
  offset = parsed_offset;
  totals = std::move(parsed);
  return true;
}

ExactlyOnceAggregator::ExactlyOnceAggregator(ShardedLog& log, ShardedLog::GroupId group,
                                             std::size_t shard, const fs::path& dir)
    : log_(log), group_(group), shard_(shard) {
  fs::create_directories(dir);
  file_ = dir / ("state-" + std::to_string(shard) + ".checkpoint");
  if (fs::exists(file_)) {
    std::ifstream in(file_);
    std::ostringstream contents;
    contents << in.rdbuf();
    // The file only ever appears whole, through a rename, so a bad one means real damage. Starting
    // over silently would double count, so this is an error.
    if (!parseCheckpoint(contents.str(), offset_, totals_)) {
      throw std::runtime_error("corrupt checkpoint: " + file_.string());
    }
  }
}

std::size_t ExactlyOnceAggregator::poll(std::size_t max_events) {
  const ReadResult read = log_.read(shard_, offset_, max_events);
  for (const Event& event : read.events) {
    KeyTotals& key_totals = totals_[event.key];
    ++key_totals.count;
    key_totals.sum += event.value;
  }
  skipped_ += read.skipped;
  offset_ = read.next_offset;
  return read.events.size();
}

void ExactlyOnceAggregator::checkpoint() {
  const fs::path temporary = file_.string() + ".tmp";
  {
    std::ofstream out(temporary, std::ios::trunc);
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << offset_ << ' ' << totals_.size() << '\n';
    for (const auto& [key, key_totals] : totals_) {
      out << std::quoted(key) << ' ' << key_totals.count << ' ' << key_totals.sum << '\n';
    }
    out << kTrailer << '\n';
    if (!out) {
      throw std::runtime_error("cannot write checkpoint: " + temporary.string());
    }
  }
  syncPath(temporary);
  fs::rename(temporary, file_);
  syncPath(file_.parent_path());
  // Only now is it safe for the log to forget these events.
  log_.commit(group_, shard_, offset_);
}

}  // namespace stream
