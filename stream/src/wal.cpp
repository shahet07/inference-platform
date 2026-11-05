#include "stream/wal.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace stream {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kHeaderBytes = 8;  // Payload size and CRC.
constexpr std::size_t kFixedPayloadBytes = 8 + 4 + 8 + 8;
constexpr std::size_t kMaxPayloadBytes = std::size_t{64} * 1024 * 1024;

std::array<std::uint32_t, 256> makeCrcTable() {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int bit = 0; bit < 8; ++bit) {
      c = ((c & 1U) != 0) ? 0xEDB88320U ^ (c >> 1U) : c >> 1U;
    }
    table[i] = c;
  }
  return table;
}

std::uint32_t crc32(std::string_view data) {
  static const std::array<std::uint32_t, 256> kTable = makeCrcTable();
  std::uint32_t c = 0xFFFFFFFFU;
  for (const char byte : data) {
    c = kTable[(c ^ static_cast<unsigned char>(byte)) & 0xFFU] ^ (c >> 8U);
  }
  return c ^ 0xFFFFFFFFU;
}

template <typename T>
void putValue(std::string& out, T value) {
  static_assert(std::endian::native == std::endian::little, "the log format is little-endian");
  const auto bytes = std::bit_cast<std::array<char, sizeof(T)>>(value);
  out.append(bytes.data(), bytes.size());
}

template <typename T>
T getValue(std::string_view data, std::size_t position) {
  std::array<char, sizeof(T)> bytes{};
  std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(position), sizeof(T), bytes.begin());
  return std::bit_cast<T>(bytes);
}

void appendRecord(std::string& out, std::uint64_t offset, const Event& event) {
  std::string payload;
  payload.reserve(kFixedPayloadBytes + event.key.size());
  putValue<std::uint64_t>(payload, offset);
  putValue<std::uint32_t>(payload, static_cast<std::uint32_t>(event.key.size()));
  payload += event.key;
  putValue<double>(payload, event.value);
  putValue<std::int64_t>(payload, event.timestamp_ms);
  putValue<std::uint32_t>(out, static_cast<std::uint32_t>(payload.size()));
  putValue<std::uint32_t>(out, crc32(payload));
  out += payload;
}

[[noreturn]] void throwErrno(const std::string& what) {
  throw std::system_error(errno, std::generic_category(), what);
}

// A plain fsync on macOS only reaches the drive's cache. F_FULLFSYNC asks the drive to flush too.
void syncFile(int fd) {
#ifdef __APPLE__
  if (fcntl(fd, F_FULLFSYNC) == 0) {
    return;
  }
  // Some filesystems do not support it, so fall through.
#endif
  if (fsync(fd) != 0) {
    throwErrno("fsync");
  }
}

// A new or renamed file only survives a power failure once its directory has been synced.
void syncDirectory(const fs::path& dir) {
  const int fd = open(dir.c_str(), O_RDONLY);
  if (fd < 0) {
    throwErrno("open directory");
  }
  const int result = fsync(fd);
  close(fd);
  if (result != 0) {
    throwErrno("fsync directory");
  }
}

std::string readFile(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot read " + path.string());
  }
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

std::string segmentFileName(std::uint64_t first_offset) {
  std::string digits = std::to_string(first_offset);
  return std::string(20 - digits.size(), '0') + digits + ".log";
}

}  // namespace

std::size_t parseRecords(std::string_view data, std::uint64_t expected_offset,
                         const std::function<void(std::uint64_t, Event)>& on_event) {
  std::size_t position = 0;
  while (data.size() - position >= kHeaderBytes) {
    const auto size = getValue<std::uint32_t>(data, position);
    const auto crc = getValue<std::uint32_t>(data, position + 4);
    if (size < kFixedPayloadBytes || size > kMaxPayloadBytes ||
        size > data.size() - position - kHeaderBytes) {
      break;
    }
    const std::string_view payload = data.substr(position + kHeaderBytes, size);
    if (crc32(payload) != crc) {
      break;
    }
    const auto offset = getValue<std::uint64_t>(payload, 0);
    const auto key_size = getValue<std::uint32_t>(payload, 8);
    if (offset != expected_offset || key_size != size - kFixedPayloadBytes) {
      break;
    }
    Event event;
    event.key.assign(payload.substr(12, key_size));
    event.value = getValue<double>(payload, 12 + key_size);
    event.timestamp_ms = getValue<std::int64_t>(payload, 20 + key_size);
    on_event(offset, std::move(event));
    ++expected_offset;
    position += kHeaderBytes + size;
  }
  return position;
}

DirectorySegmentStore::DirectorySegmentStore(fs::path dir) : dir_(std::move(dir)) {
  fs::create_directories(dir_);
}

void DirectorySegmentStore::put(const fs::path& local_file, const std::string& name) {
  const fs::path destination = dir_ / name;
  const fs::path temporary = dir_ / (name + ".tmp");
  fs::copy_file(local_file, temporary, fs::copy_options::overwrite_existing);
  const int fd = open(temporary.c_str(), O_RDONLY);
  if (fd < 0) {
    throwErrno("open archived segment");
  }
  syncFile(fd);
  close(fd);
  fs::rename(temporary, destination);
  syncDirectory(dir_);
}

std::vector<std::string> DirectorySegmentStore::list(const std::string& prefix) const {
  std::vector<std::string> names;
  for (const auto& entry : fs::directory_iterator(dir_)) {
    const std::string name = entry.path().filename().string();
    if (name.starts_with(prefix) && !name.ends_with(".tmp")) {
      names.push_back(name);
    }
  }
  std::ranges::sort(names);
  return names;
}

std::string DirectorySegmentStore::get(const std::string& name) const {
  return readFile(dir_ / name);
}

ShardWal::ShardWal(const WalConfig& config, std::size_t shard)
    : config_(config), shard_(shard), shard_dir_(config.dir / ("shard-" + std::to_string(shard))) {
  fs::create_directories(shard_dir_);
}

ShardWal::~ShardWal() {
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
  }
  has_data_.notify_all();
  durable_.notify_all();
  if (interval_syncer_.joinable()) {
    interval_syncer_.join();
  }
  if (flusher_.joinable()) {
    flusher_.join();
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

RecoveredShard ShardWal::recover(std::size_t max_events) {
  std::vector<Segment> segments;
  for (const auto& entry : fs::directory_iterator(shard_dir_)) {
    if (entry.path().extension() == ".log") {
      segments.push_back(
          {.first_offset = std::stoull(entry.path().stem().string()), .path = entry.path()});
    }
  }
  std::ranges::sort(segments, {}, &Segment::first_offset);

  RecoveredShard recovered;
  // Events are kept only up to the maximum, oldest dropped first, with reads of older ones served
  // by whatever is still on disk or in the archive.
  std::uint64_t next = segments.empty() ? 0 : segments.front().first_offset;
  recovered.first_offset = next;
  for (std::size_t i = 0; i < segments.size(); ++i) {
    const Segment& segment = segments[i];
    const bool is_last = i + 1 == segments.size();
    if (segment.first_offset != next) {
      throw std::runtime_error("write-ahead log has a gap before " + segment.path.string());
    }
    const std::string data = readFile(segment.path);
    const std::size_t valid = parseRecords(data, next, [&](std::uint64_t offset, Event event) {
      recovered.events.push_back(std::move(event));
      if (recovered.events.size() > max_events * 2) {
        // Dropping in bulk keeps this linear however long the log is.
        recovered.events.erase(recovered.events.begin(),
                               recovered.events.end() - static_cast<std::ptrdiff_t>(max_events));
      }
      next = offset + 1;
    });
    if (valid != data.size()) {
      if (!is_last) {
        // Sealed segments were synced before the next one began, so this is not a torn write.
        throw std::runtime_error("write-ahead log is corrupt in " + segment.path.string());
      }
      recovered.truncated_bytes = data.size() - valid;
      fs::resize_file(segment.path, valid);
    }
  }
  if (recovered.events.size() > max_events) {
    recovered.events.erase(recovered.events.begin(),
                           recovered.events.end() - static_cast<std::ptrdiff_t>(max_events));
  }
  recovered.next_offset = next;
  recovered.first_offset = next - recovered.events.size();

  if (segments.empty()) {
    openSegment(0);
  } else {
    for (std::size_t i = 0; i + 1 < segments.size(); ++i) {
      sealed_.push_back(segments[i]);
    }
    const Segment& last = segments.back();
    current_first_ = last.first_offset;
    fd_ = open(last.path.c_str(), O_WRONLY | O_APPEND);
    if (fd_ < 0) {
      throwErrno("open segment");
    }
    segment_size_ = fs::file_size(last.path);
    if (recovered.truncated_bytes > 0) {
      syncFile(fd_);
    }
  }
  buffered_up_to_ = next;
  durable_up_to_ = next;
  start();
  return recovered;
}

void ShardWal::start() {
  flusher_ = std::thread([this] { flushLoop(); });
  if (config_.fsync == FsyncPolicy::kInterval) {
    interval_syncer_ = std::thread([this] { intervalSyncLoop(); });
  }
}

void ShardWal::enqueue(std::uint64_t offset, const Event& event) {
  {
    const std::scoped_lock lock(mutex_);
    appendRecord(buffer_, offset, event);
    buffered_up_to_ = offset + 1;
  }
  has_data_.notify_one();
}

bool ShardWal::waitDurable(std::uint64_t offset) {
  std::unique_lock lock(mutex_);
  durable_.wait(lock, [&] { return durable_up_to_ > offset || failed_ || stopping_; });
  return durable_up_to_ > offset;
}

void ShardWal::flushLoop() {
  std::unique_lock lock(mutex_);
  while (true) {
    has_data_.wait(lock, [this] { return stopping_ || !buffer_.empty() || trim_below_ > 0; });
    const bool stopping = stopping_;
    std::string bytes;
    bytes.swap(buffer_);
    const std::uint64_t next_offset = buffered_up_to_;
    lock.unlock();
    bool ok = true;
    try {
      if (!bytes.empty()) {
        writeOut(bytes, next_offset);
        // Sharing this sync among everything buffered is the group commit.
        if (config_.fsync == FsyncPolicy::kEveryAppend) {
          syncCurrent();
        }
      }
      if (stopping) {
        syncCurrent();
      }
      applyTrim();
    } catch (const std::exception&) {
      // A failed write or sync leaves the file's state unknown, so nothing further counts as
      // durable.
      ok = false;
    }
    lock.lock();
    if (!ok) {
      failed_ = true;
      durable_.notify_all();
      return;
    }
    if (!bytes.empty()) {
      durable_up_to_ = next_offset;
    }
    durable_.notify_all();
    if (stopping && buffer_.empty()) {
      return;
    }
  }
}

// On its own thread, because a sync can take milliseconds and acknowledging under this policy only
// needs the data to have reached the OS.
void ShardWal::intervalSyncLoop() {
  std::unique_lock lock(mutex_);
  while (!stopping_) {
    durable_.wait_for(lock, config_.flush_interval, [this] { return stopping_; });
    if (stopping_) {
      return;
    }
    lock.unlock();
    int fd = -1;
    {
      const std::scoped_lock fd_lock(fd_mutex_);
      if (fd_ >= 0) {
        fd = dup(fd_);
      }
    }
    if (fd >= 0) {
      try {
        syncFile(fd);
      } catch (const std::exception&) {
        close(fd);
        lock.lock();
        failed_ = true;
        durable_.notify_all();
        return;
      }
      close(fd);
    }
    lock.lock();
  }
}

void ShardWal::writeOut(const std::string& bytes, std::uint64_t next_offset) {
  const char* data = bytes.data();
  std::size_t remaining = bytes.size();
  while (remaining > 0) {
    const ssize_t written = write(fd_, data, remaining);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throwErrno("write");
    }
    data += written;
    remaining -= static_cast<std::size_t>(written);
  }
  segment_size_ += bytes.size();
  if (segment_size_ >= config_.segment_bytes) {
    // Records stay whole because rolling only happens between whole buffers.
    syncCurrent();
    sealCurrent();
    openSegment(next_offset);
  }
}

void ShardWal::openSegment(std::uint64_t first_offset) {
  const fs::path path = shard_dir_ / segmentFileName(first_offset);
  {
    const std::scoped_lock lock(fd_mutex_);
    fd_ = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd_ < 0) {
      throwErrno("create segment");
    }
  }
  syncDirectory(shard_dir_);
  current_first_ = first_offset;
  segment_size_ = 0;
}

void ShardWal::sealCurrent() {
  const std::scoped_lock lock(fd_mutex_);
  close(fd_);
  fd_ = -1;
  sealed_.push_back(
      {.first_offset = current_first_, .path = shard_dir_ / segmentFileName(current_first_)});
}

void ShardWal::syncCurrent() { syncFile(fd_); }

void ShardWal::trimBelow(std::uint64_t offset) {
  {
    const std::scoped_lock lock(mutex_);
    if (offset <= trim_below_) {
      return;
    }
    trim_below_ = offset;
  }
  has_data_.notify_one();
}

void ShardWal::applyTrim() {
  std::uint64_t below = 0;
  {
    const std::scoped_lock lock(mutex_);
    below = trim_below_;
    trim_below_ = 0;
  }
  if (below == 0) {
    return;
  }
  // A sealed segment ends where the next one starts, so it can go if that is at or below the mark.
  while (!sealed_.empty()) {
    const std::uint64_t next_start = sealed_.size() > 1 ? sealed_[1].first_offset : current_first_;
    if (next_start > below) {
      break;
    }
    const Segment segment = sealed_.front();
    if (config_.archive) {
      config_.archive->put(segment.path, archiveName(segment.first_offset));
    }
    fs::remove(segment.path);
    sealed_.erase(sealed_.begin());
  }
}

std::string ShardWal::archiveName(std::uint64_t first_offset) const {
  return "shard-" + std::to_string(shard_) + "-" + segmentFileName(first_offset);
}

std::vector<Event> ShardWal::readArchived(std::uint64_t offset, std::size_t max_events,
                                          std::uint64_t& next_offset) const {
  std::vector<Event> events;
  next_offset = offset;
  if (!config_.archive) {
    return events;
  }
  const std::string prefix = "shard-" + std::to_string(shard_) + "-";
  for (const std::string& name : config_.archive->list(prefix)) {
    const std::uint64_t first = std::stoull(name.substr(prefix.size(), 20));
    if (first > next_offset || events.size() >= max_events) {
      if (first > next_offset) {
        break;
      }
      continue;
    }
    const std::string data = config_.archive->get(name);
    parseRecords(data, first, [&](std::uint64_t record_offset, Event event) {
      if (record_offset == next_offset && events.size() < max_events) {
        events.push_back(std::move(event));
        next_offset = record_offset + 1;
      }
    });
    if (events.size() >= max_events) {
      break;
    }
  }
  return events;
}

void ShardWal::saveGroupOffset(const std::string& group, std::uint64_t offset) {
  const fs::path destination = shard_dir_ / ("group-" + group + ".offset");
  const fs::path temporary = shard_dir_ / ("group-" + group + ".offset.tmp");
  {
    std::ofstream file(temporary, std::ios::trunc);
    file << offset;
  }
  const int fd = open(temporary.c_str(), O_RDONLY);
  if (fd < 0) {
    throwErrno("open group offset");
  }
  syncFile(fd);
  close(fd);
  fs::rename(temporary, destination);
  syncDirectory(shard_dir_);
}

std::uint64_t ShardWal::loadGroupOffset(const std::string& group) const {
  std::ifstream file(shard_dir_ / ("group-" + group + ".offset"));
  std::uint64_t offset = 0;
  file >> offset;
  return offset;
}

}  // namespace stream
