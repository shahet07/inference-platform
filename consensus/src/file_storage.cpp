#include "consensus/file_storage.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace consensus {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kHeaderBytes = 8;  // Size and CRC.
constexpr std::size_t kTermBytes = 8;
constexpr std::size_t kMaxRecordBytes = std::size_t{64} * 1024 * 1024;

[[noreturn]] void throwErrno(const std::string& what) {
  throw std::system_error(errno, std::generic_category(), what);
}

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
  const auto bytes = std::bit_cast<std::array<char, sizeof(T)>>(value);
  out.append(bytes.data(), bytes.size());
}

template <typename T>
T getValue(std::string_view data, std::size_t position) {
  std::array<char, sizeof(T)> bytes{};
  std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(position), sizeof(T), bytes.begin());
  return std::bit_cast<T>(bytes);
}

void syncFd(int fd) {
#ifdef __APPLE__
  if (fcntl(fd, F_FULLFSYNC) == 0) {
    return;
  }
#endif
  if (fsync(fd) != 0) {
    throwErrno("fsync");
  }
}

void syncPath(const fs::path& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throwErrno("open " + path.string());
  }
  syncFd(fd);
  close(fd);
}

}  // namespace

std::size_t FileStorage::parseLog(std::string_view data, std::vector<LogEntry>& entries,
                                  std::vector<std::uint64_t>& ends) {
  std::size_t position = 0;
  while (data.size() - position >= kHeaderBytes) {
    const auto size = getValue<std::uint32_t>(data, position);
    const auto crc = getValue<std::uint32_t>(data, position + 4);
    if (size < kTermBytes || size > kMaxRecordBytes ||
        size > data.size() - position - kHeaderBytes) {
      break;
    }
    const std::string_view record = data.substr(position + kHeaderBytes, size);
    if (crc32(record) != crc) {
      break;
    }
    entries.push_back({.term = getValue<std::uint64_t>(record, 0),
                       .command = std::string(record.substr(kTermBytes))});
    position += kHeaderBytes + size;
    ends.push_back(position);
  }
  return position;
}

FileStorage::FileStorage(fs::path dir, bool sync) : dir_(std::move(dir)), sync_(sync) {
  fs::create_directories(dir_);
  const fs::path state = dir_ / "state";
  if (fs::exists(state)) {
    std::ifstream in(state);
    if (!(in >> term_ >> voted_for_)) {
      throw std::runtime_error("corrupt raft state: " + state.string());
    }
  }
  const fs::path log = dir_ / "log";
  std::string data;
  if (fs::exists(log)) {
    std::ifstream in(log, std::ios::binary);
    std::ostringstream contents;
    contents << in.rdbuf();
    data = contents.str();
  }
  const std::size_t valid = parseLog(data, entries_, ends_);
  log_fd_ = open(log.c_str(), O_WRONLY | O_CREAT, 0644);
  if (log_fd_ < 0) {
    throwErrno("open raft log");
  }
  if (valid != data.size() && ftruncate(log_fd_, static_cast<off_t>(valid)) != 0) {
    throwErrno("truncate raft log");
  }
  if (lseek(log_fd_, 0, SEEK_END) < 0) {
    throwErrno("seek raft log");
  }
}

FileStorage::~FileStorage() {
  if (log_fd_ >= 0) {
    close(log_fd_);
  }
}

void FileStorage::saveTermAndVote(Term term, NodeId voted_for) {
  term_ = term;
  voted_for_ = voted_for;
  const fs::path temporary = dir_ / "state.tmp";
  {
    std::ofstream out(temporary, std::ios::trunc);
    out << term << ' ' << voted_for << '\n';
    if (!out) {
      throw std::runtime_error("cannot write raft state");
    }
  }
  if (sync_) {
    syncPath(temporary);
  }
  fs::rename(temporary, dir_ / "state");
  if (sync_) {
    syncPath(dir_);
  }
}

void FileStorage::replaceSuffix(Index keep_up_to, const std::vector<LogEntry>& entries) {
  if (keep_up_to < entries_.size()) {
    const std::uint64_t end = keep_up_to == 0 ? 0 : ends_[keep_up_to - 1];
    if (ftruncate(log_fd_, static_cast<off_t>(end)) != 0 ||
        lseek(log_fd_, static_cast<off_t>(end), SEEK_SET) < 0) {
      throwErrno("truncate raft log");
    }
    entries_.resize(keep_up_to);
    ends_.resize(keep_up_to);
  }
  std::uint64_t position = ends_.empty() ? 0 : ends_.back();
  std::string bytes;
  for (const LogEntry& entry : entries) {
    std::string record;
    record.reserve(kTermBytes + entry.command.size());
    putValue<std::uint64_t>(record, entry.term);
    record += entry.command;
    putValue<std::uint32_t>(bytes, static_cast<std::uint32_t>(record.size()));
    putValue<std::uint32_t>(bytes, crc32(record));
    bytes += record;
    position += kHeaderBytes + record.size();
    entries_.push_back(entry);
    ends_.push_back(position);
  }
  const char* data = bytes.data();
  std::size_t remaining = bytes.size();
  while (remaining > 0) {
    const ssize_t written = write(log_fd_, data, remaining);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throwErrno("write raft log");
    }
    data += written;
    remaining -= static_cast<std::size_t>(written);
  }
  if (sync_) {
    syncFd(log_fd_);
  }
}

}  // namespace consensus
