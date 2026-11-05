#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "security/api_keys.h"

namespace security {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

void processBlock(std::array<std::uint32_t, 8>& state, const unsigned char* block) {
  std::array<std::uint32_t, 64> w{};
  for (std::size_t i = 0; i < 16; ++i) {
    const unsigned char* p = block + (i * 4);
    w[i] = (std::uint32_t{p[0]} << 24U) | (std::uint32_t{p[1]} << 16U) |
           (std::uint32_t{p[2]} << 8U) | std::uint32_t{p[3]};
  }
  for (std::size_t i = 16; i < 64; ++i) {
    const std::uint32_t s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3U);
    const std::uint32_t s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10U);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::array<std::uint32_t, 8> v = state;
  for (std::size_t i = 0; i < 64; ++i) {
    const std::uint32_t s1 = std::rotr(v[4], 6) ^ std::rotr(v[4], 11) ^ std::rotr(v[4], 25);
    const std::uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
    const std::uint32_t t1 = v[7] + s1 + ch + kRoundConstants[i] + w[i];
    const std::uint32_t s0 = std::rotr(v[0], 2) ^ std::rotr(v[0], 13) ^ std::rotr(v[0], 22);
    const std::uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
    const std::uint32_t t2 = s0 + maj;
    v[7] = v[6];
    v[6] = v[5];
    v[5] = v[4];
    v[4] = v[3] + t1;
    v[3] = v[2];
    v[2] = v[1];
    v[1] = v[0];
    v[0] = t1 + t2;
  }
  for (std::size_t i = 0; i < 8; ++i) {
    state[i] += v[i];
  }
}

}  // namespace

std::string sha256Hex(std::string_view data) {
  std::array<std::uint32_t, 8> state = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const auto* bytes = reinterpret_cast<const unsigned char*>(data.data());
  std::size_t position = 0;
  for (; data.size() - position >= 64; position += 64) {
    processBlock(state, bytes + position);
  }
  // The final blocks: what is left, a 1 bit, zeros, and the length in bits.
  std::array<unsigned char, 128> tail{};
  const std::size_t left = data.size() - position;
  for (std::size_t i = 0; i < left; ++i) {
    tail[i] = bytes[position + i];
  }
  tail[left] = 0x80;
  const std::size_t tail_size = left < 56 ? 64 : 128;
  const std::uint64_t bit_length = std::uint64_t{data.size()} * 8;
  for (std::size_t i = 0; i < 8; ++i) {
    tail[tail_size - 1 - i] = static_cast<unsigned char>((bit_length >> (i * 8)) & 0xFFU);
  }
  processBlock(state, tail.data());
  if (tail_size == 128) {
    processBlock(state, tail.data() + 64);
  }

  constexpr std::string_view kHex = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (const std::uint32_t word : state) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      out += kHex[(word >> static_cast<unsigned>(shift)) & 0xFU];
    }
  }
  return out;
}

}  // namespace security
