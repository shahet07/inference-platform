#include "perf/kernels.h"

#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

namespace {

using perf::Kernel;

// A deliberately plain definition of matrix multiplication to compare against.
std::vector<float> reference(const std::vector<float>& a, const std::vector<float>& b,
                             std::size_t rows, std::size_t inner, std::size_t cols) {
  std::vector<float> out(rows * cols);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < inner; ++k) {
        sum += static_cast<double>(a[(i * inner) + k]) * static_cast<double>(b[(k * cols) + j]);
      }
      out[(i * cols) + j] = static_cast<float>(sum);
    }
  }
  return out;
}

std::vector<float> randomMatrix(std::size_t size, std::mt19937& rng) {
  std::uniform_real_distribution<float> values(-1.0F, 1.0F);
  std::vector<float> matrix(size);
  for (float& value : matrix) {
    value = values(rng);
  }
  return matrix;
}

TEST(KernelsTest, EveryKernelMatchesReferenceAtEveryRemainderSize) {
  // Fixed seed, so that a failure can be reproduced.
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(7);
  for (const Kernel& kernel : perf::availableKernels()) {
    // Sizes around the lane and unroll widths reach every cleanup loop.
    for (const std::size_t rows : {1U, 2U, 3U, 4U, 5U, 8U, 9U}) {
      for (const std::size_t inner : {1U, 3U, 4U, 5U, 8U, 9U, 17U}) {
        for (const std::size_t cols : {1U, 3U, 4U, 7U, 8U, 9U, 12U, 13U, 15U, 16U, 24U, 25U, 33U}) {
          const auto a = randomMatrix(rows * inner, rng);
          const auto b = randomMatrix(inner * cols, rng);
          const auto expected = reference(a, b, rows, inner, cols);
          std::vector<float> out(rows * cols, 99.0F);  // Results must not depend on old contents.
          kernel.matmul(a.data(), b.data(), out.data(), rows, inner, cols);
          for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_NEAR(out[i], expected[i], 1e-4F)
                << kernel.name << ' ' << rows << 'x' << inner << 'x' << cols << " at " << i;
          }
        }
      }
    }
  }
}

TEST(KernelsTest, BestIsTheLastAvailableKernel) {
  EXPECT_EQ(perf::best().name, perf::availableKernels().back().name);
}

TEST(KernelsTest, HandwrittenKernelIsAvailableOnThisArchitecture) {
#if defined(__aarch64__) || defined(__x86_64__)
  const std::string name(perf::best().name);
#if defined(__aarch64__)
  EXPECT_EQ(name, "neon_micro");
#else
  // AVX2 depends on the CPU, so either result is valid there.
  EXPECT_TRUE(name == "avx2" || name == "autovec");
#endif
#endif
}

}  // namespace
