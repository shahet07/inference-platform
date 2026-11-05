#include "perf/kernels.h"

#include <algorithm>

namespace perf {

#ifdef IP_HAS_NEON_KERNEL
void matmulNeon(const float* a, const float* b, float* out, std::size_t rows, std::size_t inner,
                std::size_t cols);
void matmulNeonMicro(const float* a, const float* b, float* out, std::size_t rows,
                     std::size_t inner, std::size_t cols);
#endif
#ifdef IP_HAS_AVX2_KERNEL
void matmulAvx2(const float* a, const float* b, float* out, std::size_t rows, std::size_t inner,
                std::size_t cols);
#endif

void matmulScalar(const float* a, const float* b, float* out, std::size_t rows, std::size_t inner,
                  std::size_t cols) {
  std::fill_n(out, rows * cols, 0.0F);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t k = 0; k < inner; ++k) {
      const float scale = a[(i * inner) + k];
      const float* b_row = b + (k * cols);
      float* out_row = out + (i * cols);
#pragma clang loop vectorize(disable) interleave(disable)
      for (std::size_t j = 0; j < cols; ++j) {
        out_row[j] += scale * b_row[j];
      }
    }
  }
}

void matmulAutoVec(const float* a, const float* b, float* out, std::size_t rows, std::size_t inner,
                   std::size_t cols) {
  std::fill_n(out, rows * cols, 0.0F);
  for (std::size_t i = 0; i < rows; ++i) {
    float* __restrict__ out_row = out + (i * cols);
    for (std::size_t k = 0; k < inner; ++k) {
      const float scale = a[(i * inner) + k];
      const float* __restrict__ b_row = b + (k * cols);
      for (std::size_t j = 0; j < cols; ++j) {
        out_row[j] += scale * b_row[j];
      }
    }
  }
}

std::vector<Kernel> availableKernels() {
  std::vector<Kernel> kernels{{.name = "scalar", .matmul = matmulScalar},
                              {.name = "autovec", .matmul = matmulAutoVec}};
#ifdef IP_HAS_NEON_KERNEL
  kernels.push_back({.name = "neon", .matmul = matmulNeon});
  kernels.push_back({.name = "neon_micro", .matmul = matmulNeonMicro});
#endif
#ifdef IP_HAS_AVX2_KERNEL
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
    kernels.push_back({.name = "avx2", .matmul = matmulAvx2});
  }
#endif
  return kernels;
}

Kernel best() { return availableKernels().back(); }

}  // namespace perf
