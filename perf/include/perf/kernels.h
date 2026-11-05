#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace perf {

// Computes out = a * b for row-major matrices: a is rows x inner, b is inner x cols, and out is
// rows x cols. `out` must not overlap the inputs.
using MatmulFn = void (*)(const float* a, const float* b, float* out, std::size_t rows,
                          std::size_t inner, std::size_t cols);

struct Kernel {
  std::string_view name;
  MatmulFn matmul;
};

// With compiler vectorization turned off, as the reference and worst case.
void matmulScalar(const float* a, const float* b, float* out, std::size_t rows, std::size_t inner,
                  std::size_t cols);
// The same loop, left to the compiler to vectorize.
void matmulAutoVec(const float* a, const float* b, float* out, std::size_t rows, std::size_t inner,
                   std::size_t cols);

// Every kernel this build and CPU can run, slowest first. The last is the one best() returns.
[[nodiscard]] std::vector<Kernel> availableKernels();
[[nodiscard]] Kernel best();

}  // namespace perf
