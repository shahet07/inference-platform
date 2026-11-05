#include <immintrin.h>

#include <algorithm>
#include <cstddef>

namespace perf {

// The AVX2 version of matmulNeon: four scaled rows of b per pass over out_row, 8 lanes at a time.
void matmulAvx2(const float* a, const float* b, float* out, std::size_t rows, std::size_t inner,
                std::size_t cols) {
  std::fill_n(out, rows * cols, 0.0F);
  for (std::size_t i = 0; i < rows; ++i) {
    const float* a_row = a + (i * inner);
    float* out_row = out + (i * cols);
    std::size_t k = 0;
    for (; k + 4 <= inner; k += 4) {
      const __m256 s0 = _mm256_set1_ps(a_row[k]);
      const __m256 s1 = _mm256_set1_ps(a_row[k + 1]);
      const __m256 s2 = _mm256_set1_ps(a_row[k + 2]);
      const __m256 s3 = _mm256_set1_ps(a_row[k + 3]);
      const float* b0 = b + (k * cols);
      const float* b1 = b0 + cols;
      const float* b2 = b1 + cols;
      const float* b3 = b2 + cols;
      std::size_t j = 0;
      for (; j + 8 <= cols; j += 8) {
        __m256 sum = _mm256_loadu_ps(out_row + j);
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(b0 + j), s0, sum);
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(b1 + j), s1, sum);
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(b2 + j), s2, sum);
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(b3 + j), s3, sum);
        _mm256_storeu_ps(out_row + j, sum);
      }
      for (; j < cols; ++j) {
        out_row[j] += (a_row[k] * b0[j]) + (a_row[k + 1] * b1[j]) + (a_row[k + 2] * b2[j]) +
                      (a_row[k + 3] * b3[j]);
      }
    }
    for (; k < inner; ++k) {
      const __m256 scale = _mm256_set1_ps(a_row[k]);
      const float* b_row = b + (k * cols);
      std::size_t j = 0;
      for (; j + 8 <= cols; j += 8) {
        _mm256_storeu_ps(out_row + j, _mm256_fmadd_ps(_mm256_loadu_ps(b_row + j), scale,
                                                      _mm256_loadu_ps(out_row + j)));
      }
      for (; j < cols; ++j) {
        out_row[j] += a_row[k] * b_row[j];
      }
    }
  }
}

}  // namespace perf
