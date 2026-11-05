#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace perf {
namespace {

// Adds four rows of b, each scaled by one element of a row of a, into a row of out, so one pass
// over out_row does the work of four. Covers the given block of rows and columns of out.
void addScaledRows(const float* a, const float* b, float* out, std::size_t row_begin,
                   std::size_t row_end, std::size_t inner, std::size_t col_begin,
                   std::size_t cols) {
  for (std::size_t i = row_begin; i < row_end; ++i) {
    const float* a_row = a + (i * inner);
    float* out_row = out + (i * cols);
    std::size_t k = 0;
    for (; k + 4 <= inner; k += 4) {
      const float32x4_t scale = vld1q_f32(a_row + k);
      const float* b0 = b + (k * cols);
      const float* b1 = b0 + cols;
      const float* b2 = b1 + cols;
      const float* b3 = b2 + cols;
      std::size_t j = col_begin;
      for (; j + 4 <= cols; j += 4) {
        float32x4_t sum = vld1q_f32(out_row + j);
        sum = vfmaq_laneq_f32(sum, vld1q_f32(b0 + j), scale, 0);
        sum = vfmaq_laneq_f32(sum, vld1q_f32(b1 + j), scale, 1);
        sum = vfmaq_laneq_f32(sum, vld1q_f32(b2 + j), scale, 2);
        sum = vfmaq_laneq_f32(sum, vld1q_f32(b3 + j), scale, 3);
        vst1q_f32(out_row + j, sum);
      }
      for (; j < cols; ++j) {
        out_row[j] += (a_row[k] * b0[j]) + (a_row[k + 1] * b1[j]) + (a_row[k + 2] * b2[j]) +
                      (a_row[k + 3] * b3[j]);
      }
    }
    for (; k < inner; ++k) {
      const float32x4_t scale = vdupq_n_f32(a_row[k]);
      const float* b_row = b + (k * cols);
      std::size_t j = col_begin;
      for (; j + 4 <= cols; j += 4) {
        vst1q_f32(out_row + j, vfmaq_f32(vld1q_f32(out_row + j), vld1q_f32(b_row + j), scale));
      }
      for (; j < cols; ++j) {
        out_row[j] += a_row[k] * b_row[j];
      }
    }
  }
}

}  // namespace

void matmulNeon(const float* a, const float* b, float* out, std::size_t rows, std::size_t inner,
                std::size_t cols) {
  std::fill_n(out, rows * cols, 0.0F);
  addScaledRows(a, b, out, 0, rows, inner, 0, cols);
}

// Computes 4 rows x 12 columns of out at a time, with the running sums held in registers across
// the whole inner loop. Each step is 3 loads and 12 multiply-adds, with no load or store of out
// until the end. b is not packed or cache-blocked, which limits it on very wide matrices.
void matmulNeonMicro(const float* a, const float* b, float* out, std::size_t rows,
                     std::size_t inner, std::size_t cols) {
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kCols = 12;
  const std::size_t full_rows = rows - (rows % kRows);
  const std::size_t full_cols = cols - (cols % kCols);

  for (std::size_t i = 0; i < full_rows; i += kRows) {
    const float* a0 = a + (i * inner);
    const float* a1 = a0 + inner;
    const float* a2 = a1 + inner;
    const float* a3 = a2 + inner;
    for (std::size_t j = 0; j < full_cols; j += kCols) {
      float32x4_t c00 = vdupq_n_f32(0.0F);
      float32x4_t c01 = c00;
      float32x4_t c02 = c00;
      float32x4_t c10 = c00;
      float32x4_t c11 = c00;
      float32x4_t c12 = c00;
      float32x4_t c20 = c00;
      float32x4_t c21 = c00;
      float32x4_t c22 = c00;
      float32x4_t c30 = c00;
      float32x4_t c31 = c00;
      float32x4_t c32 = c00;
      const float* b_block = b + j;
      for (std::size_t k = 0; k < inner; ++k) {
        const float32x4_t b0 = vld1q_f32(b_block);
        const float32x4_t b1 = vld1q_f32(b_block + 4);
        const float32x4_t b2 = vld1q_f32(b_block + 8);
        b_block += cols;
        c00 = vfmaq_n_f32(c00, b0, a0[k]);
        c01 = vfmaq_n_f32(c01, b1, a0[k]);
        c02 = vfmaq_n_f32(c02, b2, a0[k]);
        c10 = vfmaq_n_f32(c10, b0, a1[k]);
        c11 = vfmaq_n_f32(c11, b1, a1[k]);
        c12 = vfmaq_n_f32(c12, b2, a1[k]);
        c20 = vfmaq_n_f32(c20, b0, a2[k]);
        c21 = vfmaq_n_f32(c21, b1, a2[k]);
        c22 = vfmaq_n_f32(c22, b2, a2[k]);
        c30 = vfmaq_n_f32(c30, b0, a3[k]);
        c31 = vfmaq_n_f32(c31, b1, a3[k]);
        c32 = vfmaq_n_f32(c32, b2, a3[k]);
      }
      float* o0 = out + (i * cols) + j;
      float* o1 = o0 + cols;
      float* o2 = o1 + cols;
      float* o3 = o2 + cols;
      vst1q_f32(o0, c00);
      vst1q_f32(o0 + 4, c01);
      vst1q_f32(o0 + 8, c02);
      vst1q_f32(o1, c10);
      vst1q_f32(o1 + 4, c11);
      vst1q_f32(o1 + 8, c12);
      vst1q_f32(o2, c20);
      vst1q_f32(o2 + 4, c21);
      vst1q_f32(o2 + 8, c22);
      vst1q_f32(o3, c30);
      vst1q_f32(o3 + 4, c31);
      vst1q_f32(o3 + 8, c32);
    }
    // Columns left over in these rows.
    if (full_cols < cols) {
      for (std::size_t r = i; r < i + kRows; ++r) {
        std::fill(out + (r * cols) + full_cols, out + ((r + 1) * cols), 0.0F);
      }
      addScaledRows(a, b, out, i, i + kRows, inner, full_cols, cols);
    }
  }
  // Rows left over.
  std::fill(out + (full_rows * cols), out + (rows * cols), 0.0F);
  addScaledRows(a, b, out, full_rows, rows, inner, 0, cols);
}

// The micro-kernel with b copied into tightly packed 12-column panels, and the work split into
// blocks that fit in cache. Reading b then walks memory in order instead of striding across whole
// rows, the design used by BLIS, Eigen, and similar libraries. Copying has a cost, so this pays off
// once matrices are big enough for each panel to be used many times.
void matmulNeonPacked(const float* a, const float* b, float* out, std::size_t rows,
                      std::size_t inner, std::size_t cols) {
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kCols = 12;
  constexpr std::size_t kInnerBlock = 256;  // A panel of this many rows fits in a 128 KB L1 cache.
  const std::size_t full_rows = rows - (rows % kRows);
  const std::size_t full_cols = cols - (cols % kCols);
  std::fill_n(out, rows * cols, 0.0F);

  thread_local std::vector<float> packed;
  for (std::size_t inner_start = 0; inner_start < inner; inner_start += kInnerBlock) {
    const std::size_t block = std::min(kInnerBlock, inner - inner_start);
    packed.resize((full_cols / kCols) * block * kCols);
    float* write = packed.data();
    for (std::size_t j = 0; j < full_cols; j += kCols) {
      for (std::size_t k = 0; k < block; ++k) {
        std::copy_n(b + ((inner_start + k) * cols) + j, kCols, write);
        write += kCols;
      }
    }
    for (std::size_t i = 0; i < full_rows; i += kRows) {
      const float* a0 = a + (i * inner) + inner_start;
      const float* a1 = a0 + inner;
      const float* a2 = a1 + inner;
      const float* a3 = a2 + inner;
      const float* panel = packed.data();
      for (std::size_t j = 0; j < full_cols; j += kCols) {
        float* o0 = out + (i * cols) + j;
        float* o1 = o0 + cols;
        float* o2 = o1 + cols;
        float* o3 = o2 + cols;
        float32x4_t c00 = vld1q_f32(o0);
        float32x4_t c01 = vld1q_f32(o0 + 4);
        float32x4_t c02 = vld1q_f32(o0 + 8);
        float32x4_t c10 = vld1q_f32(o1);
        float32x4_t c11 = vld1q_f32(o1 + 4);
        float32x4_t c12 = vld1q_f32(o1 + 8);
        float32x4_t c20 = vld1q_f32(o2);
        float32x4_t c21 = vld1q_f32(o2 + 4);
        float32x4_t c22 = vld1q_f32(o2 + 8);
        float32x4_t c30 = vld1q_f32(o3);
        float32x4_t c31 = vld1q_f32(o3 + 4);
        float32x4_t c32 = vld1q_f32(o3 + 8);
        for (std::size_t k = 0; k < block; ++k) {
          const float32x4_t b0 = vld1q_f32(panel);
          const float32x4_t b1 = vld1q_f32(panel + 4);
          const float32x4_t b2 = vld1q_f32(panel + 8);
          panel += kCols;
          c00 = vfmaq_n_f32(c00, b0, a0[k]);
          c01 = vfmaq_n_f32(c01, b1, a0[k]);
          c02 = vfmaq_n_f32(c02, b2, a0[k]);
          c10 = vfmaq_n_f32(c10, b0, a1[k]);
          c11 = vfmaq_n_f32(c11, b1, a1[k]);
          c12 = vfmaq_n_f32(c12, b2, a1[k]);
          c20 = vfmaq_n_f32(c20, b0, a2[k]);
          c21 = vfmaq_n_f32(c21, b1, a2[k]);
          c22 = vfmaq_n_f32(c22, b2, a2[k]);
          c30 = vfmaq_n_f32(c30, b0, a3[k]);
          c31 = vfmaq_n_f32(c31, b1, a3[k]);
          c32 = vfmaq_n_f32(c32, b2, a3[k]);
        }
        vst1q_f32(o0, c00);
        vst1q_f32(o0 + 4, c01);
        vst1q_f32(o0 + 8, c02);
        vst1q_f32(o1, c10);
        vst1q_f32(o1 + 4, c11);
        vst1q_f32(o1 + 8, c12);
        vst1q_f32(o2, c20);
        vst1q_f32(o2 + 4, c21);
        vst1q_f32(o2 + 8, c22);
        vst1q_f32(o3, c30);
        vst1q_f32(o3 + 4, c31);
        vst1q_f32(o3 + 8, c32);
      }
    }
  }
  // What the panels did not cover: leftover columns of the full rows, then leftover rows.
  if (full_cols < cols) {
    addScaledRows(a, b, out, 0, full_rows, inner, full_cols, cols);
  }
  addScaledRows(a, b, out, full_rows, rows, inner, 0, cols);
}

}  // namespace perf
