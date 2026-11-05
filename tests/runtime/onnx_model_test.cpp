#include "runtime/onnx_model.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

std::filesystem::path linearModelPath() {
  return std::filesystem::path(IP_TEST_DATA_DIR) / "linear.onnx";
}

TEST(OnnxModelTest, RunsInferenceOnLinearModel) {
  const runtime::OnnxModel model(linearModelPath());
  const std::array<float, 3> input{1.0F, 2.0F, 3.0F};
  const std::array<std::int64_t, 2> shape{3, 1};

  const std::vector<float> output = model.run(input, shape);

  // linear.onnx computes y = 2x + 1.
  EXPECT_THAT(output, testing::ElementsAre(3.0F, 5.0F, 7.0F));
}

#ifdef __APPLE__
TEST(OnnxModelTest, CoreMlGivesTheSameAnswers) {
  const runtime::OnnxModel model(linearModelPath(), runtime::OnnxModel::Provider::kCoreMl);
  const std::array<float, 3> input{1.0F, 2.0F, 3.0F};
  const std::array<std::int64_t, 2> shape{3, 1};
  EXPECT_THAT(model.run(input, shape), testing::ElementsAre(3.0F, 5.0F, 7.0F));
}
#else
TEST(OnnxModelTest, CoreMlIsRefusedElsewhere) {
  EXPECT_THROW(runtime::OnnxModel(linearModelPath(), runtime::OnnxModel::Provider::kCoreMl),
               std::runtime_error);
}
#endif

TEST(OnnxModelTest, ThrowsWhenModelFileIsMissing) {
  EXPECT_THROW(runtime::OnnxModel("does_not_exist.onnx"), std::runtime_error);
}

TEST(OnnxModelTest, ThrowsWhenShapeIsSmallerThanInput) {
  const runtime::OnnxModel model(linearModelPath());
  const std::array<float, 3> input{1.0F, 2.0F, 3.0F};
  const std::array<std::int64_t, 2> shape{2, 1};

  EXPECT_THROW((void)model.run(input, shape), std::runtime_error);
}

TEST(OnnxModelTest, ThrowsWhenShapeIsLargerThanInput) {
  const runtime::OnnxModel model(linearModelPath());
  const std::array<float, 3> input{1.0F, 2.0F, 3.0F};
  const std::array<std::int64_t, 2> shape{4, 1};

  EXPECT_THROW((void)model.run(input, shape), std::runtime_error);
}

TEST(OnnxModelTest, ThrowsWhenDimensionsAreNegative) {
  const runtime::OnnxModel model(linearModelPath());
  const std::array<float, 3> input{1.0F, 2.0F, 3.0F};
  const std::array<std::int64_t, 2> shape{-3, -1};  // The product matches the input size.

  EXPECT_THROW((void)model.run(input, shape), std::runtime_error);
}

TEST(OnnxModelTest, ThrowsWhenShapeProductOverflows) {
  const runtime::OnnxModel model(linearModelPath());
  const std::array<std::int64_t, 2> shape{4'294'967'296, 4'294'967'296};  // 2^32 each.

  // The product wraps to 0 in 64 bits, which would match an empty input.
  EXPECT_THROW((void)model.run({}, shape), std::runtime_error);
}

}  // namespace
