#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace runtime {

// A loaded ONNX model with one float tensor input and one float tensor output.
// Failures to load or run the model throw std::runtime_error.
class OnnxModel {
 public:
  enum class Provider : std::uint8_t {
    kCpu,
    // Apple's Core ML, which can use the GPU and Neural Engine. Throws on other platforms.
    // Operations it cannot run fall back to the CPU, without any error.
    kCoreMl,
  };

  explicit OnnxModel(const std::filesystem::path& model_path, Provider provider = Provider::kCpu);
  ~OnnxModel();

  OnnxModel(const OnnxModel&) = delete;
  OnnxModel& operator=(const OnnxModel&) = delete;
  OnnxModel(OnnxModel&&) noexcept;
  OnnxModel& operator=(OnnxModel&&) noexcept;

  // Size of the model's last input dimension, i.e. the features per row of a batch.
  [[nodiscard]] std::int64_t inputFeatureCount() const;
  // The model's input dimensions after the first, which is the batch: {3, 224, 224} for an image
  // model, {64} for a flat one.
  [[nodiscard]] std::vector<std::int64_t> inputRowShape() const;

  // Runs one inference and returns the output tensor, flattened.
  [[nodiscard]] std::vector<float> run(std::span<const float> input,
                                       std::span<const std::int64_t> shape) const;

 private:
  struct Impl;  // Keeps ONNX Runtime headers out of this interface.
  std::unique_ptr<Impl> impl_;
};

}  // namespace runtime
