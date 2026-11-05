#include "runtime/onnx_model.h"

#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace runtime {
namespace {

Ort::SessionOptions makeSessionOptions(OnnxModel::Provider provider) {
  Ort::SessionOptions options;
  if (provider == OnnxModel::Provider::kCoreMl) {
#ifdef __APPLE__
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(options, 0));
#else
    throw std::runtime_error("Core ML is only available on Apple platforms");
#endif
  }
  // One thread per inference: concurrency will come from the server's own thread pool,
  // and an ONNX Runtime pool on top of it would oversubscribe the cores.
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  return options;
}

// ONNX Runtime only checks that the buffer is large enough for the shape, so a shape that is too
// small would silently ignore part of the input. Dimensions will come from clients, so negative
// values and overflow are checked too.
void validateShape(std::span<const std::int64_t> shape, std::size_t element_count) {
  std::int64_t expected = 1;
  for (const std::int64_t dim : shape) {
    if (dim < 0 || (dim > 0 && expected > std::numeric_limits<std::int64_t>::max() / dim)) {
      throw std::runtime_error("invalid tensor shape");
    }
    expected *= dim;
  }
  if (std::cmp_not_equal(expected, element_count)) {
    throw std::runtime_error("tensor shape does not match input size");
  }
}

}  // namespace

struct OnnxModel::Impl {
  Impl(const std::filesystem::path& model_path, Provider provider)
      : env(ORT_LOGGING_LEVEL_WARNING, "runtime"),
        session(env, model_path.c_str(), makeSessionOptions(provider)) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
      throw std::runtime_error("expected a model with one input and one output: " +
                               model_path.string());
    }
    const Ort::AllocatorWithDefaultOptions allocator;
    input_name = session.GetInputNameAllocated(0, allocator).get();
    output_name = session.GetOutputNameAllocated(0, allocator).get();
  }

  Ort::Env env;
  Ort::Session session;
  std::string input_name;
  std::string output_name;
};

OnnxModel::OnnxModel(const std::filesystem::path& model_path, Provider provider) {
  try {
    impl_ = std::make_unique<Impl>(model_path, provider);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("failed to load " + model_path.string() + ": " + e.what());
  }
}

OnnxModel::~OnnxModel() = default;
OnnxModel::OnnxModel(OnnxModel&&) noexcept = default;
OnnxModel& OnnxModel::operator=(OnnxModel&&) noexcept = default;

std::int64_t OnnxModel::inputFeatureCount() const {
  const std::vector<std::int64_t> shape =
      impl_->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  return shape.empty() ? 0 : shape.back();
}

std::vector<std::int64_t> OnnxModel::inputRowShape() const {
  const std::vector<std::int64_t> shape =
      impl_->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  return shape.empty() ? shape : std::vector<std::int64_t>(shape.begin() + 1, shape.end());
}

std::vector<float> OnnxModel::run(std::span<const float> input,
                                  std::span<const std::int64_t> shape) const {
  validateShape(shape, input.size());
  try {
    const Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // CreateTensor wraps the caller's buffer without copying and needs a non-const pointer,
    // but ONNX Runtime does not write to input tensors.
    const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input.data()), input.size(), shape.data(), shape.size());

    const std::array<const char*, 1> input_names{impl_->input_name.c_str()};
    const std::array<const char*, 1> output_names{impl_->output_name.c_str()};
    std::vector<Ort::Value> outputs =
        impl_->session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                           output_names.data(), output_names.size());

    const Ort::TensorTypeAndShapeInfo info = outputs.front().GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("expected a float output tensor");
    }
    const auto* data = outputs.front().GetTensorData<float>();
    return {data, data + info.GetElementCount()};
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("inference failed: ") + e.what());
  }
}

}  // namespace runtime
