// Compares matrix kernels and a full MLP forward pass across our kernels, Eigen, and ONNX Runtime,
// and malloc against the arena. All single-threaded. Prints one JSON object of results.
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "perf/arena.h"
#include "perf/kernels.h"
#include "runtime/onnx_model.h"

namespace {

using Clock = std::chrono::steady_clock;
using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Keeps the optimizer from removing the work being timed.
void keep(const void* pointer) { asm volatile("" : : "r"(pointer) : "memory"); }

// Median over 7 rounds of at least 50 ms each, in nanoseconds per call.
double measureNs(const std::function<void()>& work) {
  constexpr int kRounds = 7;
  constexpr auto kMinRound = std::chrono::milliseconds(50);
  std::array<double, kRounds> rounds{};
  for (double& round : rounds) {
    std::uint64_t calls = 0;
    const Clock::time_point start = Clock::now();
    Clock::time_point now = start;
    while (now - start < kMinRound) {
      work();
      ++calls;
      now = Clock::now();
    }
    round =
        std::chrono::duration<double, std::nano>(now - start).count() / static_cast<double>(calls);
  }
  std::ranges::sort(rounds);
  return rounds[kRounds / 2];
}

std::vector<float> randomValues(std::size_t size, std::mt19937& rng) {
  std::normal_distribution<float> values(0.0F, 0.1F);
  std::vector<float> out(size);
  for (float& value : out) {
    value = values(rng);
  }
  return out;
}

// The same 64 -> 256 -> 256 -> 10 network as bench/models/mlp_float.onnx. The weights differ, but
// these operations take the same time whatever the values are, apart from denormals, which these
// do not produce.
struct Network {
  static constexpr std::array<std::size_t, 4> kLayers{64, 256, 256, 10};
  std::array<std::vector<float>, 3> weights;
  std::array<std::vector<float>, 3> biases;

  explicit Network(std::mt19937& rng) {
    for (std::size_t i = 0; i < 3; ++i) {
      weights[i] = randomValues(kLayers[i] * kLayers[i + 1], rng);
      biases[i] = randomValues(kLayers[i + 1], rng);
    }
  }
};

void addBiasAndRelu(std::vector<float>& values, const std::vector<float>& bias, std::size_t rows,
                    bool relu) {
  const std::size_t cols = bias.size();
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = values.data() + (r * cols);
    for (std::size_t c = 0; c < cols; ++c) {
      const float value = row[c] + bias[c];
      row[c] = relu ? std::max(value, 0.0F) : value;
    }
  }
}

void forwardOurs(const Network& net, perf::MatmulFn matmul, const std::vector<float>& input,
                 std::size_t rows, std::array<std::vector<float>, 3>& buffers) {
  const float* current = input.data();
  for (std::size_t i = 0; i < 3; ++i) {
    buffers[i].resize(rows * Network::kLayers[i + 1]);
    matmul(current, net.weights[i].data(), buffers[i].data(), rows, Network::kLayers[i],
           Network::kLayers[i + 1]);
    addBiasAndRelu(buffers[i], net.biases[i], rows, i < 2);
    current = buffers[i].data();
  }
}

EigenMatrix forwardEigen(const std::array<EigenMatrix, 3>& weights,
                         const std::array<Eigen::RowVectorXf, 3>& biases,
                         const EigenMatrix& input) {
  EigenMatrix current = input;
  for (std::size_t i = 0; i < 3; ++i) {
    EigenMatrix next = current * weights[i];
    next.rowwise() += biases[i];
    if (i < 2) {
      next = next.cwiseMax(0.0F);
    }
    current = std::move(next);
  }
  return current;
}

// Models the allocation pattern of one batch: many short-lived vectors, all freed together.
void allocationsMalloc(std::size_t count, std::size_t floats) {
  std::vector<std::unique_ptr<std::vector<float>>> rows;
  rows.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    rows.push_back(std::make_unique<std::vector<float>>(floats));
    keep(rows.back()->data());
  }
}

void allocationsArena(perf::Arena& arena, std::size_t count, std::size_t floats) {
  arena.reset();
  using ArenaVector = std::vector<float, perf::ArenaAllocator<float>>;
  for (std::size_t i = 0; i < count; ++i) {
    auto* row =
        static_cast<ArenaVector*>(arena.allocate(sizeof(ArenaVector), alignof(ArenaVector)));
    ::new (row) ArenaVector(floats, perf::ArenaAllocator<float>(arena));
    keep(row->data());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::span args(argv, static_cast<std::size_t>(argc));
    if (args.size() != 2) {
      std::cerr << "usage: kernels_bench MLP_FLOAT_ONNX\n";
      return EXIT_FAILURE;
    }
    // Fixed seed: every run must measure the same data.
    // NOLINTNEXTLINE(bugprone-random-generator-seed)
    std::mt19937 rng(42);
    nlohmann::json results;

    // Matrix multiplication alone. FLOPs are 2 per element of rows * inner * cols.
    constexpr std::size_t kSize = 256;
    const std::vector<float> a = randomValues(kSize * kSize, rng);
    const std::vector<float> b = randomValues(kSize * kSize, rng);
    std::vector<float> out(kSize * kSize);
    const double flops = 2.0 * kSize * kSize * kSize;
    const auto gflops = [flops](double ns) { return flops / ns; };
    nlohmann::json matmul = nlohmann::json::array();
    for (const perf::Kernel& kernel : perf::availableKernels()) {
      const double ns = measureNs([&] {
        kernel.matmul(a.data(), b.data(), out.data(), kSize, kSize, kSize);
        keep(out.data());
      });
      matmul.push_back({{"impl", std::string(kernel.name)}, {"ns", ns}, {"gflops", gflops(ns)}});
    }
    const Eigen::Map<const EigenMatrix> eigen_a(a.data(), kSize, kSize);
    const Eigen::Map<const EigenMatrix> eigen_b(b.data(), kSize, kSize);
    EigenMatrix eigen_out(kSize, kSize);
    const double eigen_ns = measureNs([&] {
      eigen_out.noalias() = eigen_a * eigen_b;
      keep(eigen_out.data());
    });
    matmul.push_back({{"impl", "eigen"}, {"ns", eigen_ns}, {"gflops", gflops(eigen_ns)}});
    results["matmul_256"] = matmul;

    // A full forward pass at several batch sizes.
    const Network net(rng);
    std::array<EigenMatrix, 3> eigen_weights;
    std::array<Eigen::RowVectorXf, 3> eigen_biases;
    for (std::size_t i = 0; i < 3; ++i) {
      const auto in = static_cast<Eigen::Index>(Network::kLayers[i]);
      const auto next = static_cast<Eigen::Index>(Network::kLayers[i + 1]);
      eigen_weights[i] = Eigen::Map<const EigenMatrix>(net.weights[i].data(), in, next);
      eigen_biases[i] = Eigen::Map<const Eigen::RowVectorXf>(net.biases[i].data(), next);
    }
    const runtime::OnnxModel onnx(args[1]);
    std::array<std::vector<float>, 3> buffers;

    nlohmann::json forward = nlohmann::json::array();
    for (const std::size_t rows : {1U, 8U, 64U, 256U}) {
      const std::vector<float> input = randomValues(rows * 64, rng);
      const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(rows), 64};
      const EigenMatrix eigen_input =
          Eigen::Map<const EigenMatrix>(input.data(), static_cast<Eigen::Index>(rows), 64);
      nlohmann::json row;
      row["batch"] = rows;
      for (const perf::Kernel& kernel : perf::availableKernels()) {
        if (kernel.name == "scalar") {
          continue;
        }
        row["ours_" + std::string(kernel.name) + "_ns"] = measureNs([&] {
          forwardOurs(net, kernel.matmul, input, rows, buffers);
          keep(buffers[2].data());
        });
      }
      row["eigen_ns"] = measureNs([&] {
        const EigenMatrix output = forwardEigen(eigen_weights, eigen_biases, eigen_input);
        keep(output.data());
      });
      row["onnxruntime_ns"] = measureNs([&] {
        const std::vector<float> output = onnx.run(input, shape);
        keep(output.data());
      });
      forward.push_back(row);
    }
    results["forward"] = forward;

    // Allocation: 64 vectors of 64 floats, as one 64-row batch might hold.
    perf::Arena arena(std::size_t{1024} * 1024);
    results["allocation"] = {
        {"malloc_ns", measureNs([] { allocationsMalloc(64, 64); })},
        {"arena_ns", measureNs([&] { allocationsArena(arena, 64, 64); })},
    };

    std::cout << results.dump() << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
