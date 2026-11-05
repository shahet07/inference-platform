// Runs one model in a tight loop with no networking, and prints inferences per second as JSON.
// Running several copies at once shows how much inference slows down from sharing the hardware.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/onnx_model.h"

int main(int argc, char** argv) {
  try {
    const std::span args(argv, static_cast<std::size_t>(argc));
    if (args.size() < 3 || args.size() > 5) {
      std::cerr << "usage: model_bench MODEL SECONDS [cpu|coreml] [BATCH]\n";
      return EXIT_FAILURE;
    }
    const bool coreml = args.size() > 3 && std::string(args[3]) == "coreml";
    const runtime::OnnxModel model(args[1], coreml ? runtime::OnnxModel::Provider::kCoreMl
                                                   : runtime::OnnxModel::Provider::kCpu);
    const std::int64_t batch = args.size() > 4 ? std::stoll(args[4]) : 1;
    std::vector<std::int64_t> shape{batch};
    std::int64_t features = batch;
    for (const std::int64_t dim : model.inputRowShape()) {
      shape.push_back(dim);
      features *= dim;
    }
    const std::vector<float> input(static_cast<std::size_t>(features), 0.5F);
    const std::chrono::duration<double> run_for(std::stod(args[2]));

    std::uint64_t count = 0;
    const auto start = std::chrono::steady_clock::now();
    auto now = start;
    while (now - start < run_for) {
      const std::vector<float> output = model.run(input, shape);
      if (output.empty()) {
        throw std::runtime_error("model returned no output");
      }
      ++count;
      now = std::chrono::steady_clock::now();
    }
    const double elapsed = std::chrono::duration<double>(now - start).count();
    std::cout << nlohmann::json{{"inferences_per_sec",
                                 static_cast<double>(count) * static_cast<double>(batch) / elapsed}}
                     .dump()
              << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
