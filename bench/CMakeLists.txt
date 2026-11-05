add_executable(load_gen load_gen.cpp)
target_link_libraries(load_gen PRIVATE httplib::httplib nlohmann_json::nlohmann_json)

add_executable(stream_bench stream_bench.cpp)
target_link_libraries(stream_bench PRIVATE stream nlohmann_json::nlohmann_json)

add_executable(kernels_bench kernels_bench.cpp)
target_link_libraries(kernels_bench PRIVATE perf runtime eigen_headers nlohmann_json::nlohmann_json)
target_compile_options(kernels_bench PRIVATE -O3)

add_executable(model_bench model_bench.cpp)
target_link_libraries(model_bench PRIVATE runtime nlohmann_json::nlohmann_json)
