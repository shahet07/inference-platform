add_executable(load_gen load_gen.cpp)
target_link_libraries(load_gen PRIVATE obs httplib::httplib nlohmann_json::nlohmann_json)

add_executable(stream_bench stream_bench.cpp)
target_link_libraries(stream_bench PRIVATE stream nlohmann_json::nlohmann_json)

add_executable(kernels_bench kernels_bench.cpp)
target_link_libraries(kernels_bench PRIVATE perf runtime eigen_headers nlohmann_json::nlohmann_json)
target_compile_options(kernels_bench PRIVATE -O3)

add_executable(model_bench model_bench.cpp)
target_link_libraries(model_bench PRIVATE runtime nlohmann_json::nlohmann_json)

add_executable(wal_bench wal_bench.cpp)
target_link_libraries(wal_bench PRIVATE stream nlohmann_json::nlohmann_json)

add_executable(stream_ops_bench stream_ops_bench.cpp)
target_link_libraries(stream_ops_bench PRIVATE stream nlohmann_json::nlohmann_json)

add_executable(queue_bench queue_bench.cpp)
target_link_libraries(queue_bench PRIVATE server nlohmann_json::nlohmann_json)
