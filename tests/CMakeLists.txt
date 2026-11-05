include(GoogleTest)

add_executable(runtime_tests runtime/onnx_model_test.cpp)
target_link_libraries(runtime_tests PRIVATE runtime GTest::gtest_main GTest::gmock)
target_compile_definitions(runtime_tests PRIVATE
  IP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
gtest_discover_tests(runtime_tests)

add_executable(obs_tests obs/metrics_test.cpp obs/log_test.cpp obs/trace_test.cpp)
target_link_libraries(obs_tests PRIVATE obs GTest::gtest_main GTest::gmock nlohmann_json::nlohmann_json)
gtest_discover_tests(obs_tests)

add_executable(server_tests
  server/bounded_queue_test.cpp
  server/batching_engine_test.cpp
  server/http_server_test.cpp)
target_link_libraries(server_tests PRIVATE
  server GTest::gtest_main GTest::gmock httplib::httplib nlohmann_json::nlohmann_json)
gtest_discover_tests(server_tests)

add_executable(stream_tests
  stream/sharded_log_test.cpp
  stream/consumer_group_test.cpp
  stream/window_test.cpp
  stream/processor_test.cpp
  stream/wal_test.cpp)
target_link_libraries(stream_tests PRIVATE stream GTest::gtest_main GTest::gmock)
gtest_discover_tests(stream_tests)

add_executable(cluster_tests
  cluster/hash_ring_test.cpp
  cluster/health_monitor_test.cpp
  cluster/model_registry_test.cpp)
target_link_libraries(cluster_tests PRIVATE cluster GTest::gtest_main GTest::gmock)
gtest_discover_tests(cluster_tests)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME integration.cluster
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/cluster_test.py
    $<TARGET_FILE:inference_server> $<TARGET_FILE:inference_router>
    ${CMAKE_SOURCE_DIR}/bench/models/mlp_quant.onnx 19100)
set_tests_properties(integration.cluster PROPERTIES TIMEOUT 120)

add_executable(perf_tests perf/arena_test.cpp perf/kernels_test.cpp)
target_link_libraries(perf_tests PRIVATE perf GTest::gtest_main GTest::gmock)
gtest_discover_tests(perf_tests)

add_test(NAME integration.observability
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/observability_test.py
    $<TARGET_FILE:inference_server> $<TARGET_FILE:inference_router>
    ${CMAKE_SOURCE_DIR}/bench/models/mlp_quant.onnx 19200)
set_tests_properties(integration.observability PROPERTIES TIMEOUT 120)

add_executable(wal_crash_writer integration/wal_crash_writer.cpp)
target_link_libraries(wal_crash_writer PRIVATE stream)
add_test(NAME integration.wal_crash
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/wal_crash_test.py
    $<TARGET_FILE:wal_crash_writer>)
set_tests_properties(integration.wal_crash PROPERTIES TIMEOUT 180)
