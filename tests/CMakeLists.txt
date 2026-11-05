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
  server/http_server_test.cpp
  server/model_manager_test.cpp
  server/mpmc_queue_test.cpp)
target_link_libraries(server_tests PRIVATE
  server GTest::gtest_main GTest::gmock httplib::httplib nlohmann_json::nlohmann_json)
gtest_discover_tests(server_tests)

add_executable(stream_tests
  stream/sharded_log_test.cpp
  stream/consumer_group_test.cpp
  stream/window_test.cpp
  stream/processor_test.cpp
  stream/wal_test.cpp
  stream/sliding_window_test.cpp
  stream/interval_join_test.cpp
  stream/exactly_once_test.cpp)
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

add_test(NAME integration.hot_swap
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/hot_swap_test.py
    $<TARGET_FILE:inference_server> ${CMAKE_SOURCE_DIR}/bench/models 19300)
set_tests_properties(integration.hot_swap PROPERTIES TIMEOUT 180)

add_executable(exactly_once_worker integration/exactly_once_worker.cpp)
target_link_libraries(exactly_once_worker PRIVATE stream)
add_test(NAME integration.exactly_once
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/exactly_once_test.py
    $<TARGET_FILE:exactly_once_worker>)
set_tests_properties(integration.exactly_once PROPERTIES TIMEOUT 240)

add_test(NAME integration.chaos
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/chaos_test.py
    $<TARGET_FILE:inference_server> $<TARGET_FILE:inference_router>
    ${CMAKE_SOURCE_DIR}/bench/models/mlp_quant.onnx 19400)
set_tests_properties(integration.chaos PROPERTIES TIMEOUT 240)

add_test(NAME integration.autoscale
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/autoscale_test.py
    $<TARGET_FILE:inference_server> $<TARGET_FILE:inference_router> $<TARGET_FILE:load_gen>
    ${CMAKE_SOURCE_DIR}/bench/models/mlp_heavy_quant.onnx 19500)
set_tests_properties(integration.autoscale PROPERTIES TIMEOUT 240)

add_executable(consensus_tests consensus/raft_test.cpp consensus/file_storage_test.cpp)
target_link_libraries(consensus_tests PRIVATE consensus GTest::gtest_main)
gtest_discover_tests(consensus_tests PROPERTIES TIMEOUT 200)

add_test(NAME integration.raft
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/raft_test.py
    $<TARGET_FILE:raft_node> 19700)
set_tests_properties(integration.raft PROPERTIES TIMEOUT 240)

add_executable(net_tests net/http_parser_test.cpp net/event_server_test.cpp)
target_link_libraries(net_tests PRIVATE net GTest::gtest_main)
gtest_discover_tests(net_tests)

add_test(NAME integration.event_frontend
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/event_frontend_test.py
    $<TARGET_FILE:inference_server> $<TARGET_FILE:load_gen>
    ${CMAKE_SOURCE_DIR}/bench/models/mlp_quant.onnx 19800)
set_tests_properties(integration.event_frontend PROPERTIES TIMEOUT 120)

add_executable(security_tests security/api_keys_test.cpp)
target_link_libraries(security_tests PRIVATE security GTest::gtest_main)
gtest_discover_tests(security_tests)

add_test(NAME integration.security
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/security_test.py
    $<TARGET_FILE:inference_server> $<TARGET_FILE:inference_router>
    ${CMAKE_SOURCE_DIR}/bench/models/mlp_quant.onnx 19850)
set_tests_properties(integration.security PROPERTIES TIMEOUT 120)

if(IP_ENABLE_TLS)
  add_test(NAME integration.tls
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/tls_test.py
      $<TARGET_FILE:inference_server> $<TARGET_FILE:inference_router>
      ${CMAKE_SOURCE_DIR}/bench/models/mlp_quant.onnx 19870)
  set_tests_properties(integration.tls PROPERTIES TIMEOUT 120)
endif()

add_executable(llm_tests llm/scheduler_test.cpp)
target_link_libraries(llm_tests PRIVATE llm GTest::gtest_main GTest::gmock)
gtest_discover_tests(llm_tests)

add_test(NAME integration.llm_streaming
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/llm_streaming_test.py
    $<TARGET_FILE:llm_server> 19890)
set_tests_properties(integration.llm_streaming PROPERTIES TIMEOUT 120)
