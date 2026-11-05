add_library(server src/batching_engine.cpp src/http_server.cpp src/model_manager.cpp src/event_frontend.cpp)
target_include_directories(server PUBLIC include)
target_link_libraries(server PUBLIC obs stream cluster perf net security PRIVATE httplib::httplib nlohmann_json::nlohmann_json)

add_executable(inference_server src/main.cpp)
target_link_libraries(inference_server PRIVATE server runtime)
