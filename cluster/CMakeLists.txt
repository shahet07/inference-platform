add_library(cluster
  src/hash_ring.cpp
  src/health_monitor.cpp
  src/model_registry.cpp
  src/replicator.cpp
  src/router.cpp)
target_include_directories(cluster PUBLIC include)
target_link_libraries(cluster PUBLIC obs security PRIVATE httplib::httplib nlohmann_json::nlohmann_json)

add_executable(inference_router src/router_main.cpp)
target_link_libraries(inference_router PRIVATE cluster)
