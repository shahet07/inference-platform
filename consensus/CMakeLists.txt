add_library(consensus src/raft.cpp src/file_storage.cpp)
target_include_directories(consensus PUBLIC include)

add_executable(raft_node src/raft_node_main.cpp)
target_link_libraries(raft_node PRIVATE consensus httplib::httplib nlohmann_json::nlohmann_json)
