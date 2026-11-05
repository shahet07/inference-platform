add_library(stream src/exactly_once.cpp src/interval_join.cpp src/sliding_window.cpp src/sharded_log.cpp src/consumer_group.cpp src/wal.cpp src/window.cpp src/processor.cpp)
target_include_directories(stream PUBLIC include)
