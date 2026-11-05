add_library(stream src/sharded_log.cpp src/consumer_group.cpp src/window.cpp src/processor.cpp)
target_include_directories(stream PUBLIC include)
