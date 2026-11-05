add_library(obs src/log.cpp src/metrics.cpp src/trace.cpp)
target_include_directories(obs PUBLIC include)
