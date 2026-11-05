add_library(security src/api_keys.cpp src/sha256.cpp)
target_include_directories(security PUBLIC include)
