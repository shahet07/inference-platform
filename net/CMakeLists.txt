add_library(net src/event_server.cpp src/http_parser.cpp src/poller.cpp)
target_include_directories(net PUBLIC include)
