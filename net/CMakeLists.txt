add_library(net src/event_server.cpp src/http_parser.cpp src/poller.cpp)
target_include_directories(net PUBLIC include)

if(IP_ENABLE_IO_URING)
  find_library(IP_LIBURING uring REQUIRED)
  target_link_libraries(net PRIVATE ${IP_LIBURING})
  target_compile_definitions(net PRIVATE IP_HAS_IO_URING)
endif()
