add_library(llm src/scheduler.cpp)
target_include_directories(llm PUBLIC include)
target_link_libraries(llm PUBLIC obs)

add_executable(llm_server src/llm_server_main.cpp)
target_link_libraries(llm_server PRIVATE llm obs httplib::httplib nlohmann_json::nlohmann_json)

if(IP_ENABLE_LLAMA)
  add_library(llm_llama src/llama_backend.cpp)
  target_link_libraries(llm_llama PUBLIC llm PRIVATE llama)
  target_link_libraries(llm_server PRIVATE llm_llama)
  target_compile_definitions(llm_server PRIVATE IP_HAS_LLAMA)
endif()
