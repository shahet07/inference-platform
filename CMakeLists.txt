cmake_minimum_required(VERSION 3.24)
project(inference_platform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(IP_ENABLE_CLANG_TIDY "Run clang-tidy on project sources during the build" OFF)
option(IP_BUILD_FUZZ "Build libFuzzer targets; needs LLVM clang, not Apple clang" OFF)
set(IP_SANITIZE "" CACHE STRING "Sanitizers to enable, e.g. 'address,undefined' or 'thread'")

# Sanitizer flags are global so that dependencies built from source are instrumented too.
if(IP_SANITIZE)
  add_compile_options(-fsanitize=${IP_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${IP_SANITIZE})
endif()

# Dependencies come first so the warning flags and clang-tidy below only apply to our code.
include(cmake/Dependencies.cmake)
# Leaving out designated initializers is deliberate here, but clang 18 warns about it.
add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers)
if(IP_ENABLE_CLANG_TIDY)
  include(cmake/ClangTidy.cmake)
endif()

add_subdirectory(obs)
add_subdirectory(perf)
add_subdirectory(runtime)
add_subdirectory(stream)
add_subdirectory(cluster)
add_subdirectory(consensus)
add_subdirectory(server)
add_subdirectory(bench)

if(IP_BUILD_FUZZ)
  add_subdirectory(fuzz)
endif()

enable_testing()
add_subdirectory(tests)
