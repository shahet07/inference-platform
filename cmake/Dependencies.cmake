include(FetchContent)

# GoogleTest, built from source so it picks up the sanitizer flags.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SYSTEM
)

# ONNX Runtime, as an official prebuilt binary. Pinned by hash per platform.
set(IP_ORT_VERSION 1.22.0)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(ip_ort_platform osx-arm64)
  set(ip_ort_sha256 cab6dcbd77e7ec775390e7b73a8939d45fec3379b017c7cb74f5b204c1a1cc07)
  set(ip_ort_lib libonnxruntime.dylib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(ip_ort_platform linux-x64)
  set(ip_ort_sha256 8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3)
  set(ip_ort_lib libonnxruntime.so)
else()
  message(FATAL_ERROR "No pinned ONNX Runtime binary for "
                      "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
endif()

FetchContent_Declare(
  onnxruntime
  URL https://github.com/microsoft/onnxruntime/releases/download/v${IP_ORT_VERSION}/onnxruntime-${ip_ort_platform}-${IP_ORT_VERSION}.tgz
  URL_HASH SHA256=${ip_ort_sha256}
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)

# HTTP server and client (header-only). Compression and TLS stay off: this is an internal
# service, and their dependencies would add noise to latency measurements.
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_NON_BLOCKING_GETADDRINFO OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.20.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SYSTEM
)

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SYSTEM
)

# Eigen is the comparison baseline in bench/kernels_bench.cpp. It is header-only, so its own build
# is skipped by pointing SOURCE_SUBDIR at a directory that does not exist.
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SOURCE_SUBDIR no-cmake-build
)

FetchContent_MakeAvailable(googletest onnxruntime httplib nlohmann_json eigen)

add_library(eigen_headers INTERFACE)
target_include_directories(eigen_headers SYSTEM INTERFACE ${eigen_SOURCE_DIR})

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/${ip_ort_lib}
  IMPORTED_NO_SONAME FALSE
  INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/include
)
