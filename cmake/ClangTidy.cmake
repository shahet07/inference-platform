# Homebrew's LLVM is keg-only on macOS, so it is not on PATH by default.
find_program(IP_CLANG_TIDY clang-tidy HINTS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin)
if(NOT IP_CLANG_TIDY)
  message(FATAL_ERROR "IP_ENABLE_CLANG_TIDY is ON but clang-tidy was not found")
endif()
set(CMAKE_CXX_CLANG_TIDY ${IP_CLANG_TIDY})
