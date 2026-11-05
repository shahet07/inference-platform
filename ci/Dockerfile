# Builds and tests on Linux. On Apple Silicon, --platform linux/amd64 runs x86-64 under emulation,
# which exercises the AVX2 kernel; timings from an emulated build mean nothing.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang libclang-rt-dev llvm liburing-dev libssl-dev cmake ninja-build python3 ca-certificates git curl
ENV CXX=clang++
WORKDIR /src
COPY . .
