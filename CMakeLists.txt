cmake_minimum_required(VERSION 3.20)
project(matmul CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(MATMUL_RELEASE_FLAGS "-O3;-march=native")
add_compile_options("$<$<CONFIG:Release>:${MATMUL_RELEASE_FLAGS}>")

# Ladder order: source list order fixes table registration order. New variant
# files get appended here, in v2, v3, ... order, never inserted out of order.
set(MATMUL_SOURCES
  src/registry.cpp
  src/harness.cpp
  src/v1_naive.cpp
  # Listed ahead of being written so phases 1 and 2 can run in parallel without
  # both editing this file. An empty file here compiles and registers nothing.
  src/v2_reorder.cpp
  src/v3_tiled.cpp
  src/v4_simd.cpp
  src/v5_threaded.cpp
  # v6_cuda_naive.cu  (phase 3, requires enable_language(CUDA) below)
  # v7_cuda_tiled.cu  (phase 3)
  # v8_cublas.cu      (phase 3)
)

add_executable(matmul ${MATMUL_SOURCES})
target_include_directories(matmul PRIVATE include)

# Reproducibility header: stamp the actual release flags into the binary so
# the harness can print them verbatim.
string(REPLACE ";" " " MATMUL_BUILD_FLAGS_STR "${MATMUL_RELEASE_FLAGS}")
target_compile_definitions(matmul PRIVATE
  "MATMUL_BUILD_FLAGS=\"${MATMUL_BUILD_FLAGS_STR}\"")

# --- Phase 3 CUDA hook (do not enable in phase 0) ---------------------------
# enable_language(CUDA)
# set(CMAKE_CUDA_ARCHITECTURES 89)
# target_sources(matmul PRIVATE src/v6_cuda_naive.cu src/v7_cuda_tiled.cu src/v8_cublas.cu)
# target_link_libraries(matmul PRIVATE cublas)
# -----------------------------------------------------------------------------
