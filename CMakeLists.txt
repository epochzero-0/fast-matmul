cmake_minimum_required(VERSION 3.20)
project(matmul CXX)

# Default USE_CUDA to whether a usable CUDA compiler actually exists, rather
# than to ON. Defaulting ON means a plain `cmake -S . -B build` hard-fails on
# any machine without a toolkit, which is how this broke a build mid-phase.
# -DUSE_CUDA=ON still forces it and still fails loudly if CUDA is missing,
# which is what you want when you believe you have it and you do not.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(MATMUL_CUDA_DEFAULT ON)
else()
  set(MATMUL_CUDA_DEFAULT OFF)
  message(STATUS "No CUDA compiler found; building CPU variants only (v1-v5).")
endif()
option(USE_CUDA "Build the CUDA/cuBLAS variants (v6-v8)" ${MATMUL_CUDA_DEFAULT})

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
  # CUDA variants (v6-v8) live in cuda/, not here -- see the USE_CUDA-guarded
  # phase 3 hook below, which adds them via target_sources instead.
)

add_executable(matmul ${MATMUL_SOURCES})
target_include_directories(matmul PRIVATE include)

# Reproducibility header: stamp the actual release flags into the binary so
# the harness can print them verbatim.
string(REPLACE ";" " " MATMUL_BUILD_FLAGS_STR "${MATMUL_RELEASE_FLAGS}")
target_compile_definitions(matmul PRIVATE
  "MATMUL_BUILD_FLAGS=\"${MATMUL_BUILD_FLAGS_STR}\"")

# --- Phase 3 CUDA hook -------------------------------------------------------
# Guarded by USE_CUDA (default ON) so -DUSE_CUDA=OFF still configures and
# builds a working CPU-only binary on a machine with no CUDA toolkit at all.
if(USE_CUDA)
  # nvcc needs a host compiler it accepts, and its idea of "accepts" is
  # narrow: CUDA 12.0 refuses gcc 13+, and on Windows it takes only MSVC.
  # Default to the compiler building the rest of the project and let the
  # caller override with -DCMAKE_CUDA_HOST_COMPILER when that is not one
  # nvcc will tolerate.
  if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
  enable_language(CUDA)
  target_sources(matmul PRIVATE cuda/v6_naive.cu cuda/v7_tiled.cu cuda/v8_cublas.cu)
  # Set on the target, not as a bare variable: CMAKE_CUDA_ARCHITECTURES only
  # seeds the property for targets created after it, and matmul already
  # exists by this point. As a variable here it silently stays empty and
  # generation fails with "CUDA_ARCHITECTURES is empty for target".
  set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES 89)
  set_source_files_properties(
    cuda/v6_naive.cu cuda/v7_tiled.cu cuda/v8_cublas.cu
    PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(matmul PRIVATE MATMUL_USE_CUDA)
  target_link_libraries(matmul PRIVATE cublas)
endif()
# -----------------------------------------------------------------------------
