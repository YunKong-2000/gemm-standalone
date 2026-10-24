cmake_minimum_required(VERSION 3.18)
project(gemm_standalone LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(GEMM_HOST_TESTS_ONLY "Build CPU helper tests without a CUDA toolkit" OFF)
option(GEMM_PTXAS_VERBOSE "Print registers, shared memory and spills for CUDA kernels" OFF)

enable_testing()
add_executable(gemm_host_tests tests/test_host.cpp)
target_include_directories(gemm_host_tests PRIVATE include)
add_test(NAME host_tests COMMAND gemm_host_tests)
if(GEMM_HOST_TESTS_ONLY)
  return()
endif()

# 80 emits both an SM80 cubin and PTX, allowing JIT on newer GPUs.
# Override with -DCMAKE_CUDA_ARCHITECTURES=89, 90, etc. for native compilation.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA GPU architectures")
endif()
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(arch MATCHES "^[0-9]+" AND CMAKE_MATCH_0 LESS 80)
    message(FATAL_ERROR "The asynchronous-copy kernels require SM80+")
  endif()
endforeach()
enable_language(CUDA)
find_package(CUDAToolkit 11.0 REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(gemm_kernels STATIC
  src/gemm/registry.cpp
  src/gemm/gemm_naive.cu
  src/gemm/gemm_block_tile.cu
  src/gemm/gemm_block_tile_float4.cu
  src/gemm/gemm_thread_tile.cu
  src/gemm/gemm_warp_tile.cu
  src/gemm/gemm_prefetchSmem.cu
  src/gemm/gemm_prefetchRegister.cu
  src/gemm/gemm_splictK.cu
  src/gemm/gemm_cublas.cu)
target_include_directories(gemm_kernels PUBLIC include)
target_link_libraries(gemm_kernels PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(gemm_kernels PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo>")
if(GEMM_PTXAS_VERBOSE)
  target_compile_options(gemm_kernels PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>")
endif()

add_executable(gemm_bench src/benchmark.cpp)
target_link_libraries(gemm_bench PRIVATE gemm_kernels)
add_executable(gemm_cuda_tests tests/test_cuda.cpp)
target_link_libraries(gemm_cuda_tests PRIVATE gemm_kernels)
add_executable(gemm_profile src/profile.cpp)
target_link_libraries(gemm_profile PRIVATE gemm_kernels)
add_test(NAME cuda_correctness COMMAND gemm_cuda_tests)
add_test(NAME cuda_graph_smoke COMMAND gemm_bench --shapes 128,256,64
  --warmup 1 --repeat 2 --batch 2 --output graph_smoke.csv)
add_test(NAME cuda_event_smoke COMMAND gemm_bench --shapes 256,128,128
  --timing event --warmup 1 --repeat 2 --batch 2 --output event_smoke.csv)
add_test(NAME cuda_profile_smoke COMMAND gemm_profile --impl split_k
  --shape 128,256,96 --warmup 1 --validate-only)
set_tests_properties(cuda_correctness cuda_graph_smoke cuda_event_smoke cuda_profile_smoke
  PROPERTIES LABELS CUDA RUN_SERIAL TRUE)
