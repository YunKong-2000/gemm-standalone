#pragma once

#include "bench_utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

inline void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
}
inline void check_cublas(cublasStatus_t status, const char* expression) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(expression) + ": cuBLAS status " + std::to_string(status));
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr)

// Caller-owned device buffers, allocated on the current CUDA device.
// A: dense column-major [M,K]; B/C: dense row-major [K,N]/[M,N].
// No input/output overlap. Calls enqueue work on stream; caller synchronizes.
// handle is required only for cuBLAS; use host pointer mode.
struct GemmArgs {
  const float* A;
  const float* B;
  float* C;
  int M, N, K;
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
};

inline void check_gemm_args(const GemmArgs& args, int BM, int BN, int BK, bool vectorized = true) {
  check_shape({args.M, args.N, args.K}, BM, BN, BK);
  if (!args.A || !args.B || !args.C) throw std::invalid_argument("A/B/C must not be null");
  if (vectorized && (reinterpret_cast<std::uintptr_t>(args.A) % 16 ||
                     reinterpret_cast<std::uintptr_t>(args.B) % 16 ||
                     reinterpret_cast<std::uintptr_t>(args.C) % 16))
    throw std::invalid_argument("A/B/C must be 16-byte aligned");
}

void gemm_naive(const GemmArgs& args);
void gemm_block_tile(const GemmArgs& args);
void gemm_block_tile_float4_16x16(const GemmArgs& args);
void gemm_block_tile_float4_16x32(const GemmArgs& args);
void gemm_block_tile_float4_32x32(const GemmArgs& args);
void gemm_thread_tile_4x4(const GemmArgs& args);
void gemm_thread_tile_4x8(const GemmArgs& args);
void gemm_thread_tile_8x8(const GemmArgs& args);
void gemm_thread_tile_8x8_warp4x8(const GemmArgs& args);
void gemm_thread_tile_8x16(const GemmArgs& args);
void gemm_warp_tile(const GemmArgs& args);
void gemm_prefetchSmem(const GemmArgs& args);
void gemm_prefetchSmem_contiguous8x8(const GemmArgs& args);
void gemm_prefetchSmem_contiguous8x8_warp2x16(const GemmArgs& args);
void gemm_prefetchReg(const GemmArgs& args);
void gemm_prefetchReg_contiguous8x8(const GemmArgs& args);
void gemm_prefetchReg_contiguous8x8_warp2x16(const GemmArgs& args);
void gemm_splictK(const GemmArgs& args);
void gemm_splictK_contiguous8x8(const GemmArgs& args);
void gemm_splictK_contiguous8x8_warp2x16(const GemmArgs& args);
void gemm_cublas(const GemmArgs& args);

struct Kernel {
  const char* name;
  int stage, BM, BN, BK;
  void (*launch)(const GemmArgs&);
  // When selected individually, also benchmark this subtile and its layout variants.
  const char* subtile_baseline = nullptr;
};
const std::vector<Kernel>& gemm_kernels();
