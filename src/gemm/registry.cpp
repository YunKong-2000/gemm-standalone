#include "gemm/gemm.h"

const std::vector<Kernel>& gemm_kernels() {
  static const std::vector<Kernel> kernels = {
    {"naive", 1, 16, 16, 1, gemm_naive},
    {"block_tile", 2, 16, 16, 8, gemm_block_tile},
    {"block_tile_float4_16x16", 3, 16, 16, 8, gemm_block_tile_float4_16x16},
    {"block_tile_float4_16x32", 3, 16, 32, 8, gemm_block_tile_float4_16x32},
    {"block_tile_float4_32x32", 3, 32, 32, 8, gemm_block_tile_float4_32x32},
    {"thread_tile_4x4", 4, 128, 128, 8, gemm_thread_tile_4x4},
    {"thread_tile_4x8", 4, 128, 128, 8, gemm_thread_tile_4x8},
    {"thread_tile_8x8", 4, 128, 128, 8, gemm_thread_tile_8x8},
    {"thread_tile_8x8_warp4x8", 4, 128, 128, 8, gemm_thread_tile_8x8_warp4x8},
    {"thread_tile_8x16", 4, 128, 128, 8, gemm_thread_tile_8x16},
    {"warp_tile", 5, 128, 128, 8, gemm_warp_tile},
    {"prefetch_smem", 6, 128, 128, 8, gemm_prefetchSmem},
    {"prefetch_reg", 7, 128, 128, 8, gemm_prefetchReg},
    {"split_k", 8, 128, 128, 16, gemm_splictK},
    {"cublas", 9, 1, 1, 1, gemm_cublas},
  };
  return kernels;
}
