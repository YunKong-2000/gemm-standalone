#pragma once

#include "gemm/gemm.h"

__forceinline__ __device__ void VectorizedAccess(const float* src, float* dst) {
  *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
}

// A: [K,M] storage, As: [BK,BM]; B: [K,N], Bs: [BK,BN].
// Synchronous float4 loads: no prefetch and only one shared-memory buffer.
template<int SD, int BK, int THREADS>
__forceinline__ __device__ void loadSmem(
    const float* src, float* dst, const int GD, const int tid)
{
  #pragma unroll
  for (int i = tid; i < BK * SD / 4; i += THREADS) {
    const int row = i / (SD / 4);
    const int col = (i % (SD / 4)) * 4;
    VectorizedAccess(src + int64_t(row) * GD + col, dst + row * SD + col);
  }
}
