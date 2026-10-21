#pragma once

#include "gemm/gemm_regular.cuh"
#include "gemm/thread_tile_layout.h"

namespace gemm_detail {
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

// Row-major thread order for contiguous thread tiles.
template<int TM, int TN>
struct ContiguousThreadTileLayout {
  __device__ __forceinline__ static int row(int tid, int element) {
    return (tid / (BN / TN)) * TM + element;
  }
  __device__ __forceinline__ static int col(int tid, int element) {
    return (tid % (BN / TN)) * TN + element;
  }
};

// All synchronous thread/warp tiles use this implementation. Layout only
// supplies A/B fragment coordinates and matching C coordinates; it cannot
// customize loading, arithmetic, synchronization, or launch constraints.
template<int TM, int TN, class Layout = ContiguousThreadTileLayout<TM, TN>>
// The 4x4 variant launches 1024 threads: on SM80 its register budget is
// at most 64 registers/thread (65536 registers/block). Without launch bounds,
// ptxas can optimize for fewer threads and produce an unlaunchable kernel.
// Bound each specialization independently: 1024 / 512 / 256 / 128 threads.
__global__ __launch_bounds__((BM / TM) * (BN / TN), 1)
void gemm_thread_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
  constexpr int THREADS = (BM / TM) * (BN / TN);
  static_assert(BM % TM == 0 && BN % TN == 0);
  static_assert(TM % 4 == 0 && TN % 4 == 0);
  static_assert(THREADS <= 1024);
  const int tid = threadIdx.x;
  const int by = blockIdx.y, bx = blockIdx.x;
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  alignas(16) float rAcc[TM][TN] = {};
  alignas(16) float rA[TM], rB[TN];

  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    loadSmem<BM, BK, THREADS>(A + int64_t(k_iter) * M + by * BM, &As[0][0], M, tid);
    loadSmem<BN, BK, THREADS>(B + int64_t(k_iter) * N + bx * BN, &Bs[0][0], N, tid);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      #pragma unroll
      for (int i = 0; i < TM; i += 4) {
        VectorizedAccess(&As[k][Layout::row(tid, i)], rA + i);
      }
      #pragma unroll
      for (int j = 0; j < TN; j += 4) {
        VectorizedAccess(&Bs[k][Layout::col(tid, j)], rB + j);
      }
      #pragma unroll
      for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
          rAcc[i][j] += rA[i] * rB[j];
        }
      }
    }
    __syncthreads();
  }
  // Use the same layout for operand fragments and output ownership.
  #pragma unroll
  for (int i = 0; i < TM; i++) {
    #pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const int row = by * BM + Layout::row(tid, i);
      const int col = bx * BN + Layout::col(tid, j);
      VectorizedAccess(&rAcc[i][j], C + int64_t(row) * N + col);
    }
  }
}

template<int TM, int TN, class Layout = ContiguousThreadTileLayout<TM, TN>>
void launch_thread_tile(const GemmArgs& args, const char* name) {
  check_gemm_args(args, BM, BN, BK);
  const int M = args.M, N = args.N, K = args.K;
  gemm_thread_tile_kernel<TM, TN, Layout><<<dim3(N / BN, M / BM), (BM / TM) * (BN / TN), 0, args.stream>>>(
      args.A, args.B, args.C, M, N, K);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    cudaFuncAttributes attributes{};
    const auto attribute_status = cudaFuncGetAttributes(&attributes, gemm_thread_tile_kernel<TM, TN, Layout>);
    std::string message = std::string(name) +
        ": " + cudaGetErrorString(status) +
        "; threads/block=" + std::to_string((BM / TM) * (BN / TN));
    if (attribute_status == cudaSuccess) {
      message += "; registers/thread=" + std::to_string(attributes.numRegs) +
          "; static_smem_bytes=" + std::to_string(attributes.sharedSizeBytes) +
          "; function_max_threads=" + std::to_string(attributes.maxThreadsPerBlock);
    }
    throw std::runtime_error(message);
  }
}
} // namespace gemm_detail
