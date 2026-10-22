#include "gemm/gemm_regular.cuh"

namespace {
constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 64;
constexpr int THREADS = BM * BN;

__global__ void gemm_block_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
  const int tid = threadIdx.y * BN + threadIdx.x;
  const int by = blockIdx.y, bx = blockIdx.x;
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];
  float acc = 0;
  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    for (int i = tid; i < BK * BM; i += THREADS) {
      As[i / BM][i % BM] = A[int64_t(k_iter + i / BM) * M + by * BM + i % BM];
    }
    for (int i = tid; i < BK * BN; i += THREADS) {
      Bs[i / BN][i % BN] = B[int64_t(k_iter + i / BN) * N + bx * BN + i % BN];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      acc += As[k][threadIdx.y] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  C[int64_t(by * BM + threadIdx.y) * N + bx * BN + threadIdx.x] = acc;
}
} // namespace

void gemm_block_tile(const GemmArgs& args) {
  check_gemm_args(args, BM, BN, BK, false);
  const int M = args.M, N = args.N, K = args.K;
  gemm_block_tile_kernel<<<dim3(N / BN, M / BM), dim3(BN, BM), 0, args.stream>>>(
      args.A, args.B, args.C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
