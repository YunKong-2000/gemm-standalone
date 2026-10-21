#include "gemm/gemm_regular.cuh"

namespace {
constexpr int BM = 16;
constexpr int BN = 16;

__global__ void gemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
  const int row = blockIdx.y * BM + threadIdx.y;
  const int col = blockIdx.x * BN + threadIdx.x;
  float acc = 0;
  for (int k = 0; k < K; k++) {
    acc += A[int64_t(k) * M + row] * B[int64_t(k) * N + col];
  }
  C[int64_t(row) * N + col] = acc;
}
} // namespace

void gemm_naive(const GemmArgs& args) {
  check_gemm_args(args, BM, BN, 1, false);
  const int M = args.M, N = args.N, K = args.K;
  gemm_naive_kernel<<<dim3(N / BN, M / BM), dim3(BN, BM), 0, args.stream>>>(
      args.A, args.B, args.C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
