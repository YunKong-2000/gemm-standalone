#include "gemm/gemm_regular.cuh"

namespace {
constexpr int BK = 8;

template<int BM, int BN>
__global__ void gemm_block_tile_float4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
  constexpr int THREADS = BM * BN;
  static_assert(THREADS <= 1024);
  const int tid = threadIdx.x;
  const int ty = tid / BN, tx = tid % BN;
  const int by = blockIdx.y, bx = blockIdx.x;
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  float acc = 0;
  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    loadSmem<BM, BK, THREADS>(A + int64_t(k_iter) * M + by * BM, &As[0][0], M, tid);
    loadSmem<BN, BK, THREADS>(B + int64_t(k_iter) * N + bx * BN, &Bs[0][0], N, tid);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      acc += As[k][ty] * Bs[k][tx];
    }
    __syncthreads();
  }
  // Exactly one scalar accumulator and one C element per thread.
  C[int64_t(by * BM + ty) * N + bx * BN + tx] = acc;
}

template<int BM, int BN>
void launch_block_tile_float4(const GemmArgs& args) {
  check_gemm_args(args, BM, BN, BK);
  const int M = args.M, N = args.N, K = args.K;
  gemm_block_tile_float4_kernel<BM, BN><<<dim3(N / BN, M / BM), BM * BN, 0, args.stream>>>(
      args.A, args.B, args.C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
} // namespace

void gemm_block_tile_float4_16x16(const GemmArgs& args) {
  launch_block_tile_float4<16, 16>(args);
}
void gemm_block_tile_float4_16x32(const GemmArgs& args) {
  launch_block_tile_float4<16, 32>(args);
}
void gemm_block_tile_float4_32x32(const GemmArgs& args) {
  launch_block_tile_float4<32, 32>(args);
}
