#include "gemm/gemm_regular.cuh"

namespace {
constexpr int THREADS = 256;
constexpr int WARPSIZE = 32;
constexpr int WARP_NUM_M = 2;
constexpr int WARP_NUM_N = 4;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int WM = BM / WARP_NUM_M;
constexpr int WN = BN / WARP_NUM_N;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THR_NUM_N = 4;
constexpr int THR_NUM_M = 8;
constexpr int VEC_LEN = 4;
constexpr int WM_SUB = THR_NUM_M * VEC_LEN;
constexpr int WN_SUB = THR_NUM_N * VEC_LEN;
using Element = float;
using ElementVec = float4;

static_assert(TM == 8 && TN == 8);
static_assert(THREADS == WARPSIZE * WARP_NUM_M * WARP_NUM_N);
static_assert(THR_NUM_M * THR_NUM_N == WARPSIZE);
static_assert(WM == THR_NUM_M * TM && WN == THR_NUM_N * TN);

__inline__ __device__ void InnerProduct(Element* sA, Element* sB, Element rAcc[TM][TN], const int lane_id) {
  alignas(16) Element rA[TM];
  alignas(16) Element rB[TN];
  const int lane_x = lane_id % THR_NUM_N;
  const int lane_y = lane_id / THR_NUM_N;
  #pragma unroll
  for (int k = 0; k < BK; k++) {
    const int M0 = k * BM + lane_y * VEC_LEN;
    const int N0 = k * BN + lane_x * VEC_LEN;
    VectorizedAccess(sA + M0, rA);
    VectorizedAccess(sB + N0, rB);
    VectorizedAccess(sA + M0 + WM_SUB , rA + TM / 2);
    VectorizedAccess(sB + N0 + WN_SUB, rB + TN / 2);
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        rAcc[i][j] += rA[i] * rB[j];
      }
    }
  }
}

__global__ void gemm_warp_tile_kernel(
  const int M,
  const int N,
  const int K,
  const Element* A,
  const Element* B,
  Element* C
)
{
  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARPSIZE;
  const int warp_x = warp_id % WARP_NUM_N;
  const int warp_y = warp_id / WARP_NUM_N;
  const int lane_id = tid % WARPSIZE;

  // Compile-time storage alignment, not runtime alignment checks.
  __shared__ __align__(16) Element As[BK][BM];
  __shared__ __align__(16) Element Bs[BK][BN];
  alignas(16) Element rAcc[TM][TN];

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      rAcc[i][j] = 0;
    }
  }

  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    loadSmem<BM, BK, THREADS>(A + int64_t(k_iter) * M + by * BM, &As[0][0], M, tid);
    loadSmem<BN, BK, THREADS>(B + int64_t(k_iter) * N + bx * BN, &Bs[0][0], N, tid);
    __syncthreads();
    InnerProduct(&As[0][0] + WM * warp_y, &Bs[0][0] + WN * warp_x, rAcc, lane_id);
    __syncthreads();
  }

  // Four 4x4 subtiles per thread; the two row/column groups are separated
  // by WM_SUB=32 and WN_SUB=16 inside a 64x32 warp tile.
  const int lane_x = lane_id % THR_NUM_N;
  const int lane_y = lane_id / THR_NUM_N;
  #pragma unroll
  for (int a = 0; a < 2; a++) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int b = 0; b < 2; b++) {
        int row = by * BM + warp_y * WM + lane_y * VEC_LEN + a * WM_SUB + i;
        int col = bx * BN + warp_x * WN + lane_x * VEC_LEN + b * WN_SUB;
        VectorizedAccess(&rAcc[a * 4 + i][b * 4], C + int64_t(row) * N + col);
      }
    }
  }
}// kernel

} // namespace

void gemm_warp_tile(const GemmArgs& args) {
  check_gemm_args(args, BM, BN, BK);
  const int M = args.M, N = args.N, K = args.K;
  gemm_warp_tile_kernel<<<dim3(N / BN, M / BM), THREADS, 0, args.stream>>>(
      M, N, K, args.A, args.B, args.C);
  CUDA_CHECK(cudaGetLastError());
}
