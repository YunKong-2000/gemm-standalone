#include "gemm/gemm_regular.cuh"

namespace {
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

template<int TM, int TN, bool WARP_4X8 = false>
// The 4x4 variant launches 1024 threads: on SM80 its register budget is
// at most 64 registers/thread (65536 registers/block). Without launch bounds,
// ptxas can optimize for fewer threads and produce an unlaunchable kernel.
// Bound each specialization independently: 1024 / 512 / 256 threads.
__global__ __launch_bounds__((BM / TM) * (BN / TN), 1)
void gemm_thread_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
  constexpr int THREADS = (BM / TM) * (BN / TN);
  static_assert(THREADS <= 1024);
  const int tid = threadIdx.x;
  int ty, tx;
  if constexpr (WARP_4X8) {
    static_assert(TM == 8 && TN == 8);
    // Each warp has 4x8 threads and covers a contiguous 32x64 output tile.
    // Eight warps form a 4x2 grid within the 128x128 block tile.
    constexpr int WARPS_N = BN / (8 * TN);
    static_assert(BM % (4 * TM) == 0 && BN % (8 * TN) == 0);
    const int warp_id = tid / 32, lane_id = tid % 32;
    ty = (warp_id / WARPS_N) * 4 + lane_id / 8;
    tx = (warp_id % WARPS_N) * 8 + lane_id % 8;
  } else {
    ty = tid / (BN / TN);
    tx = tid % (BN / TN);
  }
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
        VectorizedAccess(&As[k][ty * TM + i], rA + i);
      }
      #pragma unroll
      for (int j = 0; j < TN; j += 4) {
        VectorizedAccess(&Bs[k][tx * TN + j], rB + j);
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
  // Each thread owns one contiguous TM x TN rectangle.
  #pragma unroll
  for (int i = 0; i < TM; i++) {
    #pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const int row = by * BM + ty * TM + i;
      const int col = bx * BN + tx * TN + j;
      VectorizedAccess(&rAcc[i][j], C + int64_t(row) * N + col);
    }
  }
}

template<int TM, int TN, bool WARP_4X8 = false>
void launch_thread_tile(const GemmArgs& args) {
  check_gemm_args(args, BM, BN, BK);
  const int M = args.M, N = args.N, K = args.K;
  gemm_thread_tile_kernel<TM, TN, WARP_4X8><<<dim3(N / BN, M / BM), (BM / TM) * (BN / TN), 0, args.stream>>>(
      args.A, args.B, args.C, M, N, K);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    cudaFuncAttributes attributes{};
    const auto attribute_status = cudaFuncGetAttributes(&attributes, gemm_thread_tile_kernel<TM, TN, WARP_4X8>);
    std::string message = "thread_tile_" + std::to_string(TM) + "x" + std::to_string(TN) +
        (WARP_4X8 ? "_warp4x8" : "") +
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
} // namespace

void gemm_thread_tile_4x4(const GemmArgs& args) {
  launch_thread_tile<4, 4>(args);
}
void gemm_thread_tile_4x8(const GemmArgs& args) {
  launch_thread_tile<4, 8>(args);
}
void gemm_thread_tile_8x8(const GemmArgs& args) {
  launch_thread_tile<8, 8>(args);
}
void gemm_thread_tile_8x8_warp4x8(const GemmArgs& args) {
  launch_thread_tile<8, 8, true>(args);
}
