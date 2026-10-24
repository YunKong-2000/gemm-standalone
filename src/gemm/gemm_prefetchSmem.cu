#include "gemm/gemm.h"
#include "gemm/thread_tile_layout.h"

namespace {
constexpr int THREADS = 256;
constexpr int MIN_BLOCKS_PER_SM = 2;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int VEC_LEN = 4;
constexpr int STAGES = 2;
using Element = float;
using ElementVec = float4;

static_assert(TM == 8 && TN == 8);
static_assert(THREADS * TM * TN == BM * BN);

__inline__ __device__ void VectorizedAccess(const Element* src, Element* dst)
{
  *reinterpret_cast<ElementVec*>(dst) = *reinterpret_cast<const ElementVec*>(src);
}

template<bool Contiguous, int LaneCols = Contiguous ? 8 : 4>
__inline__ __device__ void InnerProduct(Element* sA, Element* sB, Element rAcc[TM][TN], const int tid) {
  alignas(16) Element rA[TM];
  alignas(16) Element rB[TN];
  using Layout = ThreadTileLayout8x8<Contiguous, LaneCols>;
  #pragma unroll
  for (int k = 0; k < BK; k++) {
    const int M0 = k * BM + Layout::row(tid, 0);
    const int N0 = k * BN + Layout::col(tid, 0);
    VectorizedAccess(sA + M0, rA);
    VectorizedAccess(sB + N0, rB);
    VectorizedAccess(sA + M0 + Layout::row_group_stride , rA + TM / 2);
    VectorizedAccess(sB + N0 + Layout::col_group_stride, rB + TN / 2);
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        rAcc[i][j] += rA[i] * rB[j];
      }
    }
  }
}

// SM80+: copy one aligned float4 directly from global to shared memory.
// Completion is tracked per thread; a block barrier is still required before
// another thread consumes the copied data.
__forceinline__ __device__ void CopyAsync16(const Element* src, Element* dst) {
  const unsigned int shared_addr = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :: "r"(shared_addr), "l"(src) : "memory");
}

__forceinline__ __device__ void CommitAsyncCopies() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__forceinline__ __device__ void WaitAsyncCopies() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// global A:(K, M), As:(BK, BM), GD=M, SD=BM
// global B:(K, N), Bs:(BK, BN), GD=N, SD=BN
// Enqueue only: the caller commits A/B together and waits before consumption.
template<int SD>
__forceinline__ __device__ void loadSmemAsync(
  const Element* src,
  Element* dst,
  const int GD,
  const int tid) {
  const int VEC_PER_ROW = SD / VEC_LEN;
  const int VEC_NUM = (BK * SD) / VEC_LEN;
  #pragma unroll
  for (int i = tid; i < VEC_NUM; i += THREADS) {
    const int row = i / VEC_PER_ROW;
    const int col = (i % VEC_PER_ROW) * VEC_LEN;
    CopyAsync16(src + int64_t(row) * GD + col, dst + row * SD + col);
  }
}

// Host validation guarantees full tiles and aligned global pointers. All
// device loads/stores are unconditional; there are no element bounds checks.
// On SM80, 64K registers / (256 threads * 2 blocks) caps allocation at 128
// registers/thread. Without this bound, scheduling the unrolled inner product
// can extend fragment lifetimes enough to leave room for only one block/SM.
// The two shared-memory stages use 16 KiB/block, so two blocks also fit in smem.
template<bool Contiguous, int LaneCols = Contiguous ? 8 : 4>
__global__ __launch_bounds__(THREADS, MIN_BLOCKS_PER_SM) void gemm_kernel(
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
  using Layout = ThreadTileLayout8x8<Contiguous, LaneCols>;

  // Compile-time storage alignment, not runtime alignment checks.
  __shared__ __align__(16) Element As[STAGES][BK][BM];
  __shared__ __align__(16) Element Bs[STAGES][BK][BN];
  alignas(16) Element rAcc[TM][TN];

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      rAcc[i][j] = 0;
    }
  }

  loadSmemAsync<BM>(A + by * BM, &As[0][0][0], M, tid);
  loadSmemAsync<BN>(B + bx * BN, &Bs[0][0][0], N, tid);
  CommitAsyncCopies();
  WaitAsyncCopies();
  __syncthreads();
  int stage = 0;
  const int k_tile_num = K / BK;
  #pragma unroll
  for (int k = 0; k < k_tile_num - 1; k++) {
    // The preceding barrier protects reuse of the other stage. Copy the next
    // tile asynchronously while LDS/FFMA consume the current stage.
    loadSmemAsync<BM>(A + int64_t(k + 1) * BK * M + by * BM, &As[stage ^ 1][0][0], M, tid);
    loadSmemAsync<BN>(B + int64_t(k + 1) * BK * N + bx * BN, &Bs[stage ^ 1][0][0], N, tid);
    CommitAsyncCopies();
    InnerProduct<Contiguous, LaneCols>(&As[stage][0][0], &Bs[stage][0][0], rAcc, tid);
    // Wait only after independent computation. Each thread waits for its own
    // copies; the barrier makes the full tile ready and retires current readers.
    WaitAsyncCopies();
    __syncthreads();
    stage ^= 1;
  }
  // Drain the final tile without a prefetch or a final barrier. For K == BK,
  // the main loop is empty and this computes the initially loaded tile.
  InnerProduct<Contiguous, LaneCols>(&As[stage][0][0], &Bs[stage][0][0], rAcc, tid);

  #pragma unroll
  for (int a = 0; a < 2; a++) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int b = 0; b < 2; b++) {
        int row = by * BM + Layout::row(tid, a * 4 + i);
        int col = bx * BN + Layout::col(tid, b * 4);
        VectorizedAccess(&rAcc[a * 4 + i][b * 4], C + int64_t(row) * N + col);
      }
    }
  }
}// kernel

template<bool Contiguous, int LaneCols = Contiguous ? 8 : 4>
void launch_variant(const GemmArgs& args) {
  check_gemm_args(args, BM, BN, BK);
  const int M = args.M, N = args.N, K = args.K;
  gemm_kernel<Contiguous, LaneCols><<<dim3(N / BN, M / BM), THREADS, 0, args.stream>>>(
      M, N, K, args.A, args.B, args.C);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gemm_prefetchSmem(const GemmArgs& args) {
  launch_variant<false>(args);
}

void gemm_prefetchSmem_contiguous8x8(const GemmArgs& args) {
  launch_variant<true>(args);
}

void gemm_prefetchSmem_contiguous8x8_warp2x16(const GemmArgs& args) {
  launch_variant<true, 16>(args);
}
