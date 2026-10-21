#pragma once

#ifdef __CUDACC__
#define GEMM_LAYOUT_INLINE __host__ __device__ __forceinline__
#else
#define GEMM_LAYOUT_INLINE inline
#endif

// All layouts cover a 128x128 CTA tile with 256 threads and 64 accumulators
// per thread. The same mapping is used for A/B fragments and C writeback.
template<bool Contiguous, int LaneCols = Contiguous ? 8 : 4>
struct ThreadTileLayout8x8 {
  static_assert(Contiguous ? (LaneCols == 8 || LaneCols == 16) : LaneCols == 4);
  static constexpr int warp_m = Contiguous ? (32 / LaneCols) * 8 : 64;
  static constexpr int warp_n = Contiguous ? LaneCols * 8 : 32;
  static constexpr int warps_n = 128 / warp_n;
  static constexpr int lane_cols = LaneCols;
  static constexpr int lane_stride = Contiguous ? 8 : 4;
  static constexpr int row_group_stride = Contiguous ? 4 : 32;
  static constexpr int col_group_stride = Contiguous ? 4 : 16;

  GEMM_LAYOUT_INLINE static constexpr int row(int tid, int element) {
    return (tid / 32 / warps_n) * warp_m +
        (tid % 32 / lane_cols) * lane_stride +
        (element / 4) * row_group_stride + element % 4;
  }

  GEMM_LAYOUT_INLINE static constexpr int col(int tid, int element) {
    return (tid / 32 % warps_n) * warp_n +
        (tid % 32 % lane_cols) * lane_stride +
        (element / 4) * col_group_stride + element % 4;
  }
};

#undef GEMM_LAYOUT_INLINE
