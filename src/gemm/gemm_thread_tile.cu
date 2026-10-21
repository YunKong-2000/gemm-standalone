#include "gemm/gemm_thread_tile.cuh"

void gemm_thread_tile_4x4(const GemmArgs& args) {
  gemm_detail::launch_thread_tile<4, 4>(args, "thread_tile_4x4");
}
void gemm_thread_tile_4x8(const GemmArgs& args) {
  gemm_detail::launch_thread_tile<4, 8>(args, "thread_tile_4x8");
}
void gemm_thread_tile_8x8(const GemmArgs& args) {
  gemm_detail::launch_thread_tile<8, 8, ThreadTileLayout8x8<true, 16>>(args, "thread_tile_8x8");
}
void gemm_thread_tile_8x8_warp4x8(const GemmArgs& args) {
  gemm_detail::launch_thread_tile<8, 8, ThreadTileLayout8x8<true, 8>>(args, "thread_tile_8x8_warp4x8");
}
