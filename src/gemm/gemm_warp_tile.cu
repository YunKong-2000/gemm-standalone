#include "gemm/gemm_thread_tile.cuh"

void gemm_warp_tile(const GemmArgs& args) {
  gemm_detail::launch_thread_tile<8, 8, ThreadTileLayout8x8<false>>(args, "warp_tile");
}
