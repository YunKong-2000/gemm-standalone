#include "bench_utils.h"
#include "gemm/thread_tile_layout.h"
#include <iostream>

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function fn) {
  bool rejected = false;
  try { fn(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "expected invalid input to be rejected");
}

template<bool Contiguous, int LaneCols = Contiguous ? 8 : 4> void check_thread_layout() {
  using Layout = ThreadTileLayout8x8<Contiguous, LaneCols>;
  std::vector<int> visits(128 * 128, 0);
  for (int tid = 0; tid < 256; ++tid) {
    for (int i = 0; i < 8; ++i) {
      const int row = Layout::row(tid, i), col = Layout::col(tid, i);
      require(row >= 0 && row < 128 && col >= 0 && col < 128, "fragment bounds");
      require(row == Layout::row(tid, 0) + (i / 4) * Layout::row_group_stride + i % 4,
              "A float4 fragment matches output rows");
      require(col == Layout::col(tid, 0) + (i / 4) * Layout::col_group_stride + i % 4,
              "B float4 fragment matches output columns");
      if (i % 4 == 0) require(row % 4 == 0 && col % 4 == 0, "float4 alignment");
      if constexpr (Contiguous) {
        require(row == Layout::row(tid, 0) + i && col == Layout::col(tid, 0) + i,
                "contiguous 8x8 thread tile");
        const int warp = tid / 32, lane = tid % 32;
        if constexpr (LaneCols == 16) {
          require(row == (tid / 16) * 8 + i, "2x16 warp thread rows match row-major thread order");
          require(col == (tid % 16) * 8 + i, "2x16 warp thread columns match row-major thread order");
          require(row / 16 == warp, "each warp owns one 16-row output strip");
        } else {
          require(row == (warp / 2) * 32 + (lane / 8) * 8 + i, "4x8 warp thread rows");
          require(col == (warp % 2) * 64 + (lane % 8) * 8 + i, "4x8 warp thread columns");
        }
      } else {
        const int warp = tid / 32, lane = tid % 32;
        require(row == (warp / 4) * 64 + (lane / 4) * 4 + (i / 4) * 32 + i % 4,
                "original subtile rows preserved");
        require(col == (warp % 4) * 32 + (lane % 4) * 4 + (i / 4) * 16 + i % 4,
                "original subtile columns preserved");
      }
      for (int j = 0; j < 8; ++j) ++visits[row * 128 + Layout::col(tid, j)];
    }
  }
  for (int count : visits) require(count == 1, "each CTA output must be owned exactly once");
}

int main() {
  try {
    check_thread_layout<false>();
    check_thread_layout<true>();
    check_thread_layout<true, 16>();
    const auto shapes = default_shapes();
    require(shapes.size() == 9, "nine default shapes");
    require(shapes.back().M == 4096 && shapes.back().N == 4096 && shapes.back().K == 8192,
            "last default shape is 4096x4096x8192 in M,N,K order");
    for (auto s : shapes) {
      // BK=64 block tiles also satisfy BK=16 tiles and Split-K K%32.
      check_shape(s, 128, 128, 64);
      require(s.M <= 4096 && s.N <= 8192 && s.K <= 8192, "default shape bounds");
    }
    auto s = parse_shape("128x256x16");
    require(s.M == 128 && s.N == 256 && s.K == 16, "M,N,K order");
    for (const char* invalid : {"1,2", "1,2,3,4", "0,2,3", "1,-2,3", "1,2,3tail", "2147483648,2,3"})
      rejects([&] { parse_shape(invalid); });
    rejects([] { check_shape({127,128,16},128,128,16); });
    rejects([] { check_shape({128,128,8},128,128,16); });
    rejects([] { check_shape({1048576,128,16},16,16,1); });
    require(matrix_bytes<float>(4096,4096) == 67108864, "matrix byte count");
    rejects([] { matrix_bytes<float>(-1, 4); });
    const auto stats = summarize({4,1,3,2});
    require(stats.mean == 2.5 && stats.median == 2.5 && stats.minimum == 1 && stats.maximum == 4, "statistics");
    require(summarize({1,2,9}).median == 2, "odd median");
    rejects([] { summarize({}); });
    rejects([] { summarize({0}); });
    const std::vector<float> a{1,4,2,5,3,6}, b{7,8,9,10,11,12};
    require(reference_element(a,b,{2,2,3},0,0) == 58, "column-major A indexing");
    require(reference_element(a,b,{2,2,3},1,1) == 154, "row-major B indexing");
    require(compare({58,64,139,154},{58,64,139,154}).failures == 0, "exact result");
    require(compare({58,64,139,0},{58,64,139,154}).failures == 1, "missing write");
    require(compare({std::numeric_limits<float>::quiet_NaN()},{0}).failures == 1, "NaN rejection");
    require(compare({std::numeric_limits<float>::infinity()},{0}).failures == 1, "infinity rejection");
    rejects([] { compare({1},{}); });
    std::cout << "Host tests passed: shapes, limits, statistics, thread layouts, accuracy.\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
