# Standalone CUDA GEMM 优化链路

纯 C++17 / CUDA 核心项目，包含 19 个手写 FP32 GEMM 变体和一个严格 FP32 cuBLAS 对照。编译和测试不依赖 PyTorch、Python、LibTorch 或 CUTLASS；结果可视化是可选的 Python 后处理。

这些实现复制自 `cuda-ai-infra-kernels/csrc/gemm`。保留手写 kernel 的计算和布局映射，将 tensor 接口、分配和 stream 获取替换成独立 C++ host 接口。同步 thread tile / warp tile 共用一个 kernel 模板，并声明与启动线程数一致的 `__launch_bounds__`，约束编译器寄存器分配。原仓库未修改；Tensor Core/CUTLASS 不纳入此次复制。

## 一键运行

需要 Linux/Windows CUDA 开发环境、NVIDIA SM80+ GPU、CUDA Toolkit 11+（含 nvcc/cuBLAS）、CMake 3.18+、支持 C++17 的 host 编译器。以下脚本使用 Bash：

```bash
bash scripts/bench_gemm_all.sh
```

依次完成 CMake 配置、编译、CPU/GPU 回归测试、全量正确性校验和性能测试。默认目标是 `80`：包含 SM80 原生代码和 PTX，新架构可 JIT。建议按目标 GPU 指定原生架构，例如：

```bash
# A800 / A100
CUDA_ARCHITECTURES=80 bash scripts/bench_gemm_all.sh
# RTX 4090（toolkit 需支持该架构）
CUDA_ARCHITECTURES=89 bash scripts/bench_gemm_all.sh
```

可设置 `BUILD_JOBS` 控制编译并行数、`GEMM_BUILD_DIR` 指定构建目录。脚本的回归测试默认使用 GPU 0；benchmark 可用 `--device` 选择设备。

手动构建及运行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
./build/gemm_bench
```

## 默认规模和实现

规模顺序为 **M×N×K**，与原参考热力图一致：

| M | N | K |
| ---: | ---: | ---: |
| 256 | 256 | 256 |
| 1024 | 1024 | 1024 |
| 2048 | 2048 | 2048 |
| 4096 | 4096 | 2048 |
| 4096 | 1024 | 4096 |
| 1024 | 4096 | 4096 |
| 4096 | 4096 | 4096 |

每种规模运行下面全部 14 个实现，共 **98 个组合**。

| 阶段 | impl | block tile BM×BN×BK | 每线程输出 |
| --- | --- | --- | --- |
| 1 | `naive` | 16×16×1 | 一个元素，无 smem |
| 2 | `block_tile` | 16×16×8 | 一个元素，smem 标量搬运 |
| 3 | `block_tile_float4_16x16` | 16×16×8 | 一个元素，float4 搬运 |
| 3 | `block_tile_float4_16x32` | 16×32×8 | 一个元素，float4 搬运 |
| 3 | `block_tile_float4_32x32` | 32×32×8 | 一个元素，float4 搬运 |
| 4 | `thread_tile_4x4` | 128×128×8 | 连续 4×4 |
| 4 | `thread_tile_4x8` | 128×128×8 | 连续 4×8 |
| 4 | `thread_tile_8x8` | 128×128×8 | 连续 8×8 |
| 4 | `thread_tile_8x8_warp4x8` | 128×128×8 | 连续 8×8，warp 内线程按 4×8 排列 |
| 5 | `warp_tile` | 128×128×8 | 四个 4×4 子块 |
| 6 | `prefetch_smem` | 128×128×8 | smem 双缓冲 |
| 7 | `prefetch_reg` | 128×128×8 | smem + register 双缓冲 |
| 8 | `split_k` | 128×128×8 | smem + register 双缓冲，两路 K，atomicAdd |
| 9 | `cublas` | 库选择 | 严格 FP32 |

第三阶段严格保持一线程一个 C 元素，所以最大的 block tile 是 32×32（1024 个线程）。第四阶段才扩展 block tile 并引入连续 thread tile。第五阶段沿用后面三个 kernel 的 warp/lane/subtile 映射。

`thread_tile_8x8_warp4x8` 与原版连续 `thread_tile_8x8` 共用计算、搬运、同步和写回代码，仅改变线程输出坐标映射。原版每个 warp 的线程按 2×16 排列，覆盖 16×128；新版按 4×8 排列，覆盖 32×64，block 内的 8 个 warp 按 4×2 排列。两版均为 256 线程、8 KiB shared memory，每线程 64 个累加器，使用相同 launch bounds；最终寄存器数和性能需实测。全量 benchmark、CUDA 回归测试及绘图均包含新版。

`warp_tile` 也实例化同一个 `include/gemm/gemm_thread_tile.cuh` 模板：与 `thread_tile_8x8` 的唯一算法差异是编译期布局类型，分别为 `ThreadTileLayout8x8<false>` 和 `ThreadTileLayout8x8<true, 16>`。前者保留每线程四个 4×4 子块、warp 覆盖 64×32 的原映射；后者保留连续 8×8、warp 覆盖 16×128 的原映射。两者的 global/shared 搬运、A 后 B 的 float4 加载顺序、累加器初始化、乘加循环、同步、写回循环、`__restrict__`、`__launch_bounds__(256, 1)` 和 host 启动/错误检查路径全部共用。布局坐标会影响编译器生成的指令与寄存器分配，因此不保证机器码或寄存器数相同。重构后的性能应重新测量，不与旧版独立 warp kernel 的结果混用。

第 6～8 阶段仅测试原 subtile 布局的 `prefetch_smem`、`prefetch_reg` 和 `split_k`。六个连续 8×8 变体已从实现注册表移除，不再参与 benchmark、CUDA 正确性回归或 NCU 采集，也不再接受对应的 `--impl` 名称。连续变体的 kernel 源码和直接调用接口仍保留。第 4 阶段的两个连续 8×8 thread tile 实现继续测试。

三个 Split-K 变体现在同时使用 shared-memory 和 register 双缓冲。每个 BK=8 tile 先将 k=0 读入 `rA[0][8]` / `rB[0][8]`，循环中先加载 k+1 到另一组寄存器，再计算当前 k，最后单独计算 k=7。各 K 分片的第一个 tile 和最后一个 tile 均使用相同流程；两路划分、atomicAdd 和同 stream 清零不变。该版本从 `prefetch_reg` 的计算流水继续添加 Split-K；旧 CSV 中的 Split-K 没有 register 双缓冲，比较前后版本时请分别保存报告。

第四阶段的 4×4 / 4×8 / 8×8 变体分别启动 1024 / 512 / 256 个线程。模板通过 `__launch_bounds__((BM / TM) * (BN / TN), 1)` 告知编译器真实的线程数与每 SM 至少一个 block 的资源目标。尤其 A800/SM80 每 block 最多 65536 个 32-bit 寄存器，1024 线程的变体不能超过每线程 64 个寄存器；没有该约束时可能编译成功但启动报 `too many resources requested for launch`。编译器可能通过重排或 spill 满足约束，具体性能需实测。用 `cmake -S . -B build -DGEMM_PTXAS_VERBOSE=ON` 开启寄存器和 spill 报告。

所有输入均为 FP32，**A 为列主序 `[M,K]`，B/C 为行主序**。物理下标：

```cpp
A[k * M + m]
B[k * N + n]
C[m * N + n]
```

只支持正数且完整的 tile。全量运行要求 M/N 是 128 的倍数，K 是 16 的倍数；单独运行只需满足对应实现的倍数，Split-K 仍要求 `K % 16 == 0`。float4 路径要求 A/B/C 起始地址 16 字节对齐。host 检查尺寸、空指针、对齐和 grid.y 上限；无隐式布局转换、边界填充或回退。

## C++ host 直接调用

接口位于 `include/gemm/gemm.h`，库目标为 `gemm_kernels`。缓冲区由调用方分配、复用并释放。调用是异步的，不在函数内部同步，也不在计时路径分配显存。

```cpp
#include "cuda_resources.h"

int main() {
  CUDA_CHECK(cudaSetDevice(0));
  constexpr int M = 128, N = 256, K = 32;
  Stream stream;
  BlasHandle blas;
  DeviceBuffer<float> a(M * K), b(K * N), c(M * N);
  // 示例初始化为 0；实际使用中按列主序 A、行主序 B 拷贝输入。
  CUDA_CHECK(cudaMemsetAsync(a.get(), 0, M * K * sizeof(float), stream.value));
  CUDA_CHECK(cudaMemsetAsync(b.get(), 0, K * N * sizeof(float), stream.value));
  GemmArgs args{a.get(), b.get(), c.get(), M, N, K, stream.value, blas.value};
  gemm_thread_tile_8x8(args);
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  gemm_cublas(args);  // 同一份输入，覆盖 C
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
}
```

调用者保证当前设备、stream、cuBLAS handle 和 device 指针属于同一设备；A/B/C 不互相重叠。原始指针没有 stride 元数据，因此布局是接口约定。cuBLAS 入口将传入 handle 绑定到指定 stream 和 host pointer mode；并发调用请使用独立 handle。

`gemm_splictK()` 保留原函数/文件命名，连续版本入口为 `gemm_splictK_contiguous8x8()` 和 `gemm_splictK_contiguous8x8_warp2x16()`。三版每次调用均在同一 stream 上 `cudaMemsetAsync(C,0)` 后启动 kernel。清零包含在普通计时和 CUDA Graph 中，不由调用方额外处理。

## 测试和计时

```bash
# 快速验证全部实现
./build/gemm_bench --shapes 128,256,16 256,128,48 --check-only
# 自定义问题规模、预热与重复次数
./build/gemm_bench --shapes 1024,1024,1024 4096,4096,4096 \
  --warmup 20 --repeat 100 --batch 10 --output results/run.csv
# 单独测试一个变体，仍自动加入 cuBLAS 对照
./build/gemm_bench --impl thread_tile_4x8 --shapes 128,256,24
./build/gemm_bench --impl thread_tile_8x8_warp4x8 --shapes 1024,1024,1024
# 不使用 CUDA Graph
./build/gemm_bench --timing event --shapes 1024,1024,1024
./build/gemm_bench --list
./build/gemm_bench --help
```

每个规模只创建一份随机输入（均匀分布 [-1,1]），所有实现共用相同 device 指针。校验与分配、H2D/D2H 拷贝均在计时区间之外：

- 当 `M*N*K <= 16*1024*1024` 时，使用 CPU FP64 全量参考。
- 更大规模用 cuBLAS DGEMM 得到 FP64 全量参考，并用 32 个 CPU FP64 点积检查其下标/数值。输入从原始 FP32 转成 FP64。
- 全部输出按 `abs_error <= 1e-3 + 1e-3 * abs(reference)` 校验，并明确拒绝 NaN/Inf。
- 校验前将输出填成 NaN，连续运行两次，检查遗漏写入和 Split-K 旧值累加。当前规模的全部实现通过后才开始计时；计时结束也检查最终输出。
- GPU 回归额外覆盖各变体最小 tile、矩形多 tile、单/多 K tile、非默认 stream、零 K/地址对齐拒绝和 graph/event 重复执行。

默认使用 CUDA Graph：在显式非默认 stream 上预热，捕获 `batch=10` 次调用，每次 replay 的 CUDA events 耗时除以 batch，采样 `repeat=50` 轮；原生 event 模式则在两个 events 之间直接 launch batch 次。计时都包含 Split-K 的清零。固定输入/地址反复测量，属于 steady-state 缓存场景。cuBLAS 用 `CUBLAS_COMPUTE_32F_PEDANTIC`，不使用 TF32。

终端显示 median ms、TFLOPS、相对 cuBLAS 和 naive 的加速比。CSV 记录 mean/median/min/max、误差、形状、计时配置、设备/CUDA/cuBLAS 版本和布局，默认保存至 `results/gemm_<timestamp>.csv`。加速比采用同尺寸基线的 **median_ms / 当前 median_ms**。只选某一实现时不额外运行 naive，该比值显示 n/a；仅校验模式不填性能字段。失败会返回非零退出码。

CSV 保留 `speedup_over_subtile` 字段，当前测试集合没有连续预取/Split-K 变体，因此该字段留空，终端的 `vs subtile` 显示 n/a。绘图工具仍支持旧 CSV 中的连续布局结果和配对加速比，方便查看历史报告。

```bash
compute-sanitizer --tool memcheck ./build/gemm_cuda_tests
compute-sanitizer --tool racecheck ./build/gemm_cuda_tests
```

## 使用 Nsight Compute 定位性能瓶颈

需要可用的 `ncu` CLI 和 NVIDIA GPU。脚本使用 Python 标准库，不需要 PyTorch 或绘图依赖：

```bash
# 默认编译、运行 CTest，再测注册表中的全部 14 个实现；默认尺寸 4096³
python3 scripts/profile_gemm_all.py --arch 80
# 多个尺寸，输出到一个新的目录
python3 scripts/profile_gemm_all.py --arch 80 \
  --shapes 1024,1024,1024 4096,4096,4096 --output results/ncu_layout_comparison
# 只分析同步加载的三个布局
python3 scripts/profile_gemm_all.py --skip-build \
  --impls thread_tile_8x8 thread_tile_8x8_warp4x8 warp_tile
```

`--build-dir`（或 `GEMM_BUILD_DIR`）、`--jobs`、`--ncu`（或 `NCU`）、`--device`、`--warmup`、`--seed` 均可设置。构建默认开启 `GEMM_PTXAS_VERBOSE`，寄存器数和 spill 信息保存在 `build.log`。`--skip-build` 只复用已经重新构建的 `gemm_profile`；脚本不会自动判断旧可执行文件是否包含最新修改。

每个实现/尺寸启动一个独立的 `gemm_profile` 进程，按以下顺序运行：

1. 构造列主序 A、行主序 B，使用严格 FP32 cuBLAS 全量参考和 CPU FP64 抽样检查。
2. 验证目标输出、预热并同步。这些工作在采集范围之外。
3. `cudaProfilerStart()` → **仅调用指定实现一次** → 同步 → `cudaProfilerStop()`。
4. 检查采集后的输出。

NCU 使用 `--profile-from-start off`，不会采集初始化、reference 或 warmup。专用入口不自动加入其他布局/cuBLAS 对照；脚本为每个注册实现独立建报告。没有按 kernel 名过滤，也不设置 `--launch-count 1`，因此既能区分不同源文件内同名的 `gemm_kernel`，也保留 cuBLAS 单次调用可能启动的多个 device kernel。

默认 `--set full` 收集可用的 Launch Statistics、Occupancy、Memory Workload、Scheduler、Warp State 等 section。重点对比 registers/thread、spill/local-memory 流量、active warps、shared-memory bank conflicts/wavefronts，以及 long/short scoreboard、barrier stall；实际指标依 NCU 版本/GPU 而定，缺失指标不伪造为零。

默认 `--replay-mode kernel --cache-control all --clock-control base`。可以通过脚本参数 `--cache-control none --clock-control none` 改为保留缓存及现有时钟策略。NCU 指标来自 replay，并可能采用与普通 benchmark 不同的缓存/时钟条件，不能直接把 duration 与 graph benchmark 耗时混为一谈。Split-K 的清零在采集区间内发出，但 **NCU 单 kernel duration 不包含非 kernel 的 memset 操作**；包含清零的端到端比较仍使用 `gemm_bench`。

输出默认位于新的 `results/ncu_<时间>_<id>/`：

- `<impl>__<M>x<N>x<K>.ncu-rep`：可用 `ncu-ui` 打开，包含源码关联。
- 同名前缀的 `.metrics.csv` 和 `.log`：全部 raw metrics 及执行日志。
- `metrics_all.csv`：汇总所有成功报告，附实现名、尺寸和报告路径；一个实现内部的多个 kernel 保留各自的 NCU ID/名称。
- `manifest.json`：命令、源码哈希、NCU 版本、采集设置和逐项成功/失败状态。
- `configure.log`、`build.log`、`tests.log`：未使用 `--skip-build` 时生成。

单项采集失败时保留日志并继续其他实现，最终返回非零；没有报告或没有指标也视为失败。指定的 `--output` 必须是新目录，以免覆盖既有测量。

可手动采集单个实现：

```bash
ncu --profile-from-start off --set full --import-source yes \
  --export results/warp_tile_profile \
  ./build/gemm_profile --impl warp_tile --shape 4096,4096,4096 --warmup 10
```

无需 GPU 的脚本回归：

```bash
python3 -m unittest discover -s tests -p test_profile_gemm_all.py -v
```

## 收集结果并可视化

首次安装绘图依赖（Python 3.9+，无需 GPU 或 PyTorch）：

```bash
python3 -m pip install -r requirements-plot.txt
```

一键构建、回归测试、benchmark 和绘图：

```bash
bash scripts/bench_and_plot.sh
# 也支持 benchmark 参数
CUDA_ARCHITECTURES=80 bash scripts/bench_and_plot.sh \
  --shapes 256,256,256 1024,1024,1024 --repeat 50
```

每次创建独立的 `results/run_<时间>_<id>/`，保存 C++ 输出 `results.csv`、完整构建/CTest/benchmark 日志 `run.log`、绘图日志 `plot.log` 和 `plots/`。任何构建或测试失败都会保留日志并返回非零退出码，不继续生成成功报告。包装脚本管理 `--output`，不接受 `--check-only`；纯测试仍使用原来的 `bench_gemm_all.sh`。

**已有结果无需重跑 GPU**，直接收集绘图：

```bash
# 递归收集 results 下的原始 CSV
python3 scripts/plot_results.py --input results --output results/report
# 只画最新一份，适合查看刚完成的一次运行
python3 scripts/plot_results.py --input results --latest --output results/latest_report
# 指定单次或多次运行
python3 scripts/plot_results.py --input results/run_a.csv results/run_b.csv \
  --output results/comparison --formats png svg pdf
```

也可以把 CSV 拷贝到本地电脑再运行上述命令。打开输出目录中的 `index.html` 浏览报告；单一环境的具体报告为 `report.html`。

| 产物 | 内容 |
| --- | --- |
| `speedup_by_shape.png` | 相对同尺寸 naive 的加速比热力图，与参考图一样逐尺寸、逐实现展示；1× 居中，低于 1× 红色，高于 1× 绿色，对数色阶 |
| `cublas_speedup_by_shape.png` | 相对严格 FP32 cuBLAS 的加速比；大于 1× 表示快于 cuBLAS |
| `tflops_by_stage.png` | 已测实现沿优化链路的吞吐量曲线，每条线对应一种问题规模 |
| `latency_by_shape.png` | 每个实现/规模的 median ms 热力图 |
| `report.html` | 汇总图表、每种规模最快的已测手写 kernel、数据来源和统计说明 |
| `summary.csv` | 每个实现/规模的耗时、TFLOPS、加速比及样本数 |
| `collected_results.csv` | 通过校验的原始计时行，附 `source_csv` 来源路径 |
| `manifest.json` | 环境分组、来源文件、统计方法、缺失或无效行的原因 |

默认同时导出 PNG/SVG；`--formats png svg pdf` 可增加 PDF。HTML 使用 PNG，所以无论选择何种格式均会生成 PNG。图表使用英文标签和随 Matplotlib 安装的字体，避免服务器缺少中文字体。

统计规则：

- 仅使用 `verified=true`、耗时有限且大于 0 的记录。`check_only`、失败/不完整行、NaN/Inf 及未包含清零的 Split-K 行不参与性能图，排除原因写入 manifest。
- 加速比重新用 **同一 CSV、同一 M/N/K** 的基线 `median_ms / 当前 median_ms` 计算，不跨运行借用基线。缺失实现或基线显示 N/A，不按 0 补齐。
- 多次可比运行合并时，耗时取各次 `median_ms` 的中位数；加速比取各次配对比值的中位数。`runs`、`naive_pairs`、`cublas_pairs` 分别记录有效样本数。TFLOPS 用汇总耗时重新计算。
- GPU、计算能力、CUDA driver/runtime、cuBLAS、graph/event、warmup/repeat/batch、seed、A 布局任一不同都会分组到 `group_01/` 等子目录，`index.html` 链接本次全部分组。
- 现有 CSV 不记录源码版本、GPU 时钟或并发负载；比较代码修改前后结果时请分别出图，不将它们作为同一次实验的重复样本合并。
- 第 3/4 阶段的不同 tile 变体单独展示；性能不保证随着阶段序号增加而提高。

脚本回归测试（无需 CUDA）：

```bash
python3 -m unittest discover -s tests -p test_plot_results.py -v
```

## 无 CUDA 环境的 host 检查

```bash
cmake -S . -B build-host -DGEMM_HOST_TESTS_ONLY=ON
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

这仅检查 C++ 尺寸解析、约束、统计、参考布局下标和误差判断，不编译或执行 CUDA kernel。创建项目时已在 macOS 通过此检查；完整 CUDA 编译、GPU 正确性与性能仍需在 NVIDIA 环境验证。

## 文件结构

```text
CMakeLists.txt
include/bench_utils.h          纯 C++ 尺寸/统计/校验
include/cuda_resources.h       显存、stream、event、handle、graph 的 RAII
include/gemm/gemm.h            device 指针接口及 host 参数检查
include/gemm/gemm_regular.cuh  同步 float4 搬运
include/gemm/gemm_thread_tile.cuh  同步 thread/warp tile 共用 kernel 和启动器
include/gemm/thread_tile_layout.h  连续/subtile 的编译期坐标映射
src/gemm/                     独立 kernel、cuBLAS、实现注册表
src/benchmark.cpp             C++ host 驱动和 CSV 输出
src/profile.cpp               仅采集指定实现的 CUDA profiler API 范围
scripts/bench_gemm_all.sh      一键构建、回归、测试
scripts/bench_and_plot.sh      测试、收集日志、绘图
scripts/plot_results.py        CSV 收集与 Python 可视化
scripts/profile_gemm_all.py    全部实现的 NCU 采集与指标汇总
requirements-plot.txt          可选绘图依赖
tests/                       C++ host / CUDA 回归
```
