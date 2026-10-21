#include "cuda_resources.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
struct Options {
  std::vector<Shape> shapes = default_shapes();
  int warmup = 10, repeat = 50, batch = 10, device = 0, seed = 0;
  bool graph = true, check_only = false;
  std::string impl = "all", output;
};

void usage() {
  std::cout << "gemm_bench [--shapes M,N,K ...] [--impl NAME|all]\n"
            << "  --warmup 10 --repeat 50 --batch 10 --timing graph|event\n"
            << "  --device 0 --seed 0 --output results.csv --check-only\n"
            << "Defaults: 7 shapes x " << gemm_kernels().size()
            << " implementations; A column-major, B/C row-major.\n"
            << "Rates/speedups use median GPU ms; Split-K zeroing is included.\n";
}
Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    const std::string option = argv[i];
    if (option == "--check-only") { o.check_only = true; continue; }
    if (i + 1 == argc) throw std::invalid_argument("missing value for " + option);
    if (option == "--shapes") {
      o.shapes.clear();
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
        o.shapes.push_back(parse_shape(argv[++i]));
      if (o.shapes.empty()) throw std::invalid_argument("--shapes requires at least one shape");
      continue;
    }
    const std::string value = argv[++i];
    if (option == "--warmup") o.warmup = parse_integer(value, 0);
    else if (option == "--repeat") o.repeat = parse_integer(value);
    else if (option == "--batch") o.batch = parse_integer(value);
    else if (option == "--device") o.device = parse_integer(value, 0);
    else if (option == "--seed") o.seed = parse_integer(value, 0);
    else if (option == "--impl") o.impl = value;
    else if (option == "--output") o.output = value;
    else if (option == "--timing") {
      if (value != "graph" && value != "event") throw std::invalid_argument("timing must be graph or event");
      o.graph = value == "graph";
    } else throw std::invalid_argument("unknown option: " + option);
  }
  return o;
}

std::vector<double> reference(const std::vector<float>& a, const std::vector<float>& b,
                              Shape s, cudaStream_t stream, cublasHandle_t handle) {
  const auto count = matrix_bytes<double>(s.M, s.N) / sizeof(double);
  std::vector<double> expected(count);
  if (static_cast<double>(s.M) * s.N * s.K <= 16 * 1024 * 1024) {
    for (int row = 0; row < s.M; row++)
      for (int col = 0; col < s.N; col++)
        expected[static_cast<std::size_t>(row) * s.N + col] = reference_element(a, b, s, row, col);
    return expected;
  }
  // Large shapes: full FP64 GPU reference, plus independent CPU dot products.
  DeviceBuffer<double> da(a.size()), db(b.size()), dc(count);
  const std::vector<double> ha(a.begin(), a.end()), hb(b.begin(), b.end());
  CUDA_CHECK(cudaMemcpyAsync(da.get(), ha.data(), ha.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(db.get(), hb.data(), hb.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  const double alpha = 1, beta = 0;
  CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, s.N, s.M, s.K,
      &alpha, db.get(), s.N, da.get(), s.M, &beta, dc.get(), s.N));
  CUDA_CHECK(cudaMemcpyAsync(expected.data(), dc.get(), count * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < 32; i++) {
    const int row = static_cast<int>(int64_t(i) * (s.M - 1) / 31);
    const int col = static_cast<int>(int64_t((i * 13) % 32) * (s.N - 1) / 31);
    const double cpu = reference_element(a, b, s, row, col);
    const double gpu = expected[static_cast<std::size_t>(row) * s.N + col];
    if (!std::isfinite(gpu) || std::abs(cpu - gpu) > 1e-9 * (1 + std::abs(cpu)))
      throw std::runtime_error("FP64 reference disagrees with independent CPU dot product");
  }
  return expected;
}

Accuracy validate(const Kernel& kernel, const GemmArgs& args, const std::vector<double>& expected) {
  const auto bytes = matrix_bytes<float>(args.M, args.N);
  std::vector<float> output(bytes / sizeof(float));
  // NaN poison exposes missing output writes. A second call catches stale
  // accumulation, especially forgetting to reset Split-K's output.
  CUDA_CHECK(cudaMemsetAsync(args.C, 0xff, bytes, args.stream));
  Accuracy total;
  for (int iteration = 0; iteration < 2; iteration++) {
    kernel.launch(args);
    CUDA_CHECK(cudaMemcpyAsync(output.data(), args.C, bytes, cudaMemcpyDeviceToHost, args.stream));
    CUDA_CHECK(cudaStreamSynchronize(args.stream));
    const auto accuracy = compare(output, expected);
    if (accuracy.failures)
      throw std::runtime_error(std::string(kernel.name) + ": " + std::to_string(accuracy.failures) +
          " incorrect elements, max_abs_error=" + std::to_string(accuracy.max_abs_error));
    total.max_abs_error = std::max(total.max_abs_error, accuracy.max_abs_error);
  }
  return total;
}

Stats benchmark(const Kernel& kernel, const GemmArgs& args, const Options& o,
                const std::vector<double>& expected) {
  for (int i = 0; i < std::max(1, o.warmup); i++) kernel.launch(args);
  CUDA_CHECK(cudaStreamSynchronize(args.stream));
  Graph graph;
  if (o.graph) {
    CUDA_CHECK(cudaStreamBeginCapture(args.stream, cudaStreamCaptureModeGlobal));
    try {
      for (int i = 0; i < o.batch; i++) kernel.launch(args);
    } catch (...) {
      // End even an invalidated capture so the stream can be safely destroyed.
      cudaStreamEndCapture(args.stream, &graph.graph);
      throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(args.stream, &graph.graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(graph.executable, args.stream));
    CUDA_CHECK(cudaStreamSynchronize(args.stream));
  }
  Event start, end;
  std::vector<double> samples;
  samples.reserve(o.repeat);
  for (int r = 0; r < o.repeat; r++) {
    CUDA_CHECK(cudaEventRecord(start.value, args.stream));
    if (o.graph) {
      CUDA_CHECK(cudaGraphLaunch(graph.executable, args.stream));
    } else {
      for (int i = 0; i < o.batch; i++) kernel.launch(args);
    }
    CUDA_CHECK(cudaEventRecord(end.value, args.stream));
    CUDA_CHECK(cudaEventSynchronize(end.value));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, end.value));
    samples.push_back(static_cast<double>(ms) / o.batch);
  }
  // Verify the actual replay/timing path as well, outside the measured region.
  std::vector<float> output(expected.size());
  CUDA_CHECK(cudaMemcpyAsync(output.data(), args.C, output.size() * sizeof(float), cudaMemcpyDeviceToHost, args.stream));
  CUDA_CHECK(cudaStreamSynchronize(args.stream));
  if (compare(output, expected).failures)
    throw std::runtime_error(std::string(kernel.name) + ": incorrect output after timing/replay");
  return summarize(samples);
}

struct Result { const Kernel* kernel; Accuracy accuracy; Stats stats{}; };
std::string csv_string(const std::string& value) {
  std::string result = "\"";
  for (char ch : value) { if (ch == '"') result += '"'; result += ch; }
  return result + '"';
}
} // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; i++) {
      if (std::string(argv[i]) == "--help") { usage(); return 0; }
      if (std::string(argv[i]) == "--list") {
        for (const auto& k : gemm_kernels()) std::cout << k.stage << " " << k.name << '\n';
        return 0;
      }
    }
    const Options o = parse_options(argc, argv);
    std::string subtile_baseline;
    for (const auto& k : gemm_kernels())
      if (o.impl == k.name && k.subtile_baseline) subtile_baseline = k.subtile_baseline;
    std::vector<const Kernel*> selected;
    for (const auto& k : gemm_kernels())
      if (o.impl == "all" || o.impl == k.name || subtile_baseline == k.name ||
          (k.subtile_baseline && subtile_baseline == k.subtile_baseline) ||
          std::string(k.name) == "cublas") selected.push_back(&k);
    if (o.impl != "all" && o.impl != "cublas" && selected.size() == 1)
      throw std::invalid_argument("unknown implementation: " + o.impl);
    int implementation_width = 29;
    for (const auto* k : selected)
      implementation_width = std::max(implementation_width, static_cast<int>(std::string(k->name).size()) + 2);
    for (Shape s : o.shapes)
      for (const auto* k : selected) check_shape(s, k->BM, k->BN, k->BK);

    CUDA_CHECK(cudaSetDevice(o.device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, o.device));
    if (properties.major < 8) throw std::runtime_error("this project requires an SM80+ GPU");
    int driver = 0, runtime = 0, blas_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    Stream stream;
    BlasHandle blas;
    CUBLAS_CHECK(cublasGetVersion(blas.value, &blas_version));
    CUBLAS_CHECK(cublasSetStream(blas.value, stream.value));
    CUBLAS_CHECK(cublasSetPointerMode(blas.value, CUBLAS_POINTER_MODE_HOST));

    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::filesystem::path path = o.output.empty()
        ? std::filesystem::path("results") / ("gemm_" + std::to_string(stamp) + ".csv")
        : std::filesystem::path(o.output);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream csv(path);
    if (!csv) throw std::runtime_error("cannot write " + path.string());
    csv << "stage,impl,M,N,K,verified,max_abs_error,mean_ms,median_ms,min_ms,max_ms,tflops,"
           "speedup_over_cublas,speedup_over_naive,timing,warmup,repeat,batch,seed,gpu,compute_capability,"
           "driver_version,runtime_version,cublas_version,a_layout,includes_output_clear,speedup_over_subtile\n";
    csv << std::setprecision(12);
    std::cout << properties.name << " | " << o.shapes.size() << " shapes x " << selected.size()
              << " implementations | " << (o.check_only ? "correctness only" : (o.graph ? "CUDA graph" : "CUDA events"))
              << "\nA column-major; B/C row-major; cuBLAS strict FP32.\n";
    std::mt19937 rng(o.seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (Shape s : o.shapes) {
      std::cout << "\nM=" << s.M << " N=" << s.N << " K=" << s.K << std::endl;
      const auto ac = matrix_bytes<float>(s.M, s.K) / sizeof(float);
      const auto bc = matrix_bytes<float>(s.K, s.N) / sizeof(float);
      const auto cc = matrix_bytes<float>(s.M, s.N) / sizeof(float);
      std::vector<float> a(ac), b(bc);
      for (float& x : a) x = distribution(rng);
      for (float& x : b) x = distribution(rng);
      DeviceBuffer<float> da(ac), db(bc), dc(cc);
      CUDA_CHECK(cudaMemcpyAsync(da.get(), a.data(), ac * sizeof(float), cudaMemcpyHostToDevice, stream.value));
      CUDA_CHECK(cudaMemcpyAsync(db.get(), b.data(), bc * sizeof(float), cudaMemcpyHostToDevice, stream.value));
      const auto expected = reference(a, b, s, stream.value, blas.value);
      const GemmArgs args{da.get(), db.get(), dc.get(), s.M, s.N, s.K, stream.value, blas.value};
      std::vector<Result> rows;
      for (const auto* kernel : selected) {
        std::cout << "  CHECK " << kernel->name << std::endl;
        rows.push_back({kernel, validate(*kernel, args, expected), {}});
        std::cout << "  PASS " << kernel->name << std::endl;
      }
      // All variants pass before any performance numbers are generated.
      if (!o.check_only)
        for (auto& row : rows) row.stats = benchmark(*row.kernel, args, o, expected);
      double cublas_ms = 0, naive_ms = 0;
      for (const auto& row : rows) {
        if (std::string(row.kernel->name) == "cublas") cublas_ms = row.stats.median;
        if (std::string(row.kernel->name) == "naive") naive_ms = row.stats.median;
      }
      if (!o.check_only)
        std::cout << std::left << std::setw(implementation_width) << "implementation" << std::right << std::setw(12)
                  << "median ms" << std::setw(12) << "TFLOPS" << std::setw(12) << "vs cuBLAS" << std::setw(12) << "vs naive"
                  << std::setw(12) << "vs subtile" << '\n';
      for (const auto& row : rows) {
        const auto& t = row.stats;
        const double flops = o.check_only ? 0 : 2.0 * s.M * s.N * s.K / (t.median * 1e9);
        const double ratio = o.check_only ? 0 : cublas_ms / t.median;
        const double naive_ratio = o.check_only ? 0 : naive_ms / t.median;
        double subtile_ratio = 0;
        if (!o.check_only && row.kernel->subtile_baseline)
          for (const auto& baseline : rows)
            if (std::string(baseline.kernel->name) == row.kernel->subtile_baseline)
              subtile_ratio = baseline.stats.median / t.median;
        if (!o.check_only) {
          std::cout << std::left << std::setw(implementation_width) << row.kernel->name << std::right << std::fixed
                    << std::setprecision(4) << std::setw(12) << t.median << std::setw(12) << flops
                    << std::setw(12) << ratio;
          if (naive_ms) std::cout << std::setw(12) << naive_ratio;
          else std::cout << std::setw(12) << "n/a";
          if (subtile_ratio) std::cout << std::setw(12) << subtile_ratio;
          else std::cout << std::setw(12) << "n/a";
          std::cout << '\n';
        }
        csv << row.kernel->stage << ',' << row.kernel->name << ',' << s.M << ',' << s.N << ',' << s.K
            << ",true," << row.accuracy.max_abs_error << ',';
        if (o.check_only) csv << ",,,,,,";
        else {
          csv << t.mean << ',' << t.median << ',' << t.minimum << ',' << t.maximum << ','
              << flops << ',' << ratio << ',';
          if (naive_ms) csv << naive_ratio;
        }
        csv << ',' << (o.check_only ? "check_only" : (o.graph ? "graph" : "event")) << ','
            << o.warmup << ',' << o.repeat << ',' << o.batch << ',' << o.seed << ','
            << csv_string(properties.name) << ',' << properties.major << '.' << properties.minor << ','
            << driver << ',' << runtime << ',' << blas_version << ",column_major,"
            << (row.kernel->stage == 8 ? "true" : "false") << ',';
        if (subtile_ratio) csv << subtile_ratio;
        csv << '\n';
      }
      csv.flush();
      if (!csv) throw std::runtime_error("failed writing " + path.string());
    }
    std::cout << "\nSaved: " << path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
