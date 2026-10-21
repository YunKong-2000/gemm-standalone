#include "cuda_resources.h"
#include <cuda_profiler_api.h>

#include <iostream>
#include <random>

// One implementation per process. The profiling range contains just the chosen
// API call (and its synchronization), never reference GEMM or warmup launches.
int main(int argc, char** argv) {
  bool profiling = false;
  try {
    std::string impl;
    Shape shape{4096, 8192, 4096};
    int warmup = 2, device = 0, seed = 0;
    bool validate_only = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--list") {
        for (const auto& kernel : gemm_kernels()) std::cout << kernel.name << '\n';
        return 0;
      }
      if (arg == "--help") {
        std::cout << "gemm_profile --impl NAME [--shape M,N,K] [--warmup 2] "
                     "[--device 0] [--seed 0] [--validate-only]\n"
                     "Default shape: 4096,8192,4096 (M,N,K).\n"
                     "Use ncu --profile-from-start off to collect only the selected call.\n";
        return 0;
      }
      if (arg == "--validate-only") { validate_only = true; continue; }
      if (++i == argc) throw std::invalid_argument("missing value for " + arg);
      const std::string value = argv[i];
      if (arg == "--impl") impl = value;
      else if (arg == "--shape") shape = parse_shape(value);
      else if (arg == "--warmup") warmup = parse_integer(value, 0);
      else if (arg == "--device") device = parse_integer(value, 0);
      else if (arg == "--seed") seed = parse_integer(value, 0);
      else throw std::invalid_argument("unknown option: " + arg);
    }
    const Kernel* selected = nullptr;
    for (const auto& kernel : gemm_kernels()) if (impl == kernel.name) selected = &kernel;
    if (!selected) throw std::invalid_argument("--impl must name exactly one registered implementation");
    check_shape(shape, selected->BM, selected->BN, selected->BK);
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 8) throw std::runtime_error("SM80+ required");

    Stream stream;
    BlasHandle blas;
    const auto ac = matrix_bytes<float>(shape.M, shape.K) / sizeof(float);
    const auto bc = matrix_bytes<float>(shape.K, shape.N) / sizeof(float);
    const auto cc = matrix_bytes<float>(shape.M, shape.N) / sizeof(float);
    std::vector<float> a(ac), b(bc), expected(cc), output(cc);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float& v : a) v = distribution(rng);
    for (float& v : b) v = distribution(rng);
    DeviceBuffer<float> da(ac), db(bc), dc(cc), reference(cc);
    CUDA_CHECK(cudaMemcpyAsync(da.get(), a.data(), ac * sizeof(float), cudaMemcpyHostToDevice, stream.value));
    CUDA_CHECK(cudaMemcpyAsync(db.get(), b.data(), bc * sizeof(float), cudaMemcpyHostToDevice, stream.value));
    GemmArgs args{da.get(), db.get(), reference.get(), shape.M, shape.N, shape.K, stream.value, blas.value};
    gemm_cublas(args);
    CUDA_CHECK(cudaMemcpyAsync(expected.data(), reference.get(), cc * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    // Independent CPU FP64 samples also check the cuBLAS entry itself.
    for (int i = 0; i < 32; ++i) {
      const int row = static_cast<int>(int64_t(i) * (shape.M - 1) / 31);
      const int col = static_cast<int>(int64_t((i * 13) % 32) * (shape.N - 1) / 31);
      const double cpu = reference_element(a, b, shape, row, col);
      const float gpu = expected[static_cast<std::size_t>(row) * shape.N + col];
      if (!std::isfinite(gpu) || std::abs(gpu - cpu) > 1e-3 + 1e-3 * std::abs(cpu))
        throw std::runtime_error("cuBLAS reference failed CPU FP64 spot check");
    }
    const std::vector<double> expected_double(expected.begin(), expected.end());
    args.C = dc.get();
    auto check_output = [&] {
      CUDA_CHECK(cudaMemcpyAsync(output.data(), dc.get(), cc * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
      CUDA_CHECK(cudaStreamSynchronize(stream.value));
      if (compare(output, expected_double).failures)
        throw std::runtime_error(impl + ": incorrect output");
    };
    CUDA_CHECK(cudaMemsetAsync(dc.get(), 0xff, cc * sizeof(float), stream.value));
    selected->launch(args);
    check_output();
    for (int i = 0; i < warmup; ++i) selected->launch(args);
    CUDA_CHECK(cudaStreamSynchronize(stream.value));

    std::cout << "PROFILE " << impl << " " << shape.M << "x" << shape.N << "x" << shape.K
              << " | " << prop.name << std::endl;
    if (!validate_only) {
      CUDA_CHECK(cudaProfilerStart());
      profiling = true;
    }
    selected->launch(args);
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    if (profiling) {
      CUDA_CHECK(cudaProfilerStop());
      profiling = false;
    }
    check_output();
    std::cout << "PASS " << impl << '\n';
  } catch (const std::exception& error) {
    if (profiling) cudaProfilerStop();
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
