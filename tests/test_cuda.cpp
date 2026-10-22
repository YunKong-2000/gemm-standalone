#include "cuda_resources.h"
#include <iostream>

void check_split_k_register_pipeline(const Kernel& kernel, cudaStream_t stream, cublasHandle_t handle) {
  // Distinct contributions for every K row expose missing register-prefetch
  // rows and duplicated prologue/drain work across both Split-K slices.
  for (int k : {16, 32, 48, 64, 80}) {
    const Shape s{128, 256, k};
    std::vector<float> a(s.M * s.K), b(s.K * s.N), output(s.M * s.N);
    for (int kk = 0; kk < k; ++kk) {
      for (int row = 0; row < s.M; ++row) a[kk * s.M + row] = kk + 1;
      for (int col = 0; col < s.N; ++col) b[kk * s.N + col] = kk + 1;
    }
    const double sum = double(k) * (k + 1) * (2 * k + 1) / 6;
    const std::vector<double> expected(output.size(), sum);
    DeviceBuffer<float> da(a.size()), db(b.size()), dc(output.size());
    CUDA_CHECK(cudaMemcpyAsync(da.get(), a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(db.get(), b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(dc.get(), 0xff, output.size() * sizeof(float), stream));
    GemmArgs args{da.get(), db.get(), dc.get(), s.M, s.N, s.K, stream, handle};
    for (int repeat = 0; repeat < 2; ++repeat) {
      kernel.launch(args);
      CUDA_CHECK(cudaMemcpyAsync(output.data(), dc.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (compare(output, expected).max_abs_error != 0)
        throw std::runtime_error(std::string(kernel.name) + ": register pipeline K-row coverage failed at K=" + std::to_string(k));
    }
  }
}

int main() {
  try {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 8) throw std::runtime_error("SM80+ required");
    Stream stream;
    BlasHandle blas;
    for (const auto& kernel : gemm_kernels()) {
      std::cout << "RUN  " << kernel.name << std::endl;
      if (kernel.stage == 8) check_split_k_register_pipeline(kernel, stream.value, blas.value);
      // One, two and three tiles per K slice cover both final buffer stages.
      for (int scale : {1, 2, 3}) {
        const Shape s{kernel.BM * scale, kernel.BN * (scale + 1), kernel.BK * scale};
        const auto ac = matrix_bytes<float>(s.M, s.K) / sizeof(float);
        const auto bc = matrix_bytes<float>(s.K, s.N) / sizeof(float);
        const auto cc = matrix_bytes<float>(s.M, s.N) / sizeof(float);
        std::vector<float> a(ac), b(bc), output(cc);
        for (std::size_t i = 0; i < ac; i++) a[i] = static_cast<int>(i % 7) - 3;
        for (std::size_t i = 0; i < bc; i++) b[i] = static_cast<int>(i % 5) - 2;
        std::vector<double> expected(cc);
        for (int row = 0; row < s.M; row++)
          for (int col = 0; col < s.N; col++)
            expected[static_cast<std::size_t>(row) * s.N + col] = reference_element(a,b,s,row,col);
        DeviceBuffer<float> da(ac), db(bc), dc(cc);
        CUDA_CHECK(cudaMemcpyAsync(da.get(),a.data(),ac*sizeof(float),cudaMemcpyHostToDevice,stream.value));
        CUDA_CHECK(cudaMemcpyAsync(db.get(),b.data(),bc*sizeof(float),cudaMemcpyHostToDevice,stream.value));
        CUDA_CHECK(cudaMemsetAsync(dc.get(),0xff,cc*sizeof(float),stream.value));
        GemmArgs args{da.get(),db.get(),dc.get(),s.M,s.N,s.K,stream.value,blas.value};
        for (int repeat = 0; repeat < 3; repeat++) {
          kernel.launch(args);
          CUDA_CHECK(cudaMemcpyAsync(output.data(),dc.get(),cc*sizeof(float),cudaMemcpyDeviceToHost,stream.value));
          CUDA_CHECK(cudaStreamSynchronize(stream.value));
          if (compare(output,expected).max_abs_error != 0)
            throw std::runtime_error(std::string(kernel.name) + ": exact-valued test failed");
        }
        args.K = 0;
        bool rejected = false;
        try { kernel.launch(args); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("zero K was not rejected");
        if (kernel.BK > 1) {
          args.K = s.K - 1;
          rejected = false;
          try { kernel.launch(args); } catch (const std::invalid_argument&) { rejected = true; }
          if (!rejected) throw std::runtime_error("partial K tile was not rejected");
        }
        args.K = s.K;
        if (kernel.stage >= 3 && kernel.stage <= 8) {
          args.A = da.get() + 1;
          rejected = false;
          try { kernel.launch(args); } catch (const std::invalid_argument&) { rejected = true; }
          if (!rejected) throw std::runtime_error("unaligned A was not rejected");
        }
      }
      std::cout << "PASS " << kernel.name << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
