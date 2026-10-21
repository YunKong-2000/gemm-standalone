#pragma once
#include "gemm/gemm.h"

// Non-copyable owners; destructors never throw during exception unwinding.
template<typename T> class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) {
    if (!count || count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::invalid_argument("invalid device allocation size");
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return ptr_; }
 private:
  T* ptr_ = nullptr;
};
struct Stream {
  cudaStream_t value = nullptr;
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};
struct BlasHandle {
  cublasHandle_t value = nullptr;
  BlasHandle() { CUBLAS_CHECK(cublasCreate(&value)); }
  ~BlasHandle() { if (value) cublasDestroy(value); }
  BlasHandle(const BlasHandle&) = delete;
  BlasHandle& operator=(const BlasHandle&) = delete;
};
struct Event {
  cudaEvent_t value = nullptr;
  Event() { CUDA_CHECK(cudaEventCreate(&value)); }
  ~Event() { if (value) cudaEventDestroy(value); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};
struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  Graph() = default;
  ~Graph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
};
