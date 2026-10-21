#include "gemm/gemm.h"

void gemm_cublas(const GemmArgs& args) {
  check_gemm_args(args, 1, 1, 1, false);
  if (!args.handle) throw std::invalid_argument("cuBLAS requires a handle");
  CUBLAS_CHECK(cublasSetStream(args.handle, args.stream));
  CUBLAS_CHECK(cublasSetPointerMode(args.handle, CUBLAS_POINTER_MODE_HOST));
  const float alpha = 1, beta = 0;
  // C^T[N,M] = B^T[N,K] * A^T[K,M]; A storage is column-major.
  CUBLAS_CHECK(cublasGemmEx(args.handle, CUBLAS_OP_N, CUBLAS_OP_T,
      args.N, args.M, args.K, &alpha,
      args.B, CUDA_R_32F, args.N, args.A, CUDA_R_32F, args.M,
      &beta, args.C, CUDA_R_32F, args.N,
      CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
}
