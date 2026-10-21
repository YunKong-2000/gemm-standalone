#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${GEMM_BUILD_DIR:-$ROOT_DIR/build}"
if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$ROOT_DIR/$BUILD_DIR"; fi
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DGEMM_HOST_TESTS_ONLY=OFF -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80}"
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-4}"
ctest --test-dir "$BUILD_DIR" --output-on-failure
cd "$ROOT_DIR"
"$BUILD_DIR/gemm_bench" "$@"
