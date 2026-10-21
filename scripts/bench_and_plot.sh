#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
for arg in "$@"; do
  case "$arg" in
    --help)
      echo "Usage: bash scripts/bench_and_plot.sh [gemm_bench options]"
      echo "Build, run CTest, benchmark, collect CSV/logs, and generate plots."
      echo "Options such as --shapes, --impl, --timing and --repeat are forwarded."
      echo "Output is saved in a new results/run_<timestamp>_<id> directory."
      exit 0 ;;
    --output|--check-only|--list)
      echo "This report command requires timed results and manages --output. Use bench_gemm_all.sh for $arg." >&2
      exit 2 ;;
  esac
done
if ! "$PYTHON_BIN" -c 'import matplotlib, numpy' >/dev/null 2>&1; then
  echo "Install plotting dependencies first: $PYTHON_BIN -m pip install -r $ROOT_DIR/requirements-plot.txt" >&2
  exit 1
fi
mkdir -p "$ROOT_DIR/results"
RUN_DIR="$(mktemp -d "$ROOT_DIR/results/run_$(date +%Y%m%d-%H%M%S)_XXXXXX")"
# pipefail preserves build/test/benchmark failure; the full diagnostic log stays.
bash "$ROOT_DIR/scripts/bench_gemm_all.sh" "$@" --output "$RUN_DIR/results.csv" 2>&1 | tee "$RUN_DIR/run.log"
"$PYTHON_BIN" "$ROOT_DIR/scripts/plot_results.py" --input "$RUN_DIR/results.csv" --output "$RUN_DIR/plots" 2>&1 | tee "$RUN_DIR/plot.log"
echo "Run directory: $RUN_DIR"
