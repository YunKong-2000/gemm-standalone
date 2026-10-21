#!/usr/bin/env python3
"""Benchmark all GEMMs on eight shapes, then profile all at M,N,K=4096,8192,4096."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import profile_gemm_all as profiling

ROOT = Path(__file__).resolve().parents[1]


def run_logged(command, log):
    """Keep live progress and a persistent log, and propagate command failures."""
    print(f"Running: {' '.join(command)}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError(f"command exited {process.returncode}; see {log}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=Path(os.environ.get("GEMM_BUILD_DIR", ROOT / "build")),
                        help="build directory (relative paths are rooted at the repository)")
    parser.add_argument("--arch", default=os.environ.get("CUDA_ARCHITECTURES", "80"), help="CUDA architectures")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("BUILD_JOBS", "4")), help="build parallelism")
    parser.add_argument("--skip-build", action="store_true", help="reuse rebuilt binaries; still run CTest")
    parser.add_argument("--warmup", type=int, default=2, help="warmup calls for benchmark and profiling")
    parser.add_argument("--repeat", type=int, default=5, help="benchmark timing samples")
    parser.add_argument("--batch", type=int, default=1, help="GEMM calls per timing sample")
    parser.add_argument("--timing", choices=("graph", "event"), default="graph", help="benchmark timing mode")
    parser.add_argument("--device", type=int, default=0, help="CUDA device (CTest uses device 0)")
    parser.add_argument("--seed", type=int, default=0, help="input seed")
    parser.add_argument("--ncu", default=os.environ.get("NCU", "ncu"), help="Nsight Compute executable")
    parser.add_argument("--set", dest="section_set", default="full", help="NCU section set")
    parser.add_argument("--cache-control", choices=("all", "none"), default="all", help="NCU cache control")
    parser.add_argument("--clock-control", choices=("base", "none"), default="base", help="NCU clock control")
    parser.add_argument("--output", type=Path, help="new result directory; must not already exist")
    args = parser.parse_args(argv)
    if min(args.jobs, args.repeat, args.batch) <= 0 or min(args.warmup, args.device, args.seed) < 0:
        parser.error("jobs/repeat/batch must be positive; warmup/device/seed must be nonnegative")
    ncu = shutil.which(args.ncu)
    if not ncu:
        parser.error(f"Nsight Compute CLI not found: {args.ncu}")
    build = args.build_dir.expanduser()
    if not build.is_absolute():
        build = ROOT / build
    build = build.resolve()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or ROOT / "results" / f"bench_ncu_{stamp}_{uuid.uuid4().hex[:8]}").expanduser().resolve()
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    commands=[], status="running")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    try:
        commands = []
        if not args.skip_build:
            commands += [
                ("configure", ["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                               "-DGEMM_HOST_TESTS_ONLY=OFF", "-DGEMM_PTXAS_VERBOSE=ON",
                               f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}"]),
                ("build", ["cmake", "--build", str(build), "--parallel", str(args.jobs)]),
            ]
        commands += [
            ("tests", ["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error"]),
            ("benchmark", [str(build / ("gemm_bench.exe" if os.name == "nt" else "gemm_bench")),
                           "--impl", "all", "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                           "--batch", str(args.batch), "--timing", args.timing,
                           "--device", str(args.device), "--seed", str(args.seed),
                           "--output", str(output / "results.csv")]),
        ]
        for name, command in commands:
            manifest["commands"].append(dict(stage=name, argv=command))
            run_logged(command, output / f"{name}.log")
        # Discover every implementation from gemm_profile --list. Do not filter
        # kernel names or cap launch counts: a cuBLAS call may use several kernels.
        profile_args = ["--skip-build", "--build-dir", str(build), "--ncu", ncu,
                        "--shapes", "4096,8192,4096", "--warmup", str(args.warmup),
                        "--device", str(args.device), "--seed", str(args.seed),
                        "--set", args.section_set, "--cache-control", args.cache_control,
                        "--clock-control", args.clock_control, "--output", str(output / "ncu")]
        manifest["commands"].append(dict(stage="ncu", argv=[sys.executable, str(ROOT / "scripts/profile_gemm_all.py")] + profile_args))
        status = profiling.main(profile_args)
        manifest["status"] = "failed" if status else "ok"
        print(f"Benchmark CSV: {output / 'results.csv'}\nNCU reports: {output / 'ncu'}", flush=True)
        return status
    except (OSError, RuntimeError) as error:
        manifest.update(status="failed", error=str(error))
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
