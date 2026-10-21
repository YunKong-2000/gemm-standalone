#!/usr/bin/env python3
"""Build standalone GEMM and collect one Nsight Compute report per implementation/shape."""

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def parse_shape(text):
    try:
        shape = tuple(int(v) for v in text.lower().replace("x", ",").split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be M,N,K") from error
    if len(shape) != 3 or min(shape) <= 0 or shape[0] % 128 or shape[1] % 128 or shape[2] % 16:
        raise argparse.ArgumentTypeError("shapes require positive M/N multiples of 128 and K multiples of 16")
    if shape[0] // 16 > 65535 or max(shape) > 2147483647:
        raise argparse.ArgumentTypeError("shape exceeds supported grid/integer limits")
    return shape


def run_logged(command, log):
    with Path(log).open("w", encoding="utf-8") as handle:
        result = subprocess.run([str(v) for v in command], cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; see {log}")


def read_metrics(path):
    # Ignore profiler banners before the raw CSV header, if present.
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    for index, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        if "Metric Name" in fields and "Metric Value" in fields:
            rows = list(csv.DictReader(io.StringIO("\n".join(lines[index:]))))
            rows = [r for r in rows if r.get("Metric Name") and r.get("Metric Value") is not None]
            if rows:
                return rows
    raise RuntimeError(f"no raw NCU metrics collected in {path}")


def profile_one(ncu, executable, impl, shape, directory, *, warmup, device, seed,
                section_set, cache_control, clock_control):
    stem = directory / (impl + "__" + "x".join(map(str, shape)))
    report = Path(str(stem) + ".ncu-rep")
    raw_csv = Path(str(stem) + ".metrics.csv")
    command = [ncu, "--target-processes", "all", "--profile-from-start", "off",
               "--replay-mode", "kernel", "--set", section_set,
               "--cache-control", cache_control, "--clock-control", clock_control,
               "--import-source", "yes", "--export", str(stem),
               str(executable), "--impl", impl, "--shape", ",".join(map(str, shape)),
               "--warmup", str(warmup), "--device", str(device), "--seed", str(seed)]
    record = dict(impl=impl, shape=list(shape), command=command, report=str(report))
    try:
        run_logged(command, Path(str(stem) + ".log"))
        if not report.is_file():
            raise RuntimeError("NCU produced no report; inspect the profiling log")
        export_command = [ncu, "--import", str(report), "--page", "raw", "--csv"]
        record["export_command"] = export_command
        run_logged(export_command, raw_csv)
        rows = read_metrics(raw_csv)
        for row in rows:
            row.update(implementation=impl, M=shape[0], N=shape[1], K=shape[2], source_report=str(report))
        record.update(status="ok", metrics_csv=str(raw_csv), metric_rows=len(rows))
        return record, rows
    except (OSError, RuntimeError) as error:
        record.update(status="failed", error=str(error))
        return record, []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=parse_shape, default=[(4096, 4096, 4096)])
    parser.add_argument("--impls", nargs="+", help="default: every entry from gemm_profile --list")
    parser.add_argument("--build-dir", type=Path, default=Path(os.environ.get("GEMM_BUILD_DIR", ROOT / "build")))
    parser.add_argument("--arch", default=os.environ.get("CUDA_ARCHITECTURES", "80"))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("BUILD_JOBS", "4")))
    parser.add_argument("--skip-build", action="store_true", help="use an already rebuilt gemm_profile executable")
    parser.add_argument("--ncu", default=os.environ.get("NCU", "ncu"))
    parser.add_argument("--set", dest="section_set", default="full", help="NCU section set (default: full)")
    parser.add_argument("--cache-control", choices=("all", "none"), default="all")
    parser.add_argument("--clock-control", choices=("base", "none"), default="base")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="new output directory; existing directories are rejected")
    args = parser.parse_args(argv)
    if args.jobs <= 0 or min(args.warmup, args.device, args.seed) < 0:
        parser.error("jobs must be positive; warmup/device/seed must be nonnegative")
    ncu = shutil.which(args.ncu)
    if not ncu:
        parser.error(f"Nsight Compute CLI not found: {args.ncu}")
    build = args.build_dir.expanduser()
    if not build.is_absolute():
        build = ROOT / build
    build = build.resolve()
    manifest = None
    output = None
    try:
        if args.output:
            output = args.output.expanduser().resolve()
            output.mkdir(parents=True, exist_ok=False)
        else:
            parent = ROOT / "results"
            parent.mkdir(exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output = Path(tempfile.mkdtemp(prefix=f"ncu_{stamp}_", dir=parent))
        manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        source_sha256={}, commands=[], profiles=[],
                        note="NCU replay metrics; only the selected call is profiled. Split-K memset is issued "
                             "inside the range but NCU kernel duration excludes non-kernel memory operations. "
                             "Use gemm_bench for steady-state end-to-end timing including output clear.")
        for base in (ROOT / "src", ROOT / "include"):
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    manifest["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        run_logged([ncu, "--version"], output / "ncu_version.txt")
        manifest["ncu_version"] = (output / "ncu_version.txt").read_text()
        if not args.skip_build:
            commands = [
                ["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                 "-DGEMM_HOST_TESTS_ONLY=OFF", "-DGEMM_PTXAS_VERBOSE=ON", f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}"],
                ["cmake", "--build", str(build), "--parallel", str(args.jobs)],
                ["ctest", "--test-dir", str(build), "--output-on-failure"],
            ]
            for name, command in zip(("configure", "build", "tests"), commands):
                manifest["commands"].append(command)
                print(f"{name}: {output / (name + '.log')}", flush=True)
                run_logged(command, output / (name + ".log"))
        executable = build / ("gemm_profile.exe" if os.name == "nt" else "gemm_profile")
        if not executable.is_file():
            raise RuntimeError(f"missing executable: {executable}; rebuild without --skip-build")
        run_logged([str(executable), "--list"], output / "implementations.txt")
        available = (output / "implementations.txt").read_text().split()
        if not available or len(available) != len(set(available)):
            raise RuntimeError("invalid/empty kernel registry from gemm_profile --list")
        impls = list(dict.fromkeys(args.impls or available))
        unknown = set(impls) - set(available)
        if unknown:
            raise RuntimeError("unknown implementations: " + ", ".join(sorted(unknown)))
        all_rows = []
        for shape in dict.fromkeys(args.shapes):
            for impl in impls:
                print(f"NCU {impl} {'x'.join(map(str, shape))}", flush=True)
                record, rows = profile_one(ncu, executable, impl, shape, output,
                    warmup=args.warmup, device=args.device, seed=args.seed,
                    section_set=args.section_set, cache_control=args.cache_control, clock_control=args.clock_control)
                manifest["profiles"].append(record)
                all_rows.extend(rows)
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                if record["status"] != "ok":
                    print(f"FAILED {impl}: {record['error']}", file=sys.stderr, flush=True)
        if all_rows:
            fields = list(dict.fromkeys(["implementation", "M", "N", "K", "source_report"] +
                                       [key for row in all_rows for key in row]))
            with (output / "metrics_all.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(all_rows)
        failures = sum(r["status"] != "ok" for r in manifest["profiles"])
        print(f"Saved: {output} ({len(manifest['profiles']) - failures} passed, {failures} failed)")
        return 1 if failures else 0
    except (OSError, RuntimeError) as error:
        if manifest is not None:
            manifest["error"] = str(error)
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if manifest is not None:
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
