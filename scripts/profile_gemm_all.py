#!/usr/bin/env python3
"""Build standalone GEMM and collect one Nsight Compute report per implementation/shape."""

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

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
    # NCU raw CSV is commonly wide: launch metadata + one column per metric,
    # followed by a units row and one row per kernel launch. Also accept the
    # long Metric Name / Metric Unit / Metric Value form. Keep values as text.
    rows = []
    header = None
    metric_columns = []
    units = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for fields in csv.reader(handle):
            if "Metric Name" in fields and "Metric Value" in fields:
                header, metric_columns, units = fields, [], {}
                continue
            if "ID" in fields and "Kernel Name" in fields and any("__" in f for f in fields):
                header = fields
                metric_columns = [i for i, name in enumerate(header) if "__" in name]
                units = {}
                continue
            if header is None or len(fields) != len(header):
                continue  # Ignore profiler banners, blank lines, and diagnostics.
            record = dict(zip(header, fields))
            if not metric_columns:
                if record.get("Metric Name") and record.get("Metric Value", "").strip():
                    rows.append(record)
                continue
            if not record["ID"].strip() and not record["Kernel Name"].strip():
                units = {header[i]: fields[i] for i in metric_columns}
                continue
            if not record["ID"].strip().isdigit() or not record["Kernel Name"].strip():
                continue
            metadata = {name: value for name, value in record.items() if "__" not in name}
            for i in metric_columns:
                if fields[i].strip():
                    rows.append(dict(metadata, **{"Metric Name": header[i],
                        "Metric Unit": units.get(header[i], ""), "Metric Value": fields[i]}))
    if rows:
        return rows
    raise RuntimeError(f"no raw NCU metrics collected in {path}; expected a long metric table or "
                       "a wide raw table with ID, Kernel Name and metric columns; inspect the CSV and profiling log")


def write_metrics(path, rows):
    fields = list(dict.fromkeys(["implementation", "M", "N", "K", "source_report"] +
                               [key for row in rows for key in row]))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reparse_reports(directory):
    """Recover exported CSVs without rerunning NCU or changing original reports."""
    directory = directory.expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["profiles"]
    if not records:
        raise RuntimeError(f"no profiles in {directory / 'manifest.json'}")
    all_rows = []
    failures = 0
    for record in records:
        # Do not turn failed capture/validation/export commands into successes
        # merely because they left partial CSVs behind. Support old manifests.
        if record["status"] != "ok" and not record.get("error", "").startswith("no raw NCU metrics collected"):
            failures += 1
            print(f"SKIPPED {record['impl']}: {record.get('error', 'profiling failed')}", file=sys.stderr)
            continue
        stem = record["impl"] + "__" + "x".join(map(str, record["shape"]))
        try:
            rows = read_metrics(directory / (stem + ".metrics.csv"))
            m, n, k = record["shape"]
            for row in rows:
                row.update(implementation=record["impl"], M=m, N=n, K=k,
                           source_report=str(directory / (stem + ".ncu-rep")))
            all_rows.extend(rows)
        except (OSError, RuntimeError, csv.Error) as error:
            failures += 1
            print(f"FAILED {record['impl']}: {error}", file=sys.stderr)
    if all_rows:
        # Preserve the original summary and manifest, including failure history.
        output = directory / "metrics_all.reparsed.csv"
        write_metrics(output, all_rows)
        print(f"Saved: {output} ({len(records) - failures} passed, {failures} failed)")
    else:
        raise RuntimeError(f"no metrics recovered from {directory}")
    return 1 if failures else 0


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
    except (OSError, RuntimeError, csv.Error) as error:
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
    parser.add_argument("--reparse", type=Path,
                        help="only reparse an existing report directory into metrics_all.reparsed.csv; no NCU/GPU needed")
    args = parser.parse_args(argv)
    if args.reparse:
        try:
            return reparse_reports(args.reparse)
        except (OSError, RuntimeError, ValueError, KeyError, csv.Error) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
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
            # These are persistent reports, not private temporary files.
            # mkdtemp forces mode 0700: root-created container reports then
            # look empty to a host editor running as an ordinary user.
            output = parent / f"ncu_{stamp}_{uuid.uuid4().hex[:8]}"
            output.mkdir(mode=0o755)  # Still honor the caller's umask.
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
                else:
                    print(f"REPORT: {record['report']}", flush=True)
        if all_rows:
            write_metrics(output / "metrics_all.csv", all_rows)
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
