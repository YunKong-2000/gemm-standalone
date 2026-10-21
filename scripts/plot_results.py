#!/usr/bin/env python3
"""Collect standalone GEMM CSVs and plot the optimization ladder (no GPU needed)."""

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Keep variants separate: they change block/thread geometry, not just stage number.
IMPLEMENTATIONS = (
    ("naive", "1  Naive"),
    ("block_tile", "2  Shared memory"),
    ("block_tile_float4_16x16", "3  float4 / block 16x16"),
    ("block_tile_float4_16x32", "3  float4 / block 16x32"),
    ("block_tile_float4_32x32", "3  float4 / block 32x32"),
    ("thread_tile_4x4", "4  Thread tile 4x4"),
    ("thread_tile_4x8", "4  Thread tile 4x8"),
    ("thread_tile_8x8", "4  Thread tile 8x8"),
    ("thread_tile_8x8_warp4x8", "4  Thread tile 8x8 (warp 4x8)"),
    ("warp_tile", "5  Warp tile / four subtiles"),
    ("prefetch_smem", "6  Shared-memory double buffer"),
    ("prefetch_smem_contiguous8x8", "6  Smem buffer / 8x8, warp 4x8"),
    ("prefetch_smem_contiguous8x8_warp2x16", "6  Smem buffer / 8x8, warp 2x16"),
    ("prefetch_reg", "7  + Register double buffer"),
    ("prefetch_reg_contiguous8x8", "7  Register buffer / 8x8, warp 4x8"),
    ("prefetch_reg_contiguous8x8_warp2x16", "7  Register buffer / 8x8, warp 2x16"),
    ("split_k", "8  Split-K"),
    ("split_k_contiguous8x8", "8  Split-K / 8x8, warp 4x8"),
    ("split_k_contiguous8x8_warp2x16", "8  Split-K / 8x8, warp 2x16"),
    ("cublas", "9  cuBLAS FP32"),
)
LABELS = dict(IMPLEMENTATIONS)
SUBTILE_BASELINES = {
    "prefetch_smem_contiguous8x8": "prefetch_smem",
    "prefetch_smem_contiguous8x8_warp2x16": "prefetch_smem",
    "prefetch_reg_contiguous8x8": "prefetch_reg",
    "prefetch_reg_contiguous8x8_warp2x16": "prefetch_reg",
    "split_k_contiguous8x8": "split_k",
    "split_k_contiguous8x8_warp2x16": "split_k",
}
SPLIT_K_IMPLS = {"split_k", "split_k_contiguous8x8", "split_k_contiguous8x8_warp2x16"}
ENV_FIELDS = (
    "gpu", "compute_capability", "driver_version", "runtime_version",
    "cublas_version", "timing", "warmup", "repeat", "batch", "seed", "a_layout",
)
REQUIRED_FIELDS = {"stage", "impl", "M", "N", "K", "verified", "median_ms", "max_abs_error", *ENV_FIELDS}
METRICS = ("median_ms", "tflops", "speedup_over_naive", "speedup_over_cublas", "speedup_over_subtile")
SUMMARY_FIELDS = ("impl", "stage", "M", "N", "K", "runs", *METRICS,
                  "naive_pairs", "cublas_pairs", "subtile_pairs", "max_abs_error")


def environment(row):
    return tuple(row[field] for field in ENV_FIELDS)


def discover_inputs(inputs, output):
    """Ignore generated reports and unrelated CSVs during directory scans."""
    output = Path(output).resolve()
    found = set()
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"input does not exist: {path}")
        candidates = path.rglob("*.csv") if path.is_dir() else [path]
        for candidate in candidates:
            if output == candidate or output in candidate.parents:
                continue
            with candidate.open(newline="", encoding="utf-8-sig") as handle:
                fields = set(next(csv.reader(handle), []))
            if REQUIRED_FIELDS <= fields and "source_csv" not in fields:
                found.add(candidate)
            elif path.is_file():
                raise ValueError(f"not a raw standalone GEMM CSV: {candidate}")
    if not found:
        raise ValueError("no benchmark CSVs found; use --input results or --input path/to/run.csv")
    return sorted(found)


def read_results(paths):
    observations, skipped = [], []
    for path in paths:
        seen = set()
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for line, raw in enumerate(csv.DictReader(handle), 2):
                prefix = f"{path}:{line}"
                if None in raw or any(value is None for value in raw.values()):
                    skipped.append(f"{prefix}: incomplete CSV row")
                    continue
                if raw["verified"].lower() not in ("true", "1"):
                    skipped.append(f"{prefix}: correctness not verified")
                    continue
                if raw["timing"] == "check_only":
                    skipped.append(f"{prefix}: correctness-only row, no performance measurement")
                    continue
                try:
                    if raw["impl"] not in LABELS:
                        raise ValueError(f"unknown implementation: {raw['impl']}")
                    if raw["timing"] not in ("graph", "event"):
                        raise ValueError("unknown timing mode")
                    if raw["a_layout"] != "column_major":
                        raise ValueError("expected column_major A")
                    shape = tuple(int(raw[axis]) for axis in ("M", "N", "K"))
                    if min(shape) <= 0:
                        raise ValueError("nonpositive matrix dimension")
                    stage = int(raw["stage"])
                    if stage != int(LABELS[raw["impl"]].split()[0]):
                        raise ValueError("stage does not match implementation")
                    ms = float(raw["median_ms"])
                    error = float(raw["max_abs_error"])
                    if not math.isfinite(ms) or ms <= 0:
                        raise ValueError("median_ms must be finite and positive")
                    if not math.isfinite(error) or error < 0:
                        raise ValueError("invalid correctness error")
                    if raw["impl"] in SPLIT_K_IMPLS and raw.get("includes_output_clear", "").lower() != "true":
                        raise ValueError("Split-K timing must include output clear")
                    for field in ENV_FIELDS:
                        if not raw[field].strip():
                            raise ValueError(f"missing environment value: {field}")
                    key = (environment(raw), shape, raw["impl"])
                    if key in seen:
                        raise ValueError("duplicate implementation/shape in a single CSV")
                except (ValueError, KeyError) as error_message:
                    skipped.append(f"{prefix}: {error_message}")
                    continue
                seen.add(key)
                observations.append({**raw, "source_csv": str(path), "shape": shape,
                                     "stage": stage, "median_ms": ms, "max_abs_error": error})
    if not observations:
        raise ValueError("no verified timing rows remain\n" + "\n".join(skipped[:10]))
    return observations, skipped


def summarize_group(rows):
    # Pair baselines only inside the same source file AND shape. Never divide
    # measurements from different runs, even when their metadata matches.
    baselines = {(r["source_csv"], r["shape"], r["impl"]): r["median_ms"] for r in rows}
    buckets = defaultdict(list)
    shapes = list(dict.fromkeys(r["shape"] for r in rows))
    for row in rows:
        paired = dict(row)
        for name in ("naive", "cublas"):
            baseline = baselines.get((row["source_csv"], row["shape"], name))
            paired[f"speedup_over_{name}"] = baseline / row["median_ms"] if baseline is not None else None
        baseline = baselines.get((row["source_csv"], row["shape"], SUBTILE_BASELINES.get(row["impl"])))
        paired["speedup_over_subtile"] = baseline / row["median_ms"] if baseline is not None else None
        buckets[(row["impl"], row["shape"])].append(paired)
    summary = []
    for name, _ in IMPLEMENTATIONS:
        for shape in shapes:
            samples = buckets.get((name, shape))
            if not samples:
                continue
            ms = statistics.median(s["median_ms"] for s in samples)
            result = dict(impl=name, stage=samples[0]["stage"], M=shape[0], N=shape[1], K=shape[2],
                          runs=len(samples), median_ms=ms,
                          tflops=2.0 * math.prod(shape) / (ms * 1e9),
                          max_abs_error=max(s["max_abs_error"] for s in samples))
            for baseline in ("naive", "cublas", "subtile"):
                values = [s[f"speedup_over_{baseline}"] for s in samples
                          if s[f"speedup_over_{baseline}"] is not None]
                result[f"speedup_over_{baseline}"] = statistics.median(values) if values else None
                result[f"{baseline}_pairs"] = len(values)
            summary.append(result)
    return summary, shapes


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plots(summary, shapes, metadata, output, formats):
    # Lazy imports keep CSV collection/statistical tests independent of matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm, LinearSegmentedColormap
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 17, "axes.titleweight": "bold",
                         "savefig.facecolor": "white"})
    names = [name for name, _ in IMPLEMENTATIONS if any(r["impl"] == name for r in summary)]
    lookup = {(r["impl"], (r["M"], r["N"], r["K"])): r for r in summary}
    shape_labels = [" x ".join(map(str, shape)) for shape in shapes]
    subtitle = (f"{metadata['gpu']}  |  {metadata['timing']} timing  |  "
                f"{len(shapes)} shapes, {len(names)} implementations")
    images = []

    def save(fig, stem):
        for extension in formats:
            fig.savefig(output / f"{stem}.{extension}", dpi=170, bbox_inches="tight")
        plt.close(fig)
        images.append(stem)

    def heatmap(metric, stem, title, relative=False):
        data = np.array([[lookup.get((name, s), {}).get(metric) or np.nan for s in shapes]
                         for name in names], dtype=float)
        valid = data[np.isfinite(data)]
        fig, ax = plt.subplots(figsize=(max(11, 0.98 * len(shapes) + 6), max(5, 0.48 * len(names) + 2.8)))
        if relative:
            # Symmetric log2 coloring around 1x; ticks display actual ratios.
            bound = max(1.0, float(np.max(np.abs(np.log2(valid))))) if valid.size else 1.0
            norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
            colors = LinearSegmentedColormap.from_list("speedup", ["#bd4c66", "#f3f5f8", "#087f80"])
            colors.set_bad("#e4e7eb")
            with np.errstate(invalid="ignore"):
                image = ax.imshow(np.ma.masked_invalid(np.log2(data)), aspect="auto", cmap=colors, norm=norm)
            colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
            tick_values = sorted(set([0, -bound, bound] + [float(i) for i in range(math.ceil(-bound), math.floor(bound) + 1)]))
            if len(tick_values) > 9:
                tick_values = [-bound, -bound / 2, 0, bound / 2, bound]
            colorbar.set_ticks(tick_values)
            colorbar.set_ticklabels([f"{2**v:.2g}x" for v in tick_values])
            colorbar.set_label("Speedup (log color scale); >1x is faster")
        else:
            low, high = (float(valid.min()), float(valid.max())) if valid.size else (1, 2)
            if low == high:
                high = low * 1.01
            colors = plt.get_cmap("YlGnBu").copy()
            colors.set_bad("#e4e7eb")
            image = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=colors,
                              norm=LogNorm(vmin=low, vmax=high))
            fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035).set_label("Median GPU time (ms, log scale)")
        for i in range(len(names)):
            for j in range(len(shapes)):
                value = data[i, j]
                if not np.isfinite(value):
                    text, color = "N/A", "#727985"
                else:
                    text = f"{value:.2f}x" if relative else f"{value:.4f}"
                    rgba = image.cmap(image.norm(math.log2(value) if relative else value))
                    color = "white" if sum(w * c for w, c in zip((0.2126, 0.7152, 0.0722), rgba[:3])) < 0.55 else "#20334d"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
        ax.set_yticks(range(len(names)), [LABELS[name] for name in names])
        ax.set_xticks(range(len(shapes)), shape_labels, rotation=30, ha="right")
        ax.tick_params(length=0, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.suptitle(title, x=0.025, ha="left", fontsize=20, fontweight="bold", color="#20334d")
        fig.text(0.025, 0.91, subtitle, color="#40536b")
        fig.text(0.025, 0.012, "M x N x K. Verified outputs only. Split-K includes output clear. N/A = missing measurement or same-run baseline.", fontsize=9)
        fig.subplots_adjust(left=0.32, right=0.95, top=0.85, bottom=0.19)
        save(fig, stem)

    heatmap("speedup_over_naive", "speedup_by_shape", "Optimization gains relative to naive", True)
    heatmap("speedup_over_cublas", "cublas_speedup_by_shape", "Performance relative to strict-FP32 cuBLAS", True)
    heatmap("median_ms", "latency_by_shape", "GEMM latency across the optimization ladder")

    fig, ax = plt.subplots(figsize=(15, 8))
    colors = plt.get_cmap("tab10")
    for index, shape in enumerate(shapes):
        values = [lookup.get((name, shape), {}).get("tflops", np.nan) for name in names]
        ax.plot(range(len(names)), values, marker="o", linewidth=2, markersize=5,
                color=colors(index % 10), linestyle=("-", "--", ":")[(index // 10) % 3], label=shape_labels[index])
    ax.set_xticks(range(len(names)), [LABELS[n] for n in names], rotation=35, ha="right")
    ax.set_ylabel("TFLOPS (2MNK / median GPU time)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="M x N x K", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle("Throughput along the GEMM optimization ladder", x=0.045, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.045, 0.91, subtitle, color="#40536b")
    fig.text(0.045, 0.012, "Tile and layout variants are shown separately. Later stages are not guaranteed to be faster.", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.8, top=0.84, bottom=0.31)
    save(fig, "tflops_by_stage")
    return images


def write_report(output, metadata, summary, shapes, sources, images, skipped):
    best_rows = []
    for shape in shapes:
        candidates = [r for r in summary if (r["M"], r["N"], r["K"]) == shape and r["impl"] != "cublas"]
        if candidates:
            best = min(candidates, key=lambda r: r["median_ms"])
            ratio = best["speedup_over_cublas"]
            best_rows.append(f"<tr><td>{' x '.join(map(str, shape))}</td><td>{html.escape(best['impl'])}</td>"
                             f"<td>{best['median_ms']:.4f}</td><td>{best['tflops']:.3f}</td>"
                             f"<td>{f'{ratio:.3f}x' if ratio is not None else 'N/A'}</td></tr>")
    layout_rows = []
    for row in summary:
        baseline = SUBTILE_BASELINES.get(row["impl"])
        if baseline:
            ratio = row["speedup_over_subtile"]
            layout_rows.append(
                f"<tr><td>{row['M']} x {row['N']} x {row['K']}</td>"
                f"<td>{html.escape(row['impl'])}</td><td>{html.escape(baseline)}</td>"
                f"<td>{f'{ratio:.3f}x' if ratio is not None else 'N/A'}</td>"
                f"<td>{row['subtile_pairs']}</td></tr>")
    layout_table = ("<h2>Continuous 8x8 vs four subtiles</h2>"
                    "<p>Subtile time / continuous time, paired within each source CSV and shape. "
                    "Greater than 1x means the continuous layout is faster. "
                    "Continuous variants use 4x8 threads per warp (32x64 C tile) or, "
                    "with the warp2x16 suffix, 2x16 threads (16x128 C tile). Subtile uses 8x4.</p>"
                    "<table><tr><th>M x N x K</th><th>Continuous implementation</th>"
                    "<th>Subtile baseline</th><th>Speedup</th><th>Paired runs</th></tr>"
                    + "".join(layout_rows) + "</table>") if layout_rows else ""
    pictures = "\n".join(f'<figure><a href="{name}.png"><img src="{name}.png" alt="{name}"></a></figure>' for name in images)
    meta_html = " | ".join(f"{html.escape(k)}: {html.escape(v)}" for k, v in metadata.items())
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>GEMM optimization report</title><style>
body{{font:15px system-ui,sans-serif;color:#20334d;background:#f6f8fb;margin:0 auto;padding:36px;max-width:1500px}}
h1{{font-size:32px}}p{{line-height:1.6}}img{{width:100%;height:auto}}figure{{margin:28px 0;background:white;border-radius:10px;padding:12px}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;border-bottom:1px solid #e1e6ed;text-align:left}}
a{{color:#087f80}}code{{background:#e8edf3;padding:2px 5px}}details{{margin:20px 0}}</style>
<h1>GEMM optimization ladder</h1><p>{meta_html}</p>
<p>{len(sources)} source CSV(s), {len(shapes)} shape(s), {len(summary)} measured combinations.
<a href="summary.csv">Summary CSV</a> · <a href="collected_results.csv">Collected raw rows</a> · <a href="manifest.json">Provenance</a></p>
<p>Each observation uses median_ms. Repeated runs aggregate by median of run medians;
speedups aggregate same-file, same-shape baseline ratios by median. Missing baselines remain N/A.
Different GPU, CUDA/cuBLAS versions, timing settings, seeds or layouts are reported separately.
CSV metadata cannot detect different source revisions, GPU clocks or concurrent workloads; only combine comparable runs.</p>
<h2>Fastest measured handwritten kernel per shape</h2>
<p>This selects among the implementations present in these files; it is not an independently tuned kernel.</p>
<table><tr><th>M x N x K</th><th>Implementation</th><th>Median ms</th><th>TFLOPS</th><th>vs cuBLAS</th></tr>{''.join(best_rows)}</table>
{layout_table}
{pictures}<details><summary>Input files</summary><ul>{''.join('<li>' + html.escape(s) + '</li>' for s in sources)}</ul></details>
<p>Rejected / correctness-only rows across all inputs: {len(skipped)}. See the manifest for details.</p></html>'''
    (output / "report.html").write_text(document, encoding="utf-8")


def generate(paths, output, formats=("png", "svg")):
    observations, skipped = read_results(paths)
    groups = defaultdict(list)
    for row in observations:
        groups[environment(row)].append(row)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, (key, rows) in enumerate(groups.items(), 1):
        directory = output if len(groups) == 1 else output / f"group_{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        metadata = dict(zip(ENV_FIELDS, key))
        summary, shapes = summarize_group(rows)
        save_csv(directory / "summary.csv", summary, SUMMARY_FIELDS)
        raw_fields = list(dict.fromkeys(k for row in rows for k in row if k != "shape"))
        save_csv(directory / "collected_results.csv", rows, raw_fields)
        sources = sorted(set(row["source_csv"] for row in rows))
        manifest = {"environment": metadata, "sources": sources, "shapes": shapes,
                    "aggregation": "median of run medians; median of same-run speedup ratios",
                    "summary": summary, "skipped_rows": skipped}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        images = plots(summary, shapes, metadata, directory, formats)
        write_report(directory, metadata, summary, shapes, sources, images, skipped)
        reports.append((directory / "report.html", metadata))
    links = "".join(f'<li><a href="{path.relative_to(output).as_posix()}">{html.escape(meta["gpu"])} / '
                    f'{html.escape(meta["timing"])} / group {i}</a></li>'
                    for i, (path, meta) in enumerate(reports, 1))
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>GEMM reports</title>'
                                      '<h1>GEMM benchmark reports</h1><ul>' + links + '</ul>', encoding="utf-8")
    return reports, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", default=["results"], help="raw CSV files or directories (recursive)")
    parser.add_argument("--output", type=Path, default=Path("results/report"))
    parser.add_argument("--latest", action="store_true", help="report only the most recently modified raw CSV")
    parser.add_argument("--formats", nargs="+", choices=("png", "svg", "pdf"), default=["png", "svg"])
    args = parser.parse_args()
    try:
        output = args.output.expanduser().resolve()
        paths = discover_inputs(args.input, output)
        if args.latest:
            paths = [max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))]
        # PNG is always generated because the HTML report embeds it.
        formats = tuple(dict.fromkeys(["png", *args.formats]))
        reports, skipped = generate(paths, output, formats)
        print(f"Collected {len(paths)} CSV(s); generated {len(reports)} comparison group(s).")
        for report, _ in reports:
            print(f"Report: {report}")
        if skipped:
            print(f"Skipped {len(skipped)} invalid/correctness-only row(s); see manifest.json.")
        print(f"Index: {output / 'index.html'}")
    except ImportError as error:
        parser.exit(1, f"Plotting dependency missing: {error}. Run: python3 -m pip install -r requirements-plot.txt\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
