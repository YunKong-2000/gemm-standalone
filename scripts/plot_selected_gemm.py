#!/usr/bin/env python3
"""Plot nine selected GEMMs from one benchmark/NCU run, without mixing timers."""

import argparse
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path

from plot_results import ENV_FIELDS, environment, read_results

# Input identifiers remain compatible with the original, immutable measurements.
KERNELS = {
    "naive": "naive",
    "block_tile": "block tile 16×16",
    "block_tile_float4_16x16": "block tile_float4",
    "thread_tile_8x8": "thread_tile 8×8",
    "warp_tile": "warp_tile",
    "prefetch_smem": "双缓冲共享内存",
    "prefetch_reg": "双缓冲寄存器",
    "split_k": "split-k",
    "cublas": "cuBLAS",
}
COLORS = ["#8b939f", "#a76d39", "#d6a02a", "#337bbe", "#6c59b0",
          "#009e87", "#da5368", "#dc7e25", "#172b45"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "*"]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_benchmark(source):
    rows, skipped = read_results([source])
    if skipped:
        raise ValueError("Invalid benchmark measurements: " + "\n".join(skipped))
    rows = [row for row in rows if row["impl"] in KERNELS]
    if len({environment(row) for row in rows}) != 1:
        raise ValueError("Expected one benchmark environment")
    shapes = list(dict.fromkeys(row["shape"] for row in rows))
    lookup = {(row["impl"], row["shape"]): row for row in rows}
    for shape in shapes:
        for impl in KERNELS:
            if (impl, shape) not in lookup:
                raise ValueError(f"Missing measurement: {impl}, {shape}")
    return rows, shapes, lookup


def load_ncu(source):
    launches = defaultdict(dict)
    units = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1, "s": 1e3,
             "nsecond": 1e-6, "usecond": 1e-3, "msecond": 1, "second": 1e3}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            impl = row["implementation"]
            if impl not in KERNELS or row["Metric Name"] != "gpu__time_duration.sum":
                continue
            shape = tuple(int(row[axis]) for axis in ("M", "N", "K"))
            key = tuple(row.get(field, "") for field in
                        ("source_report", "Process ID", "Context", "Stream", "ID"))
            ms = float(row["Metric Value"].replace(",", "")) * units[row["Metric Unit"]]
            if not math.isfinite(ms) or ms <= 0 or key in launches[(impl, shape)]:
                raise ValueError("Invalid or duplicate NCU duration")
            launches[(impl, shape)][key] = ms
    records = []
    for (impl, shape), kernels in launches.items():
        ms = sum(kernels.values())
        records.append(dict(kernel=KERNELS[impl], M=shape[0], N=shape[1], K=shape[2],
                            kernel_count=len(kernels), kernel_duration_ms=ms,
                            tflops=2 * math.prod(shape) / (ms * 1e9)))
    if not records:
        raise ValueError("No NCU duration measurements")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = (args.output or run / "selected_kernel_curves").resolve()
    source = run / "results.csv"
    rows, shapes, lookup = load_benchmark(source)
    ncu_source = run / "ncu/metrics_all.csv"
    ncu_manifest = json.loads((run / "ncu/manifest.json").read_text())
    ncu_settings = ncu_manifest["settings"]
    ncu = load_ncu(ncu_source)
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Arial Unicode MS", "Noto Sans CJK SC", "Heiti SC", "SimHei")
                 if name in available), None)
    if font is None:
        raise ValueError("A Chinese font is required (e.g. Noto Sans CJK SC)")
    plt.rcParams.update({"font.family": font, "font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#25364a", "text.color": "#25364a",
                         "savefig.facecolor": "white", "svg.fonttype": "path"})
    meta = {field: rows[0][field] for field in ENV_FIELDS}
    timer_label = "CUDA Graph" if meta["timing"] == "graph" else "CUDA Event"
    subtitle = (f"{meta['gpu']}  ·  {timer_label}  ·  warmup={meta['warmup']} / "
                f"repeat={meta['repeat']} / batch={meta['batch']}")
    shape_labels = ["×".join(map(str, shape)) for shape in shapes]
    names = list(KERNELS)
    throughput = np.array([[2 * math.prod(shape) / (lookup[impl, shape]['median_ms'] * 1e9)
                            for shape in shapes] for impl in names])
    latency = np.array([[lookup[impl, shape]['median_ms'] for shape in shapes] for impl in names])
    images = []

    def save(fig, stem, title):
        for extension in ("png", "svg"):
            fig.savefig(output / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        images.append((stem, title))

    def decorate(ax):
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(15, 8))
    for i, impl in enumerate(names):
        ax.plot(range(len(shapes)), throughput[i], label=KERNELS[impl], color=COLORS[i],
                marker=MARKERS[i], linewidth=2.2, markersize=6,
                linestyle="--" if impl == "cublas" else "-")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("吞吐量 / TFLOPS")
    decorate(ax)
    ax.set_xticks(range(len(shapes)), shape_labels, rotation=25, ha="right")
    ax.set_xlabel("问题规模 M×N×K（按测试顺序，类别轴）")
    fig.suptitle("GEMM：不同问题规模下的性能曲线", x=0.07, ha="left", fontsize=23)
    fig.text(0.07, 0.91, subtitle, fontsize=11)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.51, 0.88),
               ncol=5, frameon=False, fontsize=11)
    fig.text(0.07, 0.018, "TFLOPS = 2MNK / median_ms / 10⁹；Split-K 包含输出清零；cuBLAS 使用严格 FP32。", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.21)
    save(fig, "throughput_by_size", "按问题规模比较吞吐量")

    fig, ax = plt.subplots(figsize=(15, 7))
    for i, impl in enumerate(names):
        ax.plot(range(len(shapes)), latency[i], label=KERNELS[impl], color=COLORS[i],
                marker=MARKERS[i], linewidth=2, linestyle="--" if impl == "cublas" else "-")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("GPU 耗时中位数 / ms（越低越好）")
    ax.set_xticks(range(len(shapes)), shape_labels, rotation=25, ha="right")
    ax.set_xlabel("问题规模 M×N×K（按测试顺序，类别轴）")
    decorate(ax)
    ax.legend(ncol=3, fontsize=10, frameon=False)
    fig.suptitle("GEMM：不同问题规模下的耗时", x=0.07, ha="left", fontsize=23)
    fig.text(0.07, 0.91, subtitle, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.23)
    save(fig, "latency_by_size", "按问题规模比较耗时")

    fig, axes = plt.subplots(math.ceil(len(shapes) / 3), 3, figsize=(18, 13), squeeze=False)
    for j, (shape, ax) in enumerate(zip(shapes, axes.flat)):
        ax.plot(range(len(names)), throughput[:, j], color="#337bbe", linewidth=1.8, zorder=2)
        for i in range(len(names)):
            ax.scatter(i, throughput[i, j], color=COLORS[i], marker=MARKERS[i], s=45, zorder=3)
            ax.annotate(f"{throughput[i,j]:.2f}", (i, throughput[i,j]),
                        xytext=(0, 9 if i % 2 == 0 else -16), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set_title(shape_labels[j], loc="left", fontsize=14)
        ax.set_xticks(range(len(names)), list(KERNELS.values()), rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("TFLOPS")
        ax.set_ylim(0, throughput[:, j].max() * 1.25)
        decorate(ax)
    for ax in list(axes.flat)[len(shapes):]:
        ax.set_visible(False)
    fig.suptitle("GEMM：逐规模比较 9 个实现", x=0.04, ha="left", fontsize=23)
    fig.text(0.04, 0.94, subtitle + "  ·  各面板纵轴独立", fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.10, wspace=0.2, hspace=0.86)
    save(fig, "throughput_by_kernel", "逐规模比较各实现")

    ncu_shapes = list(dict.fromkeys((r['M'], r['N'], r['K']) for r in ncu))
    for shape in ncu_shapes:
        records = {r['kernel']: r for r in ncu if (r['M'], r['N'], r['K']) == shape}
        if set(records) != set(KERNELS.values()):
            raise ValueError(f"Incomplete NCU coverage: {shape}")
        fig, ax = plt.subplots(figsize=(12, 7))
        values = [records[label]['tflops'] for label in KERNELS.values()]
        bars = ax.barh(list(KERNELS.values()), values, color=COLORS, height=0.65)
        ax.bar_label(bars, fmt="%.2f", padding=6)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.15)
        ax.set_xlabel("吞吐量 / TFLOPS（2MNK / NCU kernel duration）")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        fig.suptitle("NCU 单规模采集：" + "×".join(map(str, shape)), x=0.05, ha="left", fontsize=21)
        fig.text(0.05, 0.91, f"{meta['gpu']}  ·  kernel replay / cache-control {ncu_settings['cache_control']} / "
                 f"clock-control {ncu_settings['clock_control']}", fontsize=11)
        fig.text(0.05, 0.035, f"NCU duration 不含非 kernel 的 memset；此图与 {timer_label} 曲线采用不同计时口径。", fontsize=10)
        fig.subplots_adjust(left=0.20, right=0.95, top=0.84, bottom=0.13)
        save(fig, "ncu_" + "x".join(map(str, shape)), "NCU 单规模吞吐量")

    summary = [dict(kernel=KERNELS[impl], M=s[0], N=s[1], K=s[2],
                    median_ms=latency[i,j], tflops=throughput[i,j],
                    speedup_over_naive=latency[0,j]/latency[i,j],
                    speedup_over_cublas=latency[-1,j]/latency[i,j])
               for j,s in enumerate(shapes) for i,impl in enumerate(names)]
    write_csv(output / "summary.csv", summary)
    write_csv(output / "ncu_summary.csv", ncu)
    manifest = dict(run=str(run), environment=meta, measurements=len(summary), shapes=shapes,
                    selected_kernels=list(KERNELS.values()),
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (source, ncu_source, run/'ncu/manifest.json')},
                    methodology="Benchmark: 2MNK / median_ms / 1e9, Split-K includes output clear. "
                                "NCU: sum device-kernel durations per implementation and shape; excludes non-kernel memset.")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    figures = ''.join(f'<h2>{title}</h2><a href="{stem}.svg">SVG 矢量图</a>'
                      f'<a href="{stem}.png"><img src="{stem}.png" alt="{title}"></a>' for stem,title in images)
    (output / "report.html").write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>GEMM 性能曲线</title><style>body{{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#25364a}}
img{{width:100%;height:auto}}p{{line-height:1.8}}a{{color:#187c97}}h2{{margin-top:48px}}</style>
<h1>GEMM 性能曲线 · 9 个指定实现</h1><p>数据批次：{html.escape(run.name)}<br>{html.escape(subtitle)}</p>
<p>跨规模曲线来自同批次 results.csv，共 {len(shapes)} 个规模、{len(summary)} 条已通过正确性校验的记录。
吞吐量 = 2MNK / median_ms / 10⁹，Split-K 包含输出清零；cuBLAS 为严格 FP32。
横轴按测试顺序排列，连线仅辅助比较，相同计算量的不同形状仍独立展示。</p>
<p>本批次 NCU 仅采集 {', '.join('×'.join(map(str,s)) for s in ncu_shapes)}。
NCU 图单独使用 gpu__time_duration.sum，多个 device kernel 的耗时相加；不包含非 kernel 的 memset，未与 benchmark 数据混合。</p>
<p><a href="summary.csv">跨规模数据 CSV</a> · <a href="ncu_summary.csv">NCU 数据 CSV</a> ·
<a href="manifest.json">来源与校验信息</a></p>{figures}</html>''', encoding="utf-8")
    print(output / "report.html")


if __name__ == "__main__":
    main()
