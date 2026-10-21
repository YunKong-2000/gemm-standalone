"""CPU regressions for collection, same-run ratios, grouping, and report rendering."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_results", ROOT / "scripts/plot_results.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def measurement(impl, ms=1.0, shape=(128, 256, 16), **overrides):
    row = dict(stage=plot.LABELS[impl].split()[0], impl=impl, M=shape[0], N=shape[1], K=shape[2],
               verified="true", max_abs_error=0.0001, median_ms=ms, gpu="SYNTHETIC TEST DATA - NOT GPU MEASUREMENTS",
               compute_capability="8.0", driver_version="13000", runtime_version="12080", cublas_version="120804",
               timing="graph", warmup="10", repeat="50", batch="10", seed="0", a_layout="column_major",
               includes_output_clear="true" if impl in plot.SPLIT_K_IMPLS else "false")
    row.update(overrides)
    return row


def write_fixture(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TestPlotResults(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemm-plot-tests-")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def load(self, name, rows):
        path = self.root / name
        write_fixture(path, rows)
        return path

    def test_pairs_within_run_then_aggregates_medians(self):
        a = self.load("a.csv", [measurement("naive",10), measurement("thread_tile_8x8",2), measurement("cublas",1)])
        b = self.load("b.csv", [measurement("naive",100), measurement("thread_tile_8x8",50), measurement("cublas",25)])
        rows, skipped = plot.read_results([a,b])
        summary, shapes = plot.summarize_group(rows)
        result = next(r for r in summary if r["impl"] == "thread_tile_8x8")
        self.assertFalse(skipped)
        self.assertEqual(shapes, [(128,256,16)])
        self.assertEqual(result["median_ms"],26)
        self.assertEqual(result["speedup_over_naive"],3.5)
        self.assertEqual(result["speedup_over_cublas"],0.5)
        self.assertEqual(result["naive_pairs"],2)
        self.assertAlmostEqual(result["tflops"], 2*128*256*16/(26*1e9))

    def test_never_borrows_a_baseline_from_another_file(self):
        a = self.load("a.csv",[measurement("naive",10)])
        b = self.load("b.csv",[measurement("warp_tile",2), measurement("cublas",1)])
        rows, _ = plot.read_results([a,b])
        summary, _ = plot.summarize_group(rows)
        result = next(r for r in summary if r["impl"] == "warp_tile")
        self.assertIsNone(result["speedup_over_naive"])
        self.assertEqual(result["naive_pairs"],0)
        self.assertEqual(result["speedup_over_cublas"],0.5)

    def test_contiguous_thread_layouts_remain_separate(self):
        path = self.load("layouts.csv", [
            measurement("thread_tile_8x8", 2),
            measurement("thread_tile_8x8_warp4x8", 1),
            measurement("thread_tile_8x16", 0.75),
            measurement("cublas", 0.5),
        ])
        rows, skipped = plot.read_results([path])
        summary, _ = plot.summarize_group(rows)
        self.assertFalse(skipped)
        self.assertEqual([r["impl"] for r in summary], [
            "thread_tile_8x8", "thread_tile_8x8_warp4x8", "thread_tile_8x16", "cublas"])
        self.assertEqual([r["median_ms"] for r in summary], [2, 1, 0.75, 0.5])
        self.assertEqual(summary[1]["speedup_over_cublas"], 0.5)

    def test_invalid_and_check_only_rows_are_excluded(self):
        path = self.load("bad.csv",[
            measurement("naive"), measurement("warp_tile", verified="false"),
            measurement("prefetch_reg", median_ms="nan"),
            measurement("prefetch_smem", median_ms=0),
            measurement("cublas", timing="check_only", median_ms=""),
            measurement("split_k", includes_output_clear="false"),
            measurement("block_tile", max_abs_error="inf"),
        ])
        rows, skipped = plot.read_results([path])
        self.assertEqual([r["impl"] for r in rows],["naive"])
        self.assertEqual(len(skipped),6)

    def test_contiguous_variants_pair_with_same_run_subtiles(self):
        for contiguous, subtile in plot.SUBTILE_BASELINES.items():
            with self.subTest(impl=contiguous):
                a = self.load("a.csv", [measurement(subtile, 4), measurement(contiguous, 2)])
                b = self.load("b.csv", [measurement(subtile, 20), measurement(contiguous, 5)])
                c = self.load("c.csv", [measurement(contiguous, 1)])
                rows, skipped = plot.read_results([a, b, c])
                summary, _ = plot.summarize_group(rows)
                result = next(r for r in summary if r["impl"] == contiguous)
                self.assertFalse(skipped)
                self.assertEqual(result["speedup_over_subtile"], 3)
                self.assertEqual(result["subtile_pairs"], 2)
                plot.write_report(self.root, {}, summary, [(128, 256, 16)],
                                  [str(a), str(b), str(c)], [], [])
                report = (self.root / "report.html").read_text()
                self.assertIn("Continuous 8x8 vs four subtiles", report)
                self.assertIn("3.000x", report)

    def test_contiguous_split_k_requires_output_clear(self):
        for impl in plot.SPLIT_K_IMPLS:
            with self.subTest(impl=impl):
                path = self.load("clear.csv", [measurement("naive"),
                    measurement(impl, includes_output_clear="false")])
                rows, skipped = plot.read_results([path])
                self.assertEqual([r["impl"] for r in rows], ["naive"])
                self.assertIn("output clear", skipped[0])

    def test_three_pipeline_layouts_remain_separate(self):
        for base in ("prefetch_smem", "prefetch_reg", "split_k"):
            with self.subTest(stage=base):
                names = [base, base + "_contiguous8x8", base + "_contiguous8x8_warp2x16"]
                path = self.load("layouts.csv", [measurement(name, ms) for name, ms in zip(names, (6, 3, 2))])
                rows, skipped = plot.read_results([path])
                summary, _ = plot.summarize_group(rows)
                self.assertFalse(skipped)
                self.assertEqual([r["impl"] for r in summary], names)
                self.assertEqual([r["median_ms"] for r in summary], [6, 3, 2])
                self.assertEqual([r["speedup_over_subtile"] for r in summary], [None, 2, 3])

    def test_no_valid_timings_is_an_error(self):
        path = self.load("check.csv",[measurement("naive",timing="check_only",median_ms="")])
        with self.assertRaisesRegex(ValueError,"no verified timing"):
            plot.read_results([path])

    def test_environment_keys_separate_incompatible_measurements(self):
        base = measurement("naive")
        for field in plot.ENV_FIELDS:
            with self.subTest(field=field):
                changed = dict(base, **{field:"different"})
                self.assertNotEqual(plot.environment(base),plot.environment(changed))

    def test_discovery_deduplicates_and_excludes_generated_reports(self):
        raw = self.load("run.csv",[measurement("naive")])
        generated = self.root / "report"
        generated.mkdir()
        write_fixture(generated / "collected_results.csv",[measurement("naive")])
        (self.root / "unrelated.csv").write_text("one,two\n1,2\n")
        self.assertEqual(plot.discover_inputs([self.root,raw],generated),[raw.resolve()])

    def test_duplicate_rows_do_not_count_as_repeated_runs(self):
        path = self.load("duplicate.csv",[measurement("naive",10),measurement("naive",5)])
        rows, skipped = plot.read_results([path])
        self.assertEqual(len(rows),1)
        self.assertIn("duplicate",skipped[0])

    def test_report_renders_missing_baselines_and_multiple_groups(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("optional matplotlib dependency not installed")
        path = self.load("groups.csv",[
            measurement("warp_tile",2),measurement("cublas",1),
            measurement("warp_tile",3,timing="event"),measurement("cublas",1,timing="event"),
        ])
        output = self.root / "report"
        reports, _ = plot.generate([path],output,formats=("png",))
        self.assertEqual(len(reports),2)
        for report, meta in reports:
            self.assertTrue(report.is_file())
            self.assertIn("N/A",report.read_text())
            manifest = json.loads((report.parent / "manifest.json").read_text())
            self.assertEqual(manifest["environment"]["timing"],meta["timing"])
            self.assertEqual(len(list(report.parent.glob("*.png"))),4)
            for image in report.parent.glob("*.png"):
                self.assertGreater(image.stat().st_size,1000)
        self.assertIn("group_01/report.html",(output / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
