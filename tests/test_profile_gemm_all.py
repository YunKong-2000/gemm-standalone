"""CPU tests of NCU orchestration using synthetic reports, never GPU measurements."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("profile_gemm_all", ROOT / "scripts/profile_gemm_all.py")
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)
SYNTHETIC_CSV = ('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
                 '"0","synthetic_kernel","launch__registers_per_thread","register/thread","64"\n')
SYNTHETIC_WIDE_CSV = (
    '"ID","Process ID","Kernel Name","Context","Stream","launch__registers_per_thread",'
    '"gpu__time_duration.sum","l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"\n'
    '"","","","","","register/thread","nsecond","byte"\n'
    '"0","123","synthetic_kernel<8, 8>","1","7","64","9,759,400","0"\n'
    '"1","123","synthetic_second_kernel","1","7","96","1,234",""\n'
)


class TestNcuProfile(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemm-ncu-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commands = []
        self.fail_impl = None
        self.export_csv = SYNTHETIC_CSV

    def fake_run(self, command, log):
        command = [str(v) for v in command]
        self.commands.append(command)
        text = "SYNTHETIC TEST OUTPUT - NOT GPU MEASUREMENTS\n"
        if "--version" in command:
            text += "Synthetic NCU version\n"
        elif "--list" in command:
            text = "prefetch_smem\nsplit_k\ncublas\n"
        elif "--export" in command:
            impl = command[command.index("--impl") + 1]
            if impl == self.fail_impl:
                Path(log).write_text(text + "synthetic profiling failure\n")
                raise RuntimeError("synthetic failure")
            report = command[command.index("--export") + 1] + ".ncu-rep"
            Path(report).write_text(text)
        elif "--import" in command:
            text = self.export_csv
        Path(log).write_text(text)

    def test_profile_range_selects_exact_implementation_without_launch_count(self):
        with mock.patch.object(profile, "run_logged", self.fake_run):
            result, rows = profile.profile_one("ncu", Path("gemm_profile"), "split_k", (128, 256, 48), self.root,
                warmup=3, device=1, seed=2, section_set="full", cache_control="all", clock_control="base")
        self.assertEqual(result["status"], "ok")
        command = self.commands[0]
        self.assertEqual(command[command.index("--profile-from-start") + 1], "off")
        self.assertEqual(command[command.index("--impl") + 1], "split_k")
        self.assertEqual(command[command.index("--shape") + 1], "128,256,48")
        self.assertNotIn("--launch-count", command)  # cuBLAS may launch multiple device kernels.
        self.assertNotIn("--kernel-name", command)   # several translation units use gemm_kernel.
        self.assertEqual(rows[0]["implementation"], "split_k")

    def test_all_registered_implementations_are_collected_and_failures_persist(self):
        build = self.root / "build"
        build.mkdir()
        (build / "gemm_profile").touch()
        self.fail_impl = "split_k"
        self.export_csv = SYNTHETIC_WIDE_CSV
        output = self.root / "reports"
        with mock.patch.object(profile.shutil, "which", return_value="ncu"), \
             mock.patch.object(profile, "run_logged", self.fake_run):
            status = profile.main(["--skip-build", "--build-dir", str(build), "--output", str(output),
                                   "--shapes", "128,256,48"])
        self.assertEqual(status, 1)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual([r["impl"] for r in manifest["profiles"]], ["prefetch_smem", "split_k", "cublas"])
        self.assertEqual([r["status"] for r in manifest["profiles"]], ["ok", "failed", "ok"])
        with (output / "metrics_all.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["implementation"] for r in rows], ["prefetch_smem"] * 5 + ["cublas"] * 5)
        self.assertEqual([r["metric_rows"] for r in manifest["profiles"] if r["status"] == "ok"], [5, 5])

    def test_missing_report_is_a_failure_even_when_ncu_exits_zero(self):
        with mock.patch.object(profile, "run_logged"):
            result, rows = profile.profile_one("ncu", Path("gemm_profile"), "naive", (128,128,16), self.root,
                warmup=1, device=0, seed=0, section_set="full", cache_control="none", clock_control="none")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(rows, [])

    def test_metrics_banners_are_ignored_but_empty_exports_fail(self):
        path = self.root / "raw.csv"
        path.write_text("==PROF== synthetic banner\n" + SYNTHETIC_CSV)
        self.assertEqual(profile.read_metrics(path)[0]["Metric Value"], "64")
        path.write_text("==PROF== No kernels were profiled.\n")
        with self.assertRaisesRegex(RuntimeError, "no raw NCU metrics"):
            profile.read_metrics(path)

    def test_wide_metrics_preserve_units_launches_and_values(self):
        path = self.root / "raw.csv"
        path.write_text('\ufeff==PROF== synthetic banner\n' + SYNTHETIC_WIDE_CSV +
                        '\n==PROF== synthetic footer\n', encoding="utf-8")
        rows = profile.read_metrics(path)
        self.assertEqual(len(rows), 5)  # Empty metric is absent; measured zero is retained.
        self.assertEqual([r["ID"] for r in rows], ["0", "0", "0", "1", "1"])
        self.assertEqual(rows[0]["Kernel Name"], "synthetic_kernel<8, 8>")
        self.assertEqual(rows[0]["Metric Name"], "launch__registers_per_thread")
        self.assertEqual(rows[0]["Metric Unit"], "register/thread")
        self.assertEqual(rows[1]["Metric Value"], "9,759,400")
        self.assertEqual(rows[1]["Metric Unit"], "nsecond")
        self.assertEqual(rows[2]["Metric Value"], "0")
        self.assertEqual(rows[4]["Process ID"], "123")
        self.assertEqual(rows[4]["Stream"], "7")

    def test_wide_units_only_and_long_empty_values_are_not_measurements(self):
        path = self.root / "raw.csv"
        for text in ("\n".join(SYNTHETIC_WIDE_CSV.splitlines()[:2]),
                     SYNTHETIC_CSV.replace('"64"', '""')):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaisesRegex(RuntimeError, "no raw NCU metrics"):
                    profile.read_metrics(path)

    def test_multiline_kernel_names_and_repeated_headers(self):
        path = self.root / "raw.csv"
        path.write_text(SYNTHETIC_WIDE_CSV.replace('synthetic_kernel<8, 8>',
                                                 'synthetic_kernel<8,\n8>') + SYNTHETIC_CSV)
        rows = profile.read_metrics(path)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["Kernel Name"], 'synthetic_kernel<8,\n8>')
        self.assertEqual(rows[-1]["Metric Value"], "64")

    def test_reparse_recovers_old_exports_without_ncu_or_modifying_history(self):
        stem = self.root / "warp_tile__4096x4096x4096"
        Path(str(stem) + ".metrics.csv").write_text(SYNTHETIC_WIDE_CSV)
        record = dict(impl="warp_tile", shape=[4096, 4096, 4096], status="failed",
                      error="no raw NCU metrics collected in " + str(stem) + ".metrics.csv")
        manifest_path = self.root / "manifest.json"
        original = json.dumps(dict(profiles=[record], source_sha256={"original.cu": "original_hash"}))
        manifest_path.write_text(original)
        summary = self.root / "metrics_all.csv"
        summary.write_text("original summary\n")
        with mock.patch.object(profile.shutil, "which", side_effect=AssertionError("must not find NCU")), \
             mock.patch.object(profile, "run_logged", side_effect=AssertionError("must not run GPU")):
            status = profile.main(["--reparse", str(self.root)])
        self.assertEqual(status, 0)
        self.assertEqual(manifest_path.read_text(), original)
        self.assertEqual(summary.read_text(), "original summary\n")
        with (self.root / "metrics_all.reparsed.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["implementation"], "warp_tile")
        self.assertEqual(rows[0]["K"], "4096")
        self.assertEqual(rows[0]["source_report"], str(stem.resolve()) + ".ncu-rep")

    def test_reparse_does_not_mask_failed_capture_or_export(self):
        for impl in ("warp_tile", "thread_tile_8x8"):
            (self.root / (impl + "__128x128x16.metrics.csv")).write_text(SYNTHETIC_WIDE_CSV)
        records = [dict(impl="warp_tile", shape=[128, 128, 16], status="ok"),
                   dict(impl="thread_tile_8x8", shape=[128, 128, 16], status="failed",
                        error="command exited 1; see capture.log")]
        (self.root / "manifest.json").write_text(json.dumps(dict(profiles=records)))
        self.assertEqual(profile.main(["--reparse", str(self.root)]), 1)
        with (self.root / "metrics_all.reparsed.csv").open() as handle:
            self.assertEqual({r["implementation"] for r in csv.DictReader(handle)}, {"warp_tile"})

    def test_shape_validation(self):
        self.assertEqual(profile.parse_shape("128x256x48"), (128, 256, 48))
        for shape in ("127,128,16", "128,128,8", "0,128,16", "1,2", "x,y,z", "1048576,128,16"):
            with self.subTest(shape=shape), self.assertRaises(argparse.ArgumentTypeError):
                profile.parse_shape(shape)


if __name__ == "__main__":
    unittest.main()
