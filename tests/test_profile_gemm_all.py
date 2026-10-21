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


class TestNcuProfile(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemm-ncu-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commands = []
        self.fail_impl = None

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
            text = SYNTHETIC_CSV
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
            self.assertEqual([r["implementation"] for r in csv.DictReader(handle)], ["prefetch_smem", "cublas"])

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

    def test_shape_validation(self):
        self.assertEqual(profile.parse_shape("128x256x48"), (128, 256, 48))
        for shape in ("127,128,16", "128,128,8", "0,128,16", "1,2", "x,y,z", "1048576,128,16"):
            with self.subTest(shape=shape), self.assertRaises(argparse.ArgumentTypeError):
                profile.parse_shape(shape)


if __name__ == "__main__":
    unittest.main()
