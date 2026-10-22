"""CPU checks of orchestration and failure handling; no GPU timing is fabricated."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_and_profile as runner


class TestBenchAndProfile(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gemm-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "run"
        self.commands = []

    def run_command(self, command, log):
        self.commands.append(command)
        log.write_text("SYNTHETIC COMMAND LOG; NO GPU MEASUREMENTS\n")

    def invoke(self, extra=(), *, fail_stage=None, profile_status=0):
        def run(command, log):
            self.run_command(command, log)
            if log.stem == fail_stage:
                raise RuntimeError("synthetic command failure")

        with mock.patch.object(runner.shutil, "which", return_value="/synthetic/ncu"), \
             mock.patch.object(runner, "run_logged", side_effect=run), \
             mock.patch.object(runner.profiling, "main", return_value=profile_status) as profile, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = runner.main(["--output", str(self.output), *extra])
        return status, profile

    def test_build_once_and_forward_shared_settings_to_both_tools(self):
        status, profile = self.invoke(["--device", "2", "--seed", "17", "--arch", "89"])
        self.assertEqual(status, 0)
        self.assertEqual([Path(c[0]).name for c in self.commands], ["cmake", "cmake", "ctest", "gemm_bench"])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=89", self.commands[0])
        self.assertIn("-DGEMM_PTXAS_VERBOSE=ON", self.commands[0])
        bench = self.commands[-1]
        ncu_args = profile.call_args.args[0]
        for args in (bench, ncu_args):
            for flag, expected in (("--device", "2"), ("--seed", "17"), ("--warmup", "2")):
                self.assertEqual(args[args.index(flag) + 1], expected)
        for flag, expected in (("--repeat", "5"), ("--batch", "1"), ("--impl", "all")):
            self.assertEqual(bench[bench.index(flag) + 1], expected)
        self.assertNotIn("--shapes", bench)  # Use all nine C++ defaults.
        self.assertIn("--skip-build", ncu_args)
        self.assertNotIn("--impls", ncu_args)  # Discover the complete registry.
        self.assertEqual(ncu_args[ncu_args.index("--shapes") + 1], "4096,8192,4096")
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "ok")

    def test_skip_build_still_runs_tests_and_overrides_timing(self):
        status, profile = self.invoke(["--skip-build", "--repeat", "3", "--batch", "2", "--timing", "event"])
        self.assertEqual(status, 0)
        self.assertEqual([Path(c[0]).name for c in self.commands], ["ctest", "gemm_bench"])
        bench = self.commands[-1]
        self.assertEqual(bench[bench.index("--repeat") + 1], "3")
        self.assertEqual(bench[bench.index("--batch") + 1], "2")
        self.assertEqual(bench[bench.index("--timing") + 1], "event")
        profile.assert_called_once()

    def test_failed_benchmark_stops_before_profiling_and_keeps_logs(self):
        status, profile = self.invoke(fail_stage="benchmark")
        self.assertEqual(status, 1)
        profile.assert_not_called()
        self.assertTrue((self.output / "benchmark.log").is_file())
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")

    def test_profile_failure_is_propagated(self):
        status, _ = self.invoke(profile_status=1)
        self.assertEqual(status, 1)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")

    def test_existing_results_are_never_overwritten(self):
        self.output.mkdir()
        manifest = self.output / "manifest.json"
        manifest.write_text("existing measurement metadata")
        status, profile = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(self.commands, [])
        profile.assert_not_called()
        self.assertEqual(manifest.read_text(), "existing measurement metadata")

    def test_real_subprocess_failure_preserves_output(self):
        log = self.output.parent / "failure.log"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "exited 7"):
            runner.run_logged([sys.executable, "-c", "print('test diagnostic'); raise SystemExit(7)"], log)
        self.assertEqual(log.read_text(), "test diagnostic\n")


if __name__ == "__main__":
    unittest.main()
