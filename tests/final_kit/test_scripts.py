"""Offline entry-point tests. Run via unittest discovery from the repository root."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "final-kit"
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
PWSH = os.environ.get("PWSH") or shutil.which("pwsh")


class KitFixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.kit = Path(temp.name) / "kit [v1] & space"
        shutil.copytree(SOURCE, self.kit, ignore=shutil.ignore_patterns(
            "bundles", ".cache", ".tmp", ".venv", "receipts"))
        self.profile = self.kit / "profile.json"
        profile = json.loads((self.kit / "lineage-template.json").read_bytes())
        for key, value in profile.items():
            if "REQUIRED" in value:
                profile[key] = "a" * 64 if key.endswith("digest") else "fixture"
        self.profile.write_text(json.dumps(profile))
        self.env = dict(os.environ, MODEL_ID="fixture/model", PROFILE_FILE=str(self.profile),
                        PILOT_PYTHON=str(HERE / "fake_python.py"), DRAW_SEED="1234")
        self.task = "minerva_math500"
        self.spec = self.kit / "plans/minerva_math500-smoke.json"

    def call(self, args, **env):
        return subprocess.run(args, cwd=self.kit.parent, env=self.env | env, text=True,
                              capture_output=True, timeout=45)

    def success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


class RunnerCases:
    def args(self, label="a", selection=None):
        return [*self.runner, self.task, label, ENDPOINT, "one",
                *(selection or [str(self.spec)])]

    def bundle(self, label="a"):
        return self.kit / "bundles" / self.task / label / "one"

    def manifest(self):
        return json.loads((self.bundle() / "manifest.json").read_bytes())

    def test_run_resume_and_compare(self):
        self.success(self.call(self.args()))
        original = (self.bundle() / "SHA256SUMS").read_bytes()
        self.assertIn("SKIP", self.call(self.args()).stdout)
        self.assertEqual((self.bundle() / "SHA256SUMS").read_bytes(), original)
        self.assertEqual((self.bundle() / "native-files.relative.txt").read_text().splitlines(),
                         ["native/samples_minerva_math500_2026-11-17.jsonl"])
        self.success(self.call(self.args("b"), DRAW_SEED="1235"))
        compare = ["bash", self.kit / "compare-bundles.sh", self.spec,
                   self.bundle().parent, self.bundle("b").parent, "--aa"]
        result = self.call(compare)
        self.success(result)
        self.assertEqual(json.loads(result.stdout)["expected"], 3)
        (self.bundle() / "run.log").write_text("changed")
        self.assertNotEqual(self.call(compare).returncode, 0)

    def test_tier_range(self):
        self.task = "mmlu_pro"
        self.success(self.call(self.args(selection=["--tier", "n197", "0", "2"])))
        self.assertEqual(len(self.manifest()["sample_units"]), 2)

    def test_sample_list_fits_in_one_path_argument(self):
        local_python = self.kit / ".venv/bin/python"
        local_python.parent.mkdir(parents=True)
        shutil.copy2(self.env["PILOT_PYTHON"], local_python)
        self.success(self.call(self.args(), PILOT_PYTHON="", FIXTURE_MODE="require-sample-file"))

    def test_concurrent_retry_preserves_active_attempt(self):
        with subprocess.Popen(self.args(), env=self.env | {"FIXTURE_MODE": "wait"},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as first:
            native = self.bundle() / "native"
            for _ in range(200):
                if (native / "ready").exists() or first.poll() is not None:
                    break
                time.sleep(.05)
            try:
                self.assertTrue((native / "ready").exists())
                self.assertNotEqual(self.call(self.args()).returncode, 0)
                self.assertEqual(self.manifest()["state"], "running")
            finally:
                if native.exists():
                    (native / "release").touch()
            _, err = first.communicate(timeout=15)
            self.assertEqual(first.returncode, 0, err)

    def test_failed_evaluator_preserves_status_and_logs(self):
        result = self.call(self.args(), FIXTURE_MODE="error")
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(self.manifest()["exit_status"], 23)
        self.assertEqual(self.manifest()["state"], "failed")
        self.assertIn("startup failed", (self.bundle() / "run.log").read_text())
        self.assertTrue((self.bundle() / "SHA256SUMS").is_file())
        self.assertEqual((self.bundle() / "items.jsonl").read_bytes(), b"")
        self.assertEqual(list((self.kit / ".tmp").iterdir()), [])

    def test_unrequested_native_row_fails_attempt(self):
        self.assertNotEqual(self.call(self.args(), FIXTURE_MODE="extra").returncode, 0)
        self.assertEqual(self.manifest()["state"], "failed")

    def test_private_output_never_publishes_seal(self):
        self.assertNotEqual(self.call(self.args(), FIXTURE_MODE="private").returncode, 0)
        self.assertFalse((self.bundle() / "SHA256SUMS").exists())
        self.assertEqual(self.manifest()["state"], "export-refused")

    def test_retry_rejects_changed_source_and_selection(self):
        self.success(self.call(self.args()))
        original = self.spec.read_bytes()
        self.spec.write_bytes(original + b"\n")
        self.assertNotEqual(self.call(self.args()).returncode, 0)
        self.spec.write_bytes(original)
        with (self.kit / "tasks/minerva_math500.yaml").open("a") as stream:
            print("# changed task", file=stream)
        self.assertNotEqual(self.call(self.args()).returncode, 0)

    def test_linked_bundle_parent_is_refused_before_writing(self):
        outside = self.kit.parent / "other"
        outside.mkdir()
        (self.kit / "bundles").symlink_to(outside, target_is_directory=True)
        self.assertNotEqual(self.call(self.args()).returncode, 0)
        self.assertEqual(list(outside.iterdir()), [])


class BashRunner(RunnerCases, KitFixture):
    @property
    def runner(self):
        return ["bash", self.kit / "shard-run.sh"]

    def test_term_records_signal_and_reaps_child(self):
        with subprocess.Popen(self.args(), env=self.env | {"FIXTURE_MODE": "wait"},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            ready = self.bundle() / "native/ready"
            for _ in range(200):
                if ready.exists() or proc.poll() is not None:
                    break
                time.sleep(.05)
            proc.terminate()
            _, err = proc.communicate(timeout=10)
            self.assertTrue(ready.exists(), err)
        self.assertEqual(self.manifest()["exit_status"], 128 + signal.SIGTERM)
        child = int((self.bundle() / "lm-eval.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(child, 0)

    def test_comparison_refuses_unlisted_nested_seal_and_hidden_shard(self):
        self.success(self.call(self.args()))
        self.success(self.call(self.args("b")))
        compare = ["bash", self.kit / "compare-bundles.sh", self.spec,
                   self.bundle().parent, self.bundle("b").parent]
        extra = self.bundle() / "native/SHA256SUMS"
        extra.touch()
        self.assertNotEqual(self.call(compare).returncode, 0)
        extra.unlink()
        (self.bundle().parent / ".unfinished").mkdir()
        self.assertNotEqual(self.call(compare).returncode, 0)

    def test_seal_requires_exact_inventory(self):
        self.success(self.call(self.args()))
        seal = self.bundle() / "SHA256SUMS"
        lines = seal.read_text().splitlines()
        altered = next(i for i, line in enumerate(lines) if line.endswith("  run.log"))
        path = "units.json"
        digest = hashlib.sha256((self.bundle() / path).read_bytes()).hexdigest()
        lines[altered] = f"{digest}  ./{path}"
        seal.write_text("\n".join(lines) + "\n")
        self.assertNotEqual(self.call(self.args()).returncode, 0)

    def test_compare_requires_original_stratum_mapping(self):
        self.success(self.call(self.args()))
        self.success(self.call(self.args("b")))
        strata = self.kit / "tiers/strata.json"
        mapping = json.loads(strata.read_bytes())
        index = self.manifest()["sample_units"][0]["global_index"]
        mapping[self.task][index]["subject"] = "different stratum"
        strata.write_text(json.dumps(mapping))
        result = self.call(["bash", self.kit / "compare-bundles.sh", self.spec,
                            self.bundle().parent, self.bundle("b").parent])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


@unittest.skipUnless(PWSH, "PowerShell 7 is required for the PowerShell runner tests")
class PowerShellRunner(RunnerCases, KitFixture):
    @property
    def runner(self):
        return [PWSH, "-NoLogo", "-NoProfile", "-File", self.kit / "shard-run.ps1"]

    def test_tier_via_powershell_argument_array(self):
        args = self.args(selection=["--tier", "n197", "0", "1"])
        command = """
            $ErrorActionPreference = 'Stop'
            $params = @(ConvertFrom-Json $env:SHARD_ARGS)
            & $env:RUNNER @params
            if ($env:HF_HOME -ne 'parent cache') { throw 'Runner changed the parent environment' }
            exit $LASTEXITCODE
        """
        self.success(self.call([PWSH, "-NoProfile", "-NonInteractive", "-Command", command],
                               HF_HOME="parent cache", RUNNER=str(self.runner[-1]),
                               SHARD_ARGS=json.dumps(args[len(self.runner):])))
        self.assertEqual(len(self.manifest()["sample_units"]), 1)

    def test_can_resume_bash_bundle(self):
        args = self.args()[len(self.runner):]
        self.success(self.call(["bash", self.kit / "shard-run.sh", *args]))
        result = self.call(self.args())
        self.success(result)
        self.assertIn("SKIP", result.stdout)


class Installer(KitFixture):
    def install(self, **env):
        return self.call(["bash", self.kit / "setup/install.sh"],
                         PYTHON_BIN=str(HERE / "fake_python.py"), **env)

    def test_accepts_only_python_312_before_reserving(self):
        self.assertNotEqual(self.install(FIXTURE_VERSION="Python 3.13.0").returncode, 0)
        self.assertFalse((self.kit / ".venv").exists())

    def test_existing_venv_is_preserved(self):
        self.success(self.install())
        self.assertEqual(self.install().returncode, 73)
        self.assertTrue((self.kit / "receipts/pip-check.txt").is_file())

    def test_install_refuses_linked_receipt(self):
        outside = self.kit.parent / "existing.log"
        outside.write_text("preserve")
        (self.kit / "receipts").mkdir()
        (self.kit / "receipts/install.log").symlink_to(outside)
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(outside.read_text(), "preserve")
