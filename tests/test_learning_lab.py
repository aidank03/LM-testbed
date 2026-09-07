import json
from pathlib import Path
import tempfile
import unittest

from factor.learn.store import LearningRun

try:
    from factor.learn.experiments import choose_device, run_experiment
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    choose_device = run_experiment = None


HAS_TORCH = run_experiment is not None


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch dependency not installed")
class LearningExperimentsTest(unittest.TestCase):
    def test_line_fit_passes_frozen_gate_without_using_test_for_training(self):
        events = []
        result = run_experiment("line_fit", {"device": "cpu", "delay": 0}, events.append)
        self.assertTrue(result["gate"]["accepted"])
        self.assertFalse(result["data"]["test_used_during_training"])
        self.assertLess(result["metrics"]["weight_absolute_error"], 0.15)
        self.assertEqual(len([event for event in events if event["kind"] == "epoch"]), 80)

    def test_line_fit_is_reproducible(self):
        config = {"device": "cpu", "delay": 0, "seed": 19, "epochs": 40}
        first = run_experiment("line_fit", config)
        second = run_experiment("line_fit", config)
        self.assertEqual(first["learned"], second["learned"])
        self.assertEqual(first["metrics"], second["metrics"])

    def test_motion_loop_exposes_information_change(self):
        result = run_experiment("motion_loop", {"device": "cpu", "delay": 0})
        stages = {stage["id"]: stage for stage in result["stages"]}
        self.assertEqual(stages["baseline"]["information"], "apparent only")
        self.assertEqual(stages["more_data"]["information"], "apparent only")
        self.assertIn("reference", stages["reference"]["information"])
        self.assertTrue(result["gate"]["accepted"])
        self.assertFalse(result["data"]["test_used_during_training"])
        self.assertGreater(result["gate"]["mae_improvement_nm"], 10.0)

    def test_invalid_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "epochs"):
            run_experiment("line_fit", {"device": "cpu", "epochs": 2})
        with self.assertRaisesRegex(ValueError, "unknown lesson"):
            run_experiment("unknown", {})
        self.assertEqual(choose_device("cpu"), "cpu")


class LearningStoreTest(unittest.TestCase):
    def test_run_preserves_request_events_result_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            run = LearningRun(Path(directory), "line_fit", {"epochs": 20})
            run.emit({"kind": "epoch", "epoch": 1})
            run.emit({"kind": "epoch", "epoch": 2})
            run.complete({"score": 0.25})
            self.assertEqual(len((run.path / "events.jsonl").read_text().splitlines()), 2)
            self.assertEqual(json.loads((run.path / "result.json").read_text()), {"score": 0.25})
            manifest = json.loads((run.path / "run.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(manifest["result_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
