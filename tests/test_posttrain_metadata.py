"""Offline metadata regressions; importing this helper never loads ML models."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/local_model"
SPEC = importlib.util.spec_from_file_location("factor_run_metadata_fixture", EXPERIMENT / "run_metadata.py")
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)


class PosttrainMetadataTests(unittest.TestCase):
    def test_other_machine_reports_detected_facts_and_unknowns(self):
        with patch.object(metadata.platform, "system", return_value="Linux"), \
                patch.object(metadata.platform, "release", return_value="fixture-release"), \
                patch.object(metadata.platform, "machine", return_value="x86_64"), \
                patch.object(metadata.os, "cpu_count", return_value=16):
            result = metadata.hardware_metadata("cpu")
        self.assertEqual(result["os"], "Linux")
        self.assertEqual(result["os_release"], "fixture-release")
        self.assertEqual(result["architecture"], "x86_64")
        self.assertEqual(result["logical_cpu_count"], 16)
        self.assertEqual(result["selected_torch_device"], "cpu")
        for key in ("cpu_model", "physical_cpu_cores", "memory_bytes"):
            self.assertEqual(result[key], "unknown")

    def test_unavailable_values_are_explicitly_unknown(self):
        with patch.object(metadata.platform, "system", return_value=""), \
                patch.object(metadata.platform, "release", return_value=""), \
                patch.object(metadata.platform, "machine", return_value=""), \
                patch.object(metadata.os, "cpu_count", return_value=None):
            result = metadata.hardware_metadata("mps")
        self.assertEqual(result["selected_torch_device"], "mps")
        self.assertTrue(all(value == "unknown" for key, value in result.items()
                            if key != "selected_torch_device"))

    def test_denied_probes_do_not_invent_hardware(self):
        with patch.object(metadata.platform, "system", side_effect=OSError), \
                patch.object(metadata.platform, "release", side_effect=OSError), \
                patch.object(metadata.platform, "machine", side_effect=OSError), \
                patch.object(metadata.os, "cpu_count", side_effect=OSError):
            result = metadata.hardware_metadata("cpu")
        self.assertEqual(result["selected_torch_device"], "cpu")
        self.assertTrue(all(value == "unknown" for key, value in result.items()
                            if key != "selected_torch_device"))

    def test_original_script_snapshot_matches_measured_manifest(self):
        run = EXPERIMENT / "posttrain_run"
        manifest = json.loads((run / "manifest.json").read_text())
        snapshot = (run / "source_snapshot/posttrain.py").read_bytes()
        self.assertEqual(hashlib.sha256(snapshot).hexdigest(), manifest["source_sha256"])


if __name__ == "__main__":
    unittest.main()
