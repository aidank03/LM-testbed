"""Meaningful regression checks for the synthetic bench and structured graders."""
import copy
import math
import tempfile
import unittest
import json
from pathlib import Path
import numpy as np

from liner_stability import simulation, diagnostics, evaluation
from liner_stability import benchmarks as ai


class DiagnosticChecks(unittest.TestCase):
    def test_equivalent_width_limits(self):
        self.assertAlmostEqual(float(diagnostics.equivalent_width(0)), 0)
        rs = np.array([10., 100., 400.])
        ws = diagnostics.equivalent_width(rs)
        self.assertTrue(np.all(np.diff(ws) > 0))
        self.assertTrue(np.all(ws < 2*rs))
        thin = float(diagnostics.equivalent_width(100, 1e-8))
        self.assertAlmostEqual(thin / (1e-8*np.pi*100**2), 1.0, places=5)

    def test_clean_motion_recovery(self):
        obs, truth = simulation.simulate(1701, "nominal")
        _, pred = diagnostics.analyze(obs, bootstrap=24)
        self.assertLess(abs(pred["compression_nm"]["mean"]-truth["compression_nm"]), 1.5)
        self.assertLess(abs(pred["turnaround_ns"]["mean"]-truth["turnaround_ns"]), 2.0)

    def test_blur_correction_improves_known_amplitude(self):
        obs, truth = simulation.simulate(101701, "blurred")
        naive, corrected, _, _ = diagnostics.radiograph_estimates(obs)
        self.assertLess(abs(corrected["mean"]-truth["mode_amplitude_um"]), 1.0)
        self.assertGreater(abs(naive["mean"]-truth["mode_amplitude_um"]), 2.0)

    def test_low_mtf_abstention(self):
        obs, _ = simulation.simulate(8, "nominal")
        obs["calibration"]["sigma_z_pixels"] = 20
        _, corrected, _, flags = diagnostics.radiograph_estimates(obs)
        self.assertIsNone(corrected)
        self.assertTrue(flags)

    def test_dropout_does_not_get_confident_turnaround(self):
        obs, _ = simulation.simulate(201701, "pdv_dropout")
        _, corrected = diagnostics.analyze(obs, bootstrap=24)
        self.assertIsNone(corrected["turnaround_ns"])
        self.assertTrue(corrected["flags"])

    def test_known_mode_null_false_alarms(self):
        # A broad regression alarm, not a certification of a 1% operating rate.
        false_alarms = 0
        for seed in range(50, 110):
            obs, _ = simulation.simulate(seed, "no_structure")
            false_alarms += diagnostics.radiograph_estimates(obs)[2]
        self.assertLessEqual(false_alarms, 5)


class GraderChecks(unittest.TestCase):
    def setUp(self):
        obs, self.truth = simulation.simulate(1701, "nominal")
        _, self.pred = diagnostics.analyze(obs, bootstrap=24)

    def test_units_rejected(self):
        p = copy.deepcopy(self.pred)
        p["compression_nm"]["unit"] = "m"
        with self.assertRaises(ValueError):
            evaluation.score([self.truth], [p])

    def test_nan_rejected(self):
        p = copy.deepcopy(self.pred)
        p["compression_nm"]["mean"] = float("nan")
        with self.assertRaises(ValueError):
            evaluation.score([self.truth], [p])

    def test_duplicate_and_missing_cases_rejected(self):
        with self.assertRaises(ValueError):
            evaluation.score([self.truth], [self.pred, self.pred])
        with self.assertRaises(ValueError):
            evaluation.score([self.truth], [])

    def test_abstention_remains_in_denominator(self):
        p = copy.deepcopy(self.pred)
        p["compression_nm"] = None
        report = evaluation.score([self.truth], [p])["all"]["compression_nm"]
        self.assertEqual(report["answer_rate"], 0)
        self.assertNotIn("coverage90", report)

    def test_broader_interval_is_penalized(self):
        narrow, broad = copy.deepcopy(self.pred), copy.deepcopy(self.pred)
        y = self.truth["compression_nm"]
        narrow["compression_nm"] = diagnostics.interval(y, 1, "nm")
        broad["compression_nm"] = diagnostics.interval(y, 100, "nm")
        s1 = evaluation.score([self.truth], [narrow])["all"]["compression_nm"]["mean_interval_score90"]
        s2 = evaluation.score([self.truth], [broad])["all"]["compression_nm"]["mean_interval_score90"]
        self.assertGreater(s2, s1)

    def test_ai_export_separates_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            ai.export(Path(directory))
            public = json.loads((Path(directory)/"candidate_inputs.json").read_text())
            self.assertTrue(all("expected" not in r and "rationale" not in r for r in public))

    def test_ai_grader_correct_and_wrong_units(self):
        key = ai.cases()
        pred = [{"case_id": c["case_id"], "answer": c["expected"], "unit": c.get("unit")} for c in key]
        self.assertEqual(ai.score_answers(key, pred)["n_correct"], len(key))
        pred[0]["unit"] = "T"
        self.assertEqual(ai.score_answers(key, pred)["n_correct"], len(key)-1)
        pred[1]["answer"] = float("inf")
        self.assertEqual(ai.score_answers(key, pred)["n_correct"], len(key)-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
