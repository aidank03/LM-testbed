import copy
import tempfile
import unittest
from pathlib import Path
from factor import evaluation as v1
from factor import evaluation_v2 as v2
from factor.contracts import ContractError


class ExplicitTaskChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.paths = v2.create_suite(self.directory / "v2")

    def test_explicit_rules_are_frozen_in_every_public_case(self):
        public = v2._validate_public(self.paths["public_cases"])
        self.assertEqual(len(public["cases"]), 18)
        self.assertNotEqual(public["contract_hash"], v1._hash(v1.CONTRACT))
        for c in public["cases"]:
            self.assertEqual(c["question"], v2.QUESTION)
            self.assertIn("NEGATIVE displacement is inward", c["question"])
            self.assertIn("1.959963984540054", c["question"])
            self.assertIn("upper <= T", c["question"])
            self.assertIn("lower > T", c["question"])
            self.assertIn("no particular tool sequence is required", c["question"])
        public["cases"][0]["question"] = v1.QUESTION
        with self.assertRaises(ContractError):
            v2.baseline_results(public)

    def test_v1_remains_supported_and_unchanged(self):
        p = v1.create_suite(self.directory / "v1", n_per_condition=1)
        r = v1.score_agent_runs(p["public_cases"], p["evaluator_truth"], v1.baseline_results(p["public_cases"]))
        self.assertNotIn("complete_runs", r["provisional_gates"])
        with self.assertRaises(ContractError):
            v2.baseline_results(p["public_cases"])

    def test_missing_run_and_missing_evidence_fail_separate_gates(self):
        runs = v2.baseline_results(self.paths["public_cases"])
        report = v2.score_agent_runs(self.paths["public_cases"], self.paths["evaluator_truth"], runs)
        self.assertTrue(report["provisional_gates"]["complete_runs"])
        self.assertTrue(report["provisional_gates"]["evidence_presence"])
        bad = copy.deepcopy(runs)
        bad[0]["result"]["final"]["evidence_ids"] = ["fabricated"]
        report = v2.score_agent_runs(self.paths["public_cases"], self.paths["evaluator_truth"], bad)
        self.assertTrue(report["provisional_gates"]["complete_runs"])
        self.assertFalse(report["provisional_gates"]["evidence_presence"])
        report = v2.score_agent_runs(self.paths["public_cases"], self.paths["evaluator_truth"], runs[1:])
        self.assertFalse(report["provisional_gates"]["complete_runs"])

    def test_truth_does_not_cross_versions_or_public_boundary(self):
        with self.assertRaises(ContractError):
            v2.baseline_results(self.paths["evaluator_truth"])
        p = v1.create_suite(self.directory / "v1", n_per_condition=2)
        with self.assertRaises(ContractError):
            v2.score_agent_runs(self.paths["public_cases"], p["evaluator_truth"], [])


if __name__ == "__main__":
    unittest.main()
