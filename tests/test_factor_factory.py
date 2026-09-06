import copy
import json
from pathlib import Path
import tempfile
import unittest

from factor.factory import artifact_hash, assess_candidate, write_decision


class FactoryChecks(unittest.TestCase):
    def setUp(self):
        self.contract = {"schema_version": 1, "task_id": "fixture-provenance",
            "metric_scope": "synthetic_provenance_classification", "minimum_independent_units": 4,
            "required_evidence_gates": ["evaluator_verified"],
            "metrics": {"accuracy": {"min": 0.8, "min_delta": 0.1},
                        "invalid_label_rate": {"max": 0}}}
        self.reference = {"candidate_id": "base", "completed": True,
            "contract_sha256": artifact_hash(self.contract),
            "task_id": self.contract["task_id"], "metric_scope": self.contract["metric_scope"],
            "independent_units": 4, "test_set_sha256": "a"*64,
            "evidence_gates": {"evaluator_verified": {"passed": True, "artifact_sha256": "b"*64,
                                 "verification": "deterministic_evaluator"}},
            "metrics": {"accuracy": 0.5, "invalid_label_rate": 0}}
        self.candidate = copy.deepcopy(self.reference)
        self.candidate.update(candidate_id="candidate", metrics={"accuracy": 0.9, "invalid_label_rate": 0})

    def test_eligible_is_not_scientific_validation(self):
        result = assess_candidate(self.reference, self.candidate, self.contract)
        self.assertEqual(result["status"], "eligible_for_local_release")
        self.assertFalse(result["scientific_validation_established"])

    def test_one_failed_gate_cannot_be_offset(self):
        self.candidate["metrics"]["invalid_label_rate"] = 0.01
        self.assertEqual(assess_candidate(self.reference, self.candidate, self.contract)["status"], "blocked")

    def test_families_not_frames_control_sample_gate(self):
        self.candidate["independent_units"] = 1
        self.candidate["metrics"]["n"] = 100000
        result = assess_candidate(self.reference, self.candidate, self.contract)
        self.assertTrue(any("insufficient independent" in r for r in result["reasons"]))

    def test_classifier_does_not_become_motion(self):
        self.candidate["metric_scope"] = "motion_reconstruction"
        self.assertEqual(assess_candidate(self.reference, self.candidate, self.contract)["status"], "blocked")

    def test_posthoc_contract_change_is_blocked(self):
        self.contract["metrics"]["accuracy"]["min"] = 0.1
        result = assess_candidate(self.reference, self.candidate, self.contract)
        self.assertTrue(any("contract binding" in r for r in result["reasons"]))

    def test_unvalidated_llm_judge_is_not_evidence(self):
        self.candidate["evidence_gates"]["evaluator_verified"]["verification"] = "llm_judge"
        self.assertEqual(assess_candidate(self.reference, self.candidate, self.contract)["status"], "blocked")

    def test_distinct_or_incomplete_evaluation_is_blocked(self):
        for field, value in [("completed", False), ("test_set_sha256", "c"*64)]:
            candidate = copy.deepcopy(self.candidate)
            candidate[field] = value
            self.assertEqual(assess_candidate(self.reference, candidate, self.contract)["status"], "blocked")

    def test_decision_preserves_sources_and_refuses_overwrite(self):
        original = copy.deepcopy(self.candidate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"local_release.json"
            result = write_decision(self.reference, self.candidate, self.contract, path)
            self.assertEqual(json.loads(path.read_text()), result)
            with self.assertRaises(FileExistsError):
                write_decision(self.reference, self.candidate, self.contract, path)
        self.assertEqual(self.candidate, original)


if __name__ == "__main__":
    unittest.main()
