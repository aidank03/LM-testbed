"""Evaluator tests use negative controls and independent arithmetic answers."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from factor.contracts import ContractError
from factor.evaluation import create_suite, score_agent_runs, baseline_results, compare_methods


def load(path):
    return json.loads(Path(path).read_text())


class EvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp_path = Path(self.temp.name)
        self.suite = create_suite(self.tmp_path / "suite", seed=7, n_per_condition=8)

    def test_separation_reproducibility_and_split_groups(self):
        tmp_path, suite = self.tmp_path, self.suite
        a = create_suite(tmp_path / "a", seed=3, n_per_condition=2)
        b = create_suite(tmp_path / "b", seed=3, n_per_condition=2)
        assert load(a["public_cases"]) == load(b["public_cases"])
        ids = []
        for split in ("train", "development", "final"):
            p = create_suite(tmp_path / split, seed=3, n_per_condition=2, split=split)
            ids.append({c["case_id"] for c in load(p["public_cases"])["cases"]})
        assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
        with self.assertRaises(ContractError):
            baseline_results(a["evaluator_truth"])
        with self.assertRaises(ContractError):
            create_suite(tmp_path / "a")
        public = load(a["public_cases"])
        public["cases"][0]["observation"]["truth_displacement_m"] = 0
        with self.assertRaises(ContractError):
            baseline_results(public)


    def test_generator_does_not_use_candidate_for_labels(self):
        tmp_path, suite = self.tmp_path, self.suite
        def prohibited(*args, **kwargs):
            raise AssertionError("generator called candidate")
        with patch("factor.motion.infer_motion", prohibited):
            p = create_suite(tmp_path / "independent", n_per_condition=1)
        assert len(load(p["public_cases"])["cases"]) == 9


    def test_counterfactuals_are_indistinguishable_without_reference(self):
        tmp_path, suite = self.tmp_path, self.suite
        pub = {c["case_id"]: c["observation"] for c in load(suite["public_cases"])["cases"]}
        truth = load(suite["evaluator_truth"])["cases"]
        pair = next(t["pair_id"] for t in truth if t["condition"] == "ambiguous")
        twins = [t for t in truth if t["pair_id"] == pair and t["condition"] == "ambiguous"]
        assert twins[0]["physical_decision"] != twins[1]["physical_decision"]
        a, b = [dict(pub[t["case_id"]]) for t in twins]
        a.pop("observation_id"); b.pop("observation_id")
        assert a == b
        assert {t["ambiguity_reason"] for t in twins} == {"structural"}


    def test_known_arithmetic_and_deliberately_wrong_confidence(self):
        tmp_path, suite = self.tmp_path, self.suite
        results = []
        for t in load(suite["evaluator_truth"])["cases"]:
            x = t["displacement_m"]
            decision = "compression_not_supported" if t["physical_decision"] == "compression_supported" else "compression_supported"
            results.append({"case_id": t["case_id"], "result": {"execution": "completed", "final": {"decision": decision, "evidence_ids": ["invented"]},
                           "numerical_result": {"unit": "m", "point_m": x + 5e-9, "interval_m": [x + 4e-9, x + 6e-9]}}})
        report = score_agent_runs(suite["public_cases"], suite["evaluator_truth"], results)
        for metric in ("bias_nm", "mae_nm", "rmse_nm"):
            self.assertAlmostEqual(report["overall"][metric], 5)
        self.assertAlmostEqual(report["overall"]["mean_interval_width_nm"], 2)
        assert report["overall"]["coverage"]["rate"] == 0
        assert report["overall"]["wrong_decisive"]["rate"] == 1
        assert report["overall"]["evidence_present"]["rate"] == 0


    def test_missing_duplicate_unknown_and_invalid_predictions(self):
        tmp_path, suite = self.tmp_path, self.suite
        missing = score_agent_runs(suite["public_cases"], suite["evaluator_truth"], [])
        assert missing["overall"]["completed"]["rate"] == 0
        assert missing["overall"]["coverage"]["rate"] is None
        assert missing["provisional_gates"]["structural_abstention"] is False
        runs = baseline_results(suite["public_cases"])
        for bad in (runs + [runs[0]], [{"case_id": "unknown", "result": {}}]):
            with self.assertRaises(ContractError):
                score_agent_runs(suite["public_cases"], suite["evaluator_truth"], bad)
        bad = copy.deepcopy(runs)
        bad[0]["result"]["numerical_result"] = {"unit": "nm", "point_m": 1, "interval_m": [0, 2]}
        report = score_agent_runs(suite["public_cases"], suite["evaluator_truth"], bad)
        assert sum(r["failure"] == "invalid_numerical_output" for r in report["rows"]) == 1


    def test_baselines_fail_on_unseen_drift_without_fake_detectability(self):
        tmp_path, suite = self.tmp_path, self.suite
        report = compare_methods(suite["public_cases"], suite["evaluator_truth"], {
            method: baseline_results(suite["public_cases"], method) for method in ("naive", "bounded")})
        bounded, naive = (report["methods"][m] for m in ("bounded", "naive"))
        assert bounded["regimes"]["ambiguous"]["abstained"]["rate"] == 1
        assert naive["regimes"]["ambiguous"]["abstained"]["rate"] == 0
        for condition in ("unseen_drift", "reference_mismatch"):
            assert bounded["regimes"][condition]["contract_correct"]["rate"] == 1
            assert bounded["regimes"][condition]["wrong_decisive"]["rate"] == 1
            assert bounded["regimes"][condition]["coverage"]["rate"] == 0
        assert bounded["regimes"]["reference"]["coverage"]["rate"] >= .90
        assert bounded["regimes"]["ambiguous"]["abstained"]["wilson_95"] is None
        assert bounded["regimes"]["ambiguous"]["abstained"]["experiment_all_pass"]["total"] == 8


    def test_always_abstaining_cannot_pass_usefulness(self):
        tmp_path, suite = self.tmp_path, self.suite
        runs = [{"case_id": c["case_id"], "result": {"execution": "completed", "final": {"decision": "ambiguous", "evidence_ids": []}}}
                for c in load(suite["public_cases"])["cases"]]
        report = score_agent_runs(suite["public_cases"], suite["evaluator_truth"], runs)
        assert report["provisional_gates"]["structural_abstention"] is True
        assert report["provisional_gates"]["clean_usefulness"] is False
        assert report["overall"]["abstention_precision"]["rate"] < 1


    def test_frozen_contract_and_one_use_final(self):
        tmp_path, suite = self.tmp_path, self.suite
        p = create_suite(tmp_path / "final", split="final", n_per_condition=1)
        before = Path(p["contract"]).read_bytes()
        runs = baseline_results(p["public_cases"])
        first = score_agent_runs(p["public_cases"], p["evaluator_truth"], runs)
        second = score_agent_runs(p["public_cases"], p["evaluator_truth"], runs)
        assert first["evaluation_use"] == "one_use_final_now_exposed"
        assert second["evaluation_use"] == "exposed_final_not_confirmatory"
        assert Path(p["contract"]).read_bytes() == before
        manifest = load(p["contract"])
        manifest["contract"]["gates"]["matched_mae_nm_max"] = 10000
        Path(p["contract"]).write_text(json.dumps(manifest))
        with self.assertRaises(ContractError):
            baseline_results(p["public_cases"])

if __name__ == "__main__":
    unittest.main()
