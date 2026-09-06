"""Synthetic fixed-window evaluator. Never imported by candidate inference.

Published suites are development data. Final scoring records exposure separately
from the immutable contract; an exposed final suite cannot be a fresh final test.
"""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from .contracts import ContractError
from .motion import DECISIONS, validate_observation

NM = 1e-9
Z95 = 1.959963984540054
QUESTION = "Assess whether fixed-window radial displacement supports compression beyond the stated resolution."
CONDITIONS = ("clean", "bounded_drift", "unseen_drift", "ambiguous", "reference", "reference_noisy", "reference_mismatch")
CONTRACT = {
    "schema_version": "factor-evaluation-contract/1", "task": "motion.fixed_window/1",
    "quantity": "signed radial displacement over [0, 100 ns]; inward is negative",
    "resolution_m": 10 * NM, "noise_coverage": 0.95,
    "allowed_information": "Only public observation and question; no truth, condition, or generator seed.",
    "label_rule": "Unbounded nuisance => ambiguous. Otherwise Gaussian 95% interval enlarged by the supplied nuisance bound; upper <= -resolution => compression_supported; lower > -resolution => compression_not_supported; else ambiguous.",
    "mismatch_rule": "Contract decision uses advertised information even under undetectable mismatch; physical false confidence is scored separately.",
    "acceptance_status": "provisional engineering gates; no facility decision requirement available",
    "gates": {"matched_mae_nm_max": 10, "matched_abs_bias_nm_max": 3,
              "matched_coverage_min": 0.90, "wrong_decisive_rate_max": 0.05,
              "contract_decision_accuracy_min": 0.95, "structural_abstention_min": 1.0,
              "clean_decisive_rate_min": 0.90},
    "interpretation": "No weighted score. Confidence sets conditional on supplied noise/bounds. No melting, mechanism, raw-PDV, liner-transfer, or real-data validation claim.",
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _label(obs):
    """Evaluator contract oracle; no inference implementation or latent labels."""
    ref = obs["reference"]
    bound = ref["residual_bound_m"] if ref is not None else obs["drift_bound_m"]
    if bound is None:
        return "ambiguous", "structural"
    center = obs["apparent_displacement_m"] - (ref["displacement_m"] if ref else 0)
    variance = obs["noise_sigma_m"] ** 2
    if ref:
        variance += ref["sigma_m"] ** 2 - 2 * ref["noise_covariance_m2"]
    radius = Z95 * math.sqrt(max(0, variance)) + bound
    if center + radius <= -obs["resolution_m"]:
        return "compression_supported", "resolved"
    if center - radius > -obs["resolution_m"]:
        return "compression_not_supported", "resolved"
    return "ambiguous", "boundary_uncertainty"


def create_suite(out, seed=20260905, n_per_condition=24, split="development"):
    """Write frozen contract first, then strictly separate observations and truth."""
    if split not in ("train", "development", "final") or type(n_per_condition) is not int or n_per_condition < 1:
        raise ContractError("Use train/development/final and a positive integer count")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ContractError("Suite output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    paths = {key: out / filename for key, filename in (("contract", "contract.json"), ("public_cases", "public_cases.json"), ("evaluator_truth", "evaluator_truth.json"), ("exposure", "exposure.json"))}
    digest = _hash(CONTRACT)
    _write(paths["contract"], {"contract": CONTRACT, "contract_hash": digest})
    rng = random.Random(int(_hash([seed, split, "factor-generator/1"]), 16))
    public, truth = [], []

    def add(condition, x, y, sigma, bound, ref=None, experiment=None, pair=None):
        cid = _hash([split, rng.getrandbits(256)])[:24]
        obs = {"schema_version": "motion-observation/1", "observation_id": cid,
               "apparent_displacement_m": y, "noise_sigma_m": sigma,
               "drift_bound_m": bound, "reference": ref, "window_s": [0.0, 100e-9],
               "resolution_m": CONTRACT["resolution_m"], "source_kind": "synthetic_reconstructed",
               "calibration_id": "synthetic-contract-calibration/1"}
        validate_observation(obs)
        expected, reason = _label(obs)
        public.append({"case_id": cid, "observation": obs, "question": QUESTION})
        truth.append({"case_id": cid, "condition": condition, "experiment_id": experiment or cid,
                      "pair_id": pair, "displacement_m": x, "expected_decision": expected,
                      "ambiguity_reason": reason, "physical_decision": "compression_supported" if x <= -obs["resolution_m"] else "compression_not_supported",
                      "model_mismatch": condition in ("unseen_drift", "reference_mismatch")})

    for _ in range(n_per_condition):
        # Exact twins share measured signal/noise but have conflicting material motion.
        pair = _hash([split, rng.getrandbits(256)])[:24]
        y = -5 * NM + rng.gauss(0, NM)
        for x in (-35 * NM, 20 * NM):
            add("ambiguous", x, y, NM, None, experiment=pair, pair=pair)
            ref = {"displacement_m": -5 * NM - x + rng.gauss(0, NM), "sigma_m": NM,
                   "residual_bound_m": NM, "noise_covariance_m2": 0.0}
            add("reference", x, y, NM, None, ref, experiment=pair, pair=pair)
        for condition in ("clean", "bounded_drift", "unseen_drift", "reference_noisy", "reference_mismatch"):
            x = rng.choice((-1, 1)) * rng.uniform(25, 55) * NM
            sigma, bound, drift, ref = NM, 0.0, 0.0, None
            if condition == "bounded_drift":
                bound, drift = 8 * NM, rng.uniform(-8, 8) * NM
            if condition == "unseen_drift":
                drift = -2 * x  # wrong-sign apparent motion; indistinguishable from another clean state
            if condition.startswith("reference_"):
                drift, bound = rng.uniform(-50, 50) * NM, None
                rsigma = 25 * NM if condition == "reference_noisy" else NM
                mismatch = 0 if condition == "reference_noisy" else 2 * x
                ref = {"displacement_m": drift + mismatch + rng.gauss(0, rsigma),
                       "sigma_m": rsigma, "residual_bound_m": NM, "noise_covariance_m2": 0.0}
            add(condition, x, x + drift + rng.gauss(0, sigma), sigma, bound, ref)
    rng.shuffle(public)
    _write(paths["public_cases"], {"schema_version": "factor-cases/1", "split": split, "contract_hash": digest, "cases": public})
    _write(paths["evaluator_truth"], {"schema_version": "factor-truth/1", "split": split, "contract_hash": digest,
                                     "public_hash": _hash(public), "generator_version": "factor-generator/1", "seed": seed, "cases": truth})
    _write(paths["exposure"], {"status": "unexposed", "scoring_count": 0})
    return paths


def _validate_public(value):
    obj = _read(value)
    if not isinstance(obj, dict) or set(obj) != {"schema_version", "split", "contract_hash", "cases"} or obj["schema_version"] != "factor-cases/1":
        raise ContractError("Expected public cases, not evaluator truth or extra metadata")
    ids = set()
    for case in obj["cases"]:
        if set(case) != {"case_id", "observation", "question"} or case["case_id"] in ids:
            raise ContractError("Invalid public case fields or duplicate ID")
        validate_observation(case["observation"])
        if case["case_id"] != case["observation"]["observation_id"]:
            raise ContractError("Case and observation IDs differ")
        ids.add(case["case_id"])
    if not ids or obj["contract_hash"] != _hash(CONTRACT):
        raise ContractError("Empty cases or unknown contract")
    if isinstance(value, (str, Path)):
        manifest = _read(Path(value).parent / "contract.json")
        if manifest.get("contract_hash") != obj["contract_hash"] or _hash(manifest.get("contract")) != obj["contract_hash"]:
            raise ContractError("Frozen contract manifest failed verification")
    return obj


def baseline_results(public_cases, method="bounded"):
    """Candidate execution only; never supplies truth to infer_motion."""
    from .motion import infer_motion
    cases = _validate_public(public_cases)["cases"]
    results = []
    for c in cases:
        r = infer_motion(c["observation"], method)
        aid = "artifact_" + _hash(r)[:24]
        results.append({"case_id": c["case_id"], "result": {"execution": "completed", "final": {
            "decision": r["decision"], "evidence_ids": [aid]}, "numerical_result": r,
            "artifacts": {aid: {"kind": "motion_result", "data": r}},
            "usage": {"cloud_cost_usd": 0.0, "tool_calls": 0, "model_calls": 0, "accounting_complete": True}}})
    return results


def _wilson(successes, total):
    if not total:
        return None
    p, z2 = successes / total, Z95 ** 2
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = Z95 * math.sqrt(p * (1 - p) / total + z2 / (4 * total ** 2)) / (1 + z2 / total)
    return [max(0, center - radius), min(1, center + radius)]


def _rate(rows, key):
    eligible = [r for r in rows if r[key] is not None]
    groups = defaultdict(list)
    for r in eligible:
        groups[r["experiment_id"]].append(bool(r[key]))
    successes = sum(bool(r[key]) for r in eligible)
    independent = len(groups) == len(eligible)
    return {"successes": successes, "total": len(eligible), "rate": successes / len(eligible) if eligible else None,
            "wilson_95": _wilson(successes, len(eligible)) if independent else None,
            "experiment_all_pass": {"successes": sum(all(v) for v in groups.values()), "total": len(groups),
                                    "wilson_95": _wilson(sum(all(v) for v in groups.values()), len(groups))},
            "dependence_note": None if independent else "Paired cases: marginal Wilson omitted; all-pass interval uses independent experiment groups."}


def _summarize(rows):
    errors = [r["error_nm"] for r in rows if r["error_nm"] is not None]
    widths = [r["width_nm"] for r in rows if r["width_nm"] is not None]
    return {"cases": len(rows), "numerical_outputs": len(errors),
            "bias_nm": sum(errors) / len(errors) if errors else None,
            "mae_nm": sum(abs(v) for v in errors) / len(errors) if errors else None,
            "rmse_nm": math.sqrt(sum(v * v for v in errors) / len(errors)) if errors else None,
            "mean_interval_width_nm": sum(widths) / len(widths) if widths else None,
            "wrong_decisive_given_decisive": _rate([r for r in rows if r["decisive"]], "wrong_decisive"),
            "abstention_precision": _rate([r for r in rows if r["abstained"]], "contract_correct"),
            "required_abstention_recall": _rate([r for r in rows if r["expected_decision"] == "ambiguous"], "abstained"),
            **{k: _rate(rows, k) for k in ("completed", "contract_correct", "coverage", "abstained", "decisive", "wrong_decisive", "evidence_present")}}


def _score(public_cases, evaluator_truth, results):
    public, truth = _validate_public(public_cases), _read(evaluator_truth)
    keys = {"schema_version", "split", "contract_hash", "public_hash", "generator_version", "seed", "cases"}
    if not isinstance(truth, dict) or set(truth) != keys or truth["schema_version"] != "factor-truth/1":
        raise ContractError("Expected evaluator truth schema")
    if any(truth[k] != public[k] for k in ("split", "contract_hash")) or truth["public_hash"] != _hash(public["cases"]):
        raise ContractError("Truth and public contract/data do not match")
    case_map = {c["case_id"]: c for c in public["cases"]}
    truth_fields = {"case_id", "condition", "experiment_id", "pair_id", "displacement_m", "expected_decision", "ambiguity_reason", "physical_decision", "model_mismatch"}
    for t in truth["cases"]:
        if set(t) != truth_fields or t["condition"] not in CONDITIONS or type(t["displacement_m"]) not in (int, float) or not math.isfinite(t["displacement_m"]):
            raise ContractError("Invalid evaluator truth row")
    if len({t["case_id"] for t in truth["cases"]}) != len(truth["cases"]) or {t["case_id"] for t in truth["cases"]} != set(case_map):
        raise ContractError("Truth IDs must match public cases exactly")
    for t in truth["cases"]:
        obs = case_map[t["case_id"]]["observation"]
        physical = "compression_supported" if t["displacement_m"] <= -obs["resolution_m"] else "compression_not_supported"
        if _label(obs) != (t["expected_decision"], t["ambiguity_reason"]) or physical != t["physical_decision"]:
            raise ContractError("Truth labels disagree with the frozen contract")
    submitted = {}
    for record in _read(results):
        if not isinstance(record, dict) or set(record) != {"case_id", "result"} or record["case_id"] not in case_map or record["case_id"] in submitted:
            raise ContractError("Duplicate, unknown, or malformed result record")
        submitted[record["case_id"]] = record["result"]
    rows, usage = [], defaultdict(float)
    for t in truth["cases"]:
        cid, obs = t["case_id"], case_map[t["case_id"]]["observation"]
        result = submitted.get(cid, {})
        final = result.get("final") or {} if isinstance(result, dict) else {}
        completed = isinstance(result, dict) and result.get("execution") == "completed" and isinstance(final, dict) and final.get("decision") in DECISIONS
        decision = final.get("decision") if completed else None
        error = width = covered = None
        invalid = None
        numeric = result.get("numerical_result") if isinstance(result, dict) else None
        if numeric and isinstance(numeric, dict) and numeric.get("interval_m") is not None:
            try:
                point, interval = numeric["point_m"], numeric["interval_m"]
                vals = [point, *interval]
                if numeric.get("unit") != "m" or len(interval) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals) or not interval[0] <= point <= interval[1]:
                    raise ValueError("Invalid point, interval, or units")
                error, width = (point - t["displacement_m"]) / NM, (interval[1] - interval[0]) / NM
                covered = interval[0] <= t["displacement_m"] <= interval[1]
            except (KeyError, TypeError, ValueError):
                invalid, completed, decision = "invalid_numerical_output", False, None
        elif numeric is not None and not isinstance(numeric, dict):
            invalid, completed, decision = "invalid_numerical_output", False, None
        elif isinstance(numeric, dict) and (numeric.get("unit") != "m" or numeric.get("point_m") is not None):
            invalid, completed, decision = "point_requires_finite_interval_in_m", False, None
        decisive = decision in ("compression_supported", "compression_not_supported")
        evidence = final.get("evidence_ids", []) if isinstance(final, dict) else []
        artifacts = result.get("artifacts", {}) if isinstance(result, dict) else {}
        rows.append({"case_id": cid, "condition": t["condition"], "experiment_id": t["experiment_id"],
                     "model_mismatch": t["model_mismatch"], "ambiguity_reason": t["ambiguity_reason"],
                     "expected_decision": t["expected_decision"], "decision": decision,
                     "completed": completed, "contract_correct": completed and decision == t["expected_decision"],
                     "abstained": completed and decision == "ambiguous", "decisive": decisive,
                     "wrong_decisive": decisive and decision != t["physical_decision"],
                     "coverage": covered, "error_nm": error, "width_nm": width,
                     "evidence_present": completed and isinstance(evidence, list) and bool(evidence) and isinstance(artifacts, dict) and all(isinstance(v, str) and v in artifacts for v in evidence),
                     "failure": invalid or (None if completed else "missing_or_incomplete")})
        if isinstance(result, dict) and isinstance(result.get("usage"), dict):
            for k, v in result["usage"].items():
                if type(v) in (int, float) and math.isfinite(v) and v >= 0:
                    usage[k] += v
    regimes = {c: _summarize([r for r in rows if r["condition"] == c]) for c in CONDITIONS}
    matched = _summarize([r for r in rows if not r["model_mismatch"]])
    g = CONTRACT["gates"]
    def at_least(value, threshold): return value is not None and value >= threshold
    gates = {"matched_mae": matched["mae_nm"] is not None and matched["mae_nm"] <= g["matched_mae_nm_max"],
             "matched_bias": matched["bias_nm"] is not None and abs(matched["bias_nm"]) <= g["matched_abs_bias_nm_max"],
             "matched_coverage": at_least(matched["coverage"]["rate"], g["matched_coverage_min"]),
             "matched_wrong_decisive": matched["wrong_decisive"]["rate"] <= g["wrong_decisive_rate_max"],
             "contract_decisions": at_least(_summarize(rows)["contract_correct"]["rate"], g["contract_decision_accuracy_min"]),
             "structural_abstention": at_least(_rate([r for r in rows if r["ambiguity_reason"] == "structural"], "abstained")["rate"], g["structural_abstention_min"]),
             "clean_usefulness": at_least(regimes["clean"]["decisive"]["rate"], g["clean_decisive_rate_min"])}
    regime_gates = {c: {"mae": regimes[c]["mae_nm"] is not None and regimes[c]["mae_nm"] <= g["matched_mae_nm_max"],
                       "coverage": at_least(regimes[c]["coverage"]["rate"], g["matched_coverage_min"]),
                       "contract_decisions": at_least(regimes[c]["contract_correct"]["rate"], g["contract_decision_accuracy_min"])}
                    for c in ("clean", "bounded_drift", "reference")}
    return {"schema_version": "factor-evaluation-report/1", "split": public["split"], "contract_hash": public["contract_hash"],
            "overall": _summarize(rows), "matched_model": matched, "regimes": regimes,
            "provisional_gates": gates, "matched_regime_gates": regime_gates, "usage_totals_reported": dict(usage), "rows": rows,
            "limitations": ["Synthetic reconstructed displacement, not raw PDV or independent real data.", "Known-answer scoring is deterministic; generator shares reduced additive assumptions with estimators.", "Coverage conditions on numerical outputs; missing runs are failures, not abstentions.", "Evidence presence is not entailment. Usage is caller-reported, not independently metered.", "Provisional gates do not establish AI superiority or prospective usefulness. No human baseline has been measured."]}


def _exposure(evaluator_truth):
    truth = _read(evaluator_truth)
    if not isinstance(evaluator_truth, (str, Path)):
        return "development_exposed" if truth["split"] != "final" else "final_exposure_untracked_not_confirmatory"
    path = Path(evaluator_truth).parent / "exposure.json"
    prior = _read(path) if path.exists() else {"scoring_count": 0}
    count = prior.get("scoring_count", 0)
    _write(path, {"status": "exposed", "scoring_count": count + 1})
    return "development_exposed" if truth["split"] != "final" else ("one_use_final_now_exposed" if count == 0 else "exposed_final_not_confirmatory")


def score_agent_runs(public_cases, evaluator_truth, results):
    report = _score(public_cases, evaluator_truth, results)
    report["evaluation_use"] = _exposure(evaluator_truth)
    return report


def compare_methods(public_cases, evaluator_truth, methods):
    """One evaluation campaign; reports stay separate, with equal-information data."""
    if not isinstance(methods, dict) or not methods:
        raise ContractError("Supply named method result sets")
    reports = {name: _score(public_cases, evaluator_truth, results) for name, results in methods.items()}
    exposure = _exposure(evaluator_truth)
    for report in reports.values():
        report["evaluation_use"] = exposure
    return {"schema_version": "factor-method-comparison/1", "evaluation_use": exposure,
            "methods": reports, "claim": "Descriptive comparison; no human baseline or causal AI-advantage claim."}
