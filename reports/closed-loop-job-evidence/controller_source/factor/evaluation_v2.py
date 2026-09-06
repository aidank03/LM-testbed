"""Explicit public task semantics, reusing the frozen v1 generator and grader.

V1 remains unchanged. Normalization below is evaluator-internal reuse, after
validating v2 provenance; candidate methods receive the complete v2 question.
"""
import copy
import tempfile
from pathlib import Path
from . import evaluation as v1
from .contracts import ContractError

QUESTION = """Assess signed radial material displacement x over the observation's fixed time window. Units are meters. NEGATIVE displacement is inward (compression); positive is outward. This is a conditional reconstructed-measurement task, not a melting or instability-mechanism inference.

Apply this public decision contract, equally available to all methods:
1. The signal is y = x + optical drift + zero-mean Gaussian noise. noise_sigma_m is its standard deviation. drift_bound_m is an absolute non-motion displacement bound; null means no finite bound, not zero. Supplied calibration/noise bounds are assumptions to use for the conditional answer, not facts to infer from source_kind. Synthetic data may contain motion.
2. Without a reference, center = apparent_displacement_m, sigma = noise_sigma_m, B = drift_bound_m.
3. With a reference, center = apparent_displacement_m - reference.displacement_m; sigma = sqrt(noise_sigma_m^2 + reference.sigma_m^2 - 2*reference.noise_covariance_m2); B = reference.residual_bound_m. The reference measures optical drift, not material motion. Its supplied residual bound replaces the signal-only drift bound. Null residual bound means no finite reference-transfer bound.
4. If the applicable B is null, different material motions can explain the same measurement: choose ambiguous. Otherwise form the 95% Gaussian confidence set enlarged by the absolute nuisance bound: h = 1.959963984540054*sigma + B; lower = center-h; upper = center+h.
5. Let T = -resolution_m. Choose compression_supported only if upper <= T. Choose compression_not_supported only if lower > T: this means the whole conditional interval rules out inward motion beyond the resolution, not merely that evidence for compression is missing. Choose ambiguous otherwise. An interval that crosses T requires ambiguity even if its center is negative. These valid benchmark questions are in scope; out_of_scope is reserved for requests outside this task.
6. Undeclared drift or reference mismatch may be undetectable from the allowed observation. Report the conditional result with its assumptions; do not guess hidden simulator truth or claim a declared zero bound proves no omitted physics.

Return the registered final decision, supporting available evidence IDs, brief explanation, and next step. You may compute directly or use any allowed tools; no particular tool sequence is required. If reporting a numerical result, state its interval and units. Execution failure is not scientific ambiguity."""

CONTRACT = copy.deepcopy(v1.CONTRACT)
CONTRACT.update(schema_version="factor-evaluation-contract/2", public_question=QUESTION,
                change_reason="V1 omitted public sign, confidence-level, reference and label semantics; v2 supplies them before candidate execution.")
CONTRACT["gates"].update(completion_min=1.0, evidence_presence_min=1.0)
CONTRACT_HASH = v1._hash(CONTRACT)


def create_suite(out, seed=20260906, n_per_condition=2, split="development"):
    """Freeze v2 before generating new observations; never rewrite a v1 suite."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ContractError("V2 output directory must be new or empty")
    if split not in ("train", "development", "final") or type(n_per_condition) is not int or n_per_condition < 1:
        raise ContractError("Use train/development/final and a positive integer count")
    out.mkdir(parents=True, exist_ok=True)
    paths = {key: out / filename for key, filename in (("contract", "contract.json"), ("public_cases", "public_cases.json"), ("evaluator_truth", "evaluator_truth.json"), ("exposure", "exposure.json"))}
    v1._write(paths["contract"], {"contract": CONTRACT, "contract_hash": CONTRACT_HASH})
    with tempfile.TemporaryDirectory(prefix="factor-v2-generator-") as temporary:
        original = v1.create_suite(Path(temporary) / "source", seed, n_per_condition, split)
        public, truth = v1._read(original["public_cases"]), v1._read(original["evaluator_truth"])
    for case in public["cases"]:
        case["question"] = QUESTION
    public.update(schema_version="factor-cases/2", contract_hash=CONTRACT_HASH)
    truth.update(schema_version="factor-truth/2", contract_hash=CONTRACT_HASH,
                 public_hash=v1._hash(public["cases"]), generator_version="factor-generator/1+explicit-task/2")
    v1._write(paths["public_cases"], public)
    v1._write(paths["evaluator_truth"], truth)
    v1._write(paths["exposure"], {"status": "unexposed", "scoring_count": 0})
    return paths


def _normalized(public):
    result = copy.deepcopy(public)
    result.update(schema_version="factor-cases/1", contract_hash=v1._hash(v1.CONTRACT))
    return result


def _validate_public(value):
    public = v1._read(value)
    if not isinstance(public, dict) or public.get("schema_version") != "factor-cases/2" or public.get("contract_hash") != CONTRACT_HASH:
        raise ContractError("Expected the explicit v2 public contract")
    v1._validate_public(_normalized(public))
    if any(c["question"] != QUESTION for c in public["cases"]):
        raise ContractError("Every v2 case must carry the complete frozen public rules")
    if isinstance(value, (str, Path)):
        manifest = v1._read(Path(value).parent / "contract.json")
        if manifest.get("contract_hash") != CONTRACT_HASH or v1._hash(manifest.get("contract")) != CONTRACT_HASH:
            raise ContractError("V2 frozen manifest failed verification")
    return public


def baseline_results(public_cases, method="bounded"):
    return v1.baseline_results(_normalized(_validate_public(public_cases)), method)


def _score(public_cases, evaluator_truth, results):
    public, truth = _validate_public(public_cases), copy.deepcopy(v1._read(evaluator_truth))
    if not isinstance(truth, dict) or truth.get("schema_version") != "factor-truth/2" or truth.get("contract_hash") != CONTRACT_HASH or truth.get("public_hash") != v1._hash(public["cases"]):
        raise ContractError("V2 evaluator truth does not match the public contract/data")
    normalized = _normalized(public)
    truth.update(schema_version="factor-truth/1", contract_hash=normalized["contract_hash"], public_hash=v1._hash(normalized["cases"]))
    report = v1._score(normalized, truth, results)
    report.update(schema_version="factor-evaluation-report/2", contract_hash=CONTRACT_HASH,
                  contract_version=CONTRACT["schema_version"])
    report["provisional_gates"].update(
        complete_runs=report["overall"]["completed"]["rate"] == 1.0,
        evidence_presence=report["overall"]["evidence_present"]["rate"] == 1.0)
    report["limitations"].append("V2 tests an explicitly specified interface; comparison with v1 is confounded by changed instructions and a fresh development sample.")
    return report


def score_agent_runs(public_cases, evaluator_truth, results):
    report = _score(public_cases, evaluator_truth, results)
    report["evaluation_use"] = v1._exposure(evaluator_truth)
    return report


def compare_methods(public_cases, evaluator_truth, methods):
    if not isinstance(methods, dict) or not methods:
        raise ContractError("Supply named method result sets")
    reports = {name: _score(public_cases, evaluator_truth, results) for name, results in methods.items()}
    exposure = v1._exposure(evaluator_truth)
    for report in reports.values():
        report["evaluation_use"] = exposure
    return {"schema_version": "factor-method-comparison/2", "evaluation_use": exposure,
            "methods": reports, "claim": "Descriptive explicitly specified task comparison; no human baseline or causal AI-advantage claim."}
