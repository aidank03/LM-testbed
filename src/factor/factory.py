"""Deterministic local candidate-release gates; never a scientific validator."""
import hashlib
import json
import math
from pathlib import Path
import re


def artifact_hash(value):
    """Canonical JSON hash used to bind a contract before evaluation."""
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric(report, path):
    value = report["metrics"]
    for part in path.split("."):
        value = value[part]
    if not _number(value):
        raise ValueError("Metric must be a finite number")
    return value


def assess_candidate(reference, candidate, contract):
    """Assess two completed, task-matched evaluation envelopes.

    A valid contract has schema_version=1, task_id, metric_scope,
    minimum_independent_units, required_evidence_gates, and metrics. Each metric
    maps a dotted path to min/max/min_delta/max_delta constraints; delta is
    candidate minus reference. No threshold is guessed and no scores are mixed.
    Hash binding establishes consistency, not proof of when a contract existed.
    Trusted evaluation runners must freeze contracts and retain their provenance.
    """
    required = {"schema_version", "task_id", "metric_scope", "minimum_independent_units",
                "required_evidence_gates", "metrics"}
    if not required <= contract.keys() or contract["schema_version"] != 1:
        raise ValueError("Incomplete or unsupported promotion contract")
    minimum = contract["minimum_independent_units"]
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("minimum_independent_units must be positive")
    if not contract["task_id"] or not contract["metric_scope"] or not contract["metrics"]:
        raise ValueError("Task, metric scope and explicit metrics are required")
    if not isinstance(contract["required_evidence_gates"], list) or not contract["required_evidence_gates"]:
        raise ValueError("At least one explicit evidence gate is required")
    for path, limits in contract["metrics"].items():
        if not path or not limits or set(limits)-{"min", "max", "min_delta", "max_delta"}:
            raise ValueError("Unsupported metric constraint")
        if not all(_number(value) for value in limits.values()):
            raise ValueError("Metric constraints must be finite numbers")
    digest, reasons, measurements = artifact_hash(contract), [], {}
    for name, report in (("reference", reference), ("candidate", candidate)):
        if report.get("completed") is not True:
            reasons.append(f"{name}: evaluation not completed")
        if report.get("contract_sha256") != digest:
            reasons.append(f"{name}: missing or different predeclared contract binding")
        for field in ("task_id", "metric_scope"):
            if report.get(field) != contract[field]:
                reasons.append(f"{name}: {field} mismatch")
        count = report.get("independent_units")
        if not isinstance(count, int) or isinstance(count, bool) or count < minimum:
            reasons.append(f"{name}: insufficient independent evaluation units")
        if not _hash(report.get("test_set_sha256")):
            reasons.append(f"{name}: missing valid test-set hash")
        gates = report.get("evidence_gates", {})
        for gate in contract["required_evidence_gates"]:
            evidence = gates.get(gate, {})
            if (evidence.get("passed") is not True or not _hash(evidence.get("artifact_sha256"))
                or evidence.get("verification") not in {"deterministic_evaluator", "independent_human_review"}):
                reasons.append(f"{name}: evidence gate incomplete or untrusted: {gate}")
    if reference.get("test_set_sha256") != candidate.get("test_set_sha256"):
        reasons.append("Reference and candidate evaluated different test sets")
    for path, limits in contract["metrics"].items():
        try:
            old, new = _metric(reference, path), _metric(candidate, path)
        except (KeyError, TypeError, ValueError):
            reasons.append(f"Metric unavailable or invalid: {path}")
            continue
        delta = new-old
        measurements[path] = {"reference": old, "candidate": new, "delta": delta}
        checks = {"min": new, "max": new, "min_delta": delta, "max_delta": delta}
        for condition, limit in limits.items():
            failed = checks[condition] < limit if condition in {"min", "min_delta"} else checks[condition] > limit
            if failed:
                reasons.append(f"Metric gate failed: {path} {condition} {limit}")
    return {"schema_version": 1, "status": "blocked" if reasons else "eligible_for_local_release",
            "candidate_id": candidate.get("candidate_id"), "task_id": contract["task_id"],
            "metric_scope": contract["metric_scope"], "reasons": reasons, "measurements": measurements,
            "contract_sha256": digest, "reference_report_sha256": artifact_hash(reference),
            "candidate_report_sha256": artifact_hash(candidate),
            "scientific_validation_established": False,
            "claim": "Local contract assessment only; hashes do not certify evidence quality or scientific validity"}


def write_decision(reference, candidate, contract, path):
    """Write a new local acceptance/rejection record; preserve every source report.

    This explicit call does not start a model, alter defaults, deploy externally,
    or discard a failed candidate. Both accepted and blocked decisions are saved.
    """
    result = assess_candidate(reference, candidate, contract)
    body = json.dumps(result, indent=2, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as output:
        output.write(body)
    return result
