"""Evaluation functions for the synthetic development benchmark."""
import math
import numpy as np
from .constants import UNITS


def validate_predictions(truth, predictions):
    truth_ids = [r["case_id"] for r in truth]
    pred_ids = [r["case_id"] for r in predictions]
    if len(set(truth_ids)) != len(truth_ids) or len(set(pred_ids)) != len(pred_ids):
        raise ValueError("Duplicate case IDs are not allowed")
    if set(truth_ids) != set(pred_ids):
        raise ValueError("Prediction case IDs must exactly match truth; use explicit null to abstain")
    for row in predictions:
        if not isinstance(row.get("structure_detected"), bool):
            raise ValueError("structure_detected must be a boolean")
        for key, unit in UNITS.items():
            if key not in row:
                raise ValueError(f"Missing {key}; use null to abstain")
            v = row[key]
            if v is None:
                continue
            if v.get("unit") != unit:
                raise ValueError(f"Wrong unit for {key}; expected {unit}")
            vals = [v.get(n) for n in ("mean", "lower90", "upper90")]
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vals):
                raise ValueError(f"Non-finite or non-numeric value in {key}")
            if not vals[1] <= vals[0] <= vals[2]:
                raise ValueError(f"Invalid interval ordering in {key}")
            if key == "mode_amplitude_um" and min(vals) < 0:
                raise ValueError("Magnitude and interval bounds must be nonnegative")


def wilson(success, count):
    if not count:
        return None
    z = 1.959963984540054
    p = success / count
    den = 1 + z*z/count
    center = (p + z*z/(2*count))/den
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count))/den
    return [center-half, center+half]


def score(truth, predictions):
    validate_predictions(truth, predictions)
    by_id = {r["case_id"]: r for r in predictions}
    output = {}
    for group in ["all"] + sorted(set(r["scenario"] for r in truth)):
        rows = [r for r in truth if group == "all" or r["scenario"] == group]
        metrics = {"n_cases": len(rows)}
        for key, unit in UNITS.items():
            pairs = [(r[key], by_id[r["case_id"]][key]) for r in rows if by_id[r["case_id"]][key] is not None]
            m = {"unit": unit, "n_answered": len(pairs), "answer_rate": len(pairs)/len(rows)}
            if pairs:
                yy = np.array([p[0] for p in pairs])
                mean, low, high = [np.array([p[1][v] for p in pairs]) for v in ("mean", "lower90", "upper90")]
                coverage = (low <= yy) & (yy <= high)
                interval_score = high-low + 20*np.maximum(low-yy, 0) + 20*np.maximum(yy-high, 0)
                m.update(mae=float(np.mean(np.abs(mean-yy))), rmse=float(np.sqrt(np.mean((mean-yy)**2))),
                         coverage90=float(coverage.mean()), coverage95_wilson=wilson(int(coverage.sum()), len(pairs)),
                         mean_width90=float(np.mean(high-low)), mean_interval_score90=float(interval_score.mean()),
                         conservative_bound_count=sum(p[1].get("interval_kind") == "conservative_90_percent_bound" for p in pairs))
            metrics[key] = m
        nulls = [r for r in rows if not r["has_structure"]]
        positives = [r for r in rows if r["has_structure"]]
        fp = sum(by_id[r["case_id"]]["structure_detected"] for r in nulls)
        tp = sum(by_id[r["case_id"]]["structure_detected"] for r in positives)
        metrics["detection"] = {"n_null": len(nulls), "n_positive": len(positives),
                                "false_positives": fp, "true_positives": tp,
                                "false_positive_rate": fp/len(nulls) if nulls else None,
                                "false_positive_rate95_wilson": wilson(fp, len(nulls)),
                                "sensitivity": tp/len(positives) if positives else None}
        output[group] = metrics
    return output
