"""Rank proposed measurements by a transparent two-model separation heuristic."""
import numpy as np
from .physics import positive


def rank_designs(config):
    observables = config.get("observables")
    candidates = config.get("candidates")
    if not isinstance(observables, list) or not observables or not isinstance(candidates, list) or not candidates:
        raise ValueError("Nonempty observables and candidates are required")
    if not all(isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(x.get("unit"), str)
               and x["name"] and x["unit"] for x in observables):
        raise ValueError("Each observable needs a name and unit")
    rows, ids = [], set()
    n = len(observables)
    for item in candidates:
        cid = item.get("id")
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError("Candidate IDs must be nonempty and unique")
        ids.add(cid)
        a, b = np.asarray(item["mean_model_a"], dtype=float), np.asarray(item["mean_model_b"], dtype=float)
        cov = np.asarray(item["covariance"], dtype=float)
        if a.shape != (n,) or b.shape != (n,) or cov.shape != (n,n):
            raise ValueError("Prediction/covariance dimensions must match the observables")
        if not all(np.all(np.isfinite(x)) for x in (a,b,cov)) or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
            raise ValueError("Finite predictions and a symmetric covariance are required")
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance must be positive definite") from None
        probability = positive("usable_probability", item["usable_probability"], allow_zero=True)
        if probability > 1:
            raise ValueError("usable_probability must not exceed one")
        cost = positive("relative_cost", item["relative_cost"])
        whitened = np.linalg.solve(chol, b-a)
        d2 = float(whitened @ whitened)
        rows.append({"candidate_id": cid, "squared_standardized_separation": d2,
                     "usable_probability": probability, "relative_cost": cost,
                     "screening_utility": probability*d2/cost})
    rows.sort(key=lambda x: (-x["screening_utility"], x["candidate_id"]))
    return {"status": config.get("status", "user_supplied_predictions"), "ranking": rows,
            "criterion": "usable_probability * (delta.T covariance^-1 delta) / relative_cost",
            "limitations": ["a two-model screening heuristic, not Bayesian expected information gain",
                            "depends on supplied predictions, covariance, usability and cost",
                            "does not establish facility feasibility or a causal mechanism"], "observables": observables}
