"""Observation-only fixed-window motion inference, distinct from legacy depth.

The noise confidence set is enlarged by a deterministic nuisance bound. This is
not a raw-PDV reducer or a posterior, and the supplied bound cannot be verified
from a single confounded observation. No evaluator or generator is imported.
"""
from math import sqrt
from statistics import NormalDist
from .contracts import ContractError, finite

OBSERVATION_FIELDS = {"schema_version", "observation_id", "apparent_displacement_m",
                      "noise_sigma_m", "drift_bound_m", "reference", "window_s",
                      "resolution_m", "source_kind", "calibration_id"}
REFERENCE_FIELDS = {"displacement_m", "sigma_m", "residual_bound_m", "noise_covariance_m2"}
DECISIONS = ("compression_supported", "compression_not_supported", "ambiguous", "out_of_scope")


def validate_observation(obs):
    if not isinstance(obs, dict) or set(obs) != OBSERVATION_FIELDS:
        raise ContractError("Motion observation has missing or unknown fields")
    if obs["schema_version"] != "motion-observation/1":
        raise ContractError("Unsupported motion schema")
    for field in ("observation_id", "calibration_id"):
        if not isinstance(obs[field], str) or not 1 <= len(obs[field]) <= 120:
            raise ContractError(f"Invalid {field}")
    if obs["source_kind"] not in ("synthetic_reconstructed", "real_reconstructed"):
        raise ContractError("Specify reconstructed synthetic or real source")
    finite(obs["apparent_displacement_m"], "apparent_displacement_m", minimum=-1, maximum=1)
    finite(obs["noise_sigma_m"], "noise_sigma_m", minimum=0, maximum=1)
    finite(obs["resolution_m"], "resolution_m", minimum=1e-15, maximum=1)
    if obs["drift_bound_m"] is not None:
        finite(obs["drift_bound_m"], "drift_bound_m", minimum=0, maximum=1)
    window = obs["window_s"]
    if not isinstance(window, list) or len(window) != 2:
        raise ContractError("window_s needs two increasing finite times")
    for t in window:
        finite(t, "window_s", minimum=-1e9, maximum=1e9)
    if window[1] <= window[0]:
        raise ContractError("window_s must increase")
    ref = obs["reference"]
    if ref is not None:
        if not isinstance(ref, dict) or set(ref) != REFERENCE_FIELDS:
            raise ContractError("Invalid reference observation fields")
        finite(ref["displacement_m"], "reference displacement", minimum=-1, maximum=1)
        finite(ref["sigma_m"], "reference sigma", minimum=0, maximum=1)
        finite(ref["noise_covariance_m2"], "reference covariance")
        if abs(ref["noise_covariance_m2"]) > obs["noise_sigma_m"] * ref["sigma_m"] + 1e-30:
            raise ContractError("Reference noise covariance is impossible")
        if ref["residual_bound_m"] is not None:
            finite(ref["residual_bound_m"], "reference residual", minimum=0, maximum=1)
    return obs


def infer_motion(obs, method="bounded"):
    validate_observation(obs)
    if method not in ("naive", "bounded"):
        raise ContractError("Unknown motion method")
    center, sigma, bound = obs["apparent_displacement_m"], obs["noise_sigma_m"], obs["drift_bound_m"]
    track = "signal_only" if obs["reference"] is None else "reference_available"
    assumptions = ["Known unit gain and fixed-window reconstruction.",
                   "Known zero-mean Gaussian measurement noise."]
    if method == "naive":
        bound = 0.0
        assumptions.append("Optical drift assumed exactly zero; available nuisance metadata/reference ignored.")
    elif obs["reference"] is not None:
        ref = obs["reference"]
        center -= ref["displacement_m"]
        variance = sigma**2 + ref["sigma_m"]**2 - 2 * ref["noise_covariance_m2"]
        sigma = sqrt(max(0.0, variance))
        bound = ref["residual_bound_m"]
        assumptions.append("Reference transfer and joint Gaussian covariance valid within the supplied residual bound.")
    else:
        assumptions.append("Supplied drift bound covers all equivalent non-motion displacement.")
    result = {"schema_version": "motion-result/1", "task": "motion.fixed_window/1",
              "method": method, "information_track": track, "unit": "m",
              "window_s": obs["window_s"], "resolution_m": obs["resolution_m"],
              "point_m": None, "interval_m": None, "nominal_noise_coverage": 0.95,
              "interval_kind": "gaussian_confidence_set_widened_by_nuisance_bound",
              "inference": "ambiguous", "decision": "ambiguous", "assumptions": assumptions,
              "limitations": ["Not validated on independent real data.",
                              "Unseen violations of nuisance/calibration assumptions may be undetectable.",
                              "No phase, instability mechanism, or causal inference."],
              "next_measurement": "A calibrated drift witness with a defensible residual mismatch bound."}
    if bound is None:
        result["reason"] = "Unbounded nuisance: different material motions yield identical allowed observations."
        return result
    half = bound + NormalDist().inv_cdf(0.975) * sigma
    lower, upper = center - half, center + half
    result.update(point_m=center, interval_m=[lower, upper], inference="estimated_conditional")
    threshold = -obs["resolution_m"]
    if upper <= threshold:
        result["decision"] = "compression_supported"
    elif lower > threshold:
        result["decision"] = "compression_not_supported"
    result["reason"] = "Decision uses the entire conditional interval against the stated resolution."
    if result["decision"] != "ambiguous":
        result["next_measurement"] = "Independently validate the stated calibration and nuisance assumptions."
    return result

