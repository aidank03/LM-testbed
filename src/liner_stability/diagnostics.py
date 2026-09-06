"""Diagnostics functions for the synthetic development benchmark."""
import math
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .constants import UNITS, Z90


def equivalent_width(radius_um, absorption_per_um=0.00189):
    """Independent quadrature of 1-exp(-optical depth) across a uniform cylinder.

    Integrate using x=r sin(theta). Generator uses Cartesian ray lengths.
    Both assume the same monochromatic, uniform material; this is not
    cross-physics validation. Width and radius are in micrometres.
    """
    nodes, weights = np.polynomial.legendre.leggauss(96)
    theta = nodes * np.pi / 2
    ct = np.cos(theta)
    r = np.asarray(radius_um)[..., None]
    return r[..., 0] * np.sum((1 - np.exp(-2 * absorption_per_um * r * ct))
                              * ct * weights * np.pi / 2, axis=-1)


def _pdv_inputs(t, re, im, wavelength_nm, search_window_ns):
    t, re, im = (np.asarray(v, dtype=float) for v in (t, re, im))
    if (t.ndim != 1 or re.shape != t.shape or im.shape != t.shape or len(t) < 3
            or not all(np.all(np.isfinite(v)) for v in (t, re, im))):
        raise ValueError("PDV time and quadratures must be equal finite one-dimensional arrays")
    delta = np.diff(t)
    step = float(np.median(delta))
    if step <= 0 or np.any(delta <= 0) or not np.allclose(delta, step, rtol=1e-6, atol=1e-9):
        raise ValueError("This PDV reducer requires a strictly increasing uniform time grid")
    window = np.asarray(search_window_ns, dtype=float)
    if (window.shape != (2,) or not np.all(np.isfinite(window)) or window[0] >= window[1]
            or window[0] < t[0] or window[1] > t[-1]):
        raise ValueError("PDV search window must be ordered, finite and contained in the recorded times")
    if (isinstance(wavelength_nm, bool) or not isinstance(wavelength_nm, (int, float, np.integer, np.floating))
            or not np.isfinite(wavelength_nm) or wavelength_nm <= 0):
        raise ValueError("PDV wavelength must be finite and positive")
    selected = (t >= window[0]) & (t <= window[1])
    if selected.sum() < 3:
        raise ValueError("Insufficient samples in the PDV search window")
    return t, re, im, step, window


def pdv_point(t, re, im, wavelength_nm=1550, search_window_ns=(30.0, 60.0)):
    """Fit a kinematic minimum inside the declared window on a uniform ns grid."""
    t, re, im, step, search = _pdv_inputs(t, re, im, wavelength_nm, search_window_ns)
    phase = np.unwrap(np.arctan2(im, re))
    with np.errstate(over="ignore", invalid="ignore"):
        x = phase * wavelength_nm / (4 * np.pi)
    if not np.all(np.isfinite(x)):
        raise ValueError("PDV displacement exceeds the numerical range")
    baseline = t < 8
    if baseline.sum() < 8:
        raise ValueError("Insufficient baseline")
    x -= np.mean(x[baseline])
    smooth = gaussian_filter1d(x, 1.0 / step)
    window = np.flatnonzero((t >= search[0]) & (t <= search[1]))
    j = window[np.argmin(smooth[window])]
    local = np.abs(t - t[j]) <= 3
    if local.sum() < 5 or not np.any(t[local] < t[j]) or not np.any(t[local] > t[j]):
        raise ValueError("Insufficient samples around the local PDV minimum")
    try:
        coeff = np.polyfit(t[local] - t[j], smooth[local], 2)
    except np.linalg.LinAlgError:
        raise ValueError("PDV local quadratic fit failed") from None
    aa, bb, cc = coeff
    if not np.all(np.isfinite(coeff)) or aa <= 0 or abs(bb / (2 * aa)) > 4:
        raise ValueError("No reliable local minimum")
    turnaround = float(t[j] - bb / (2 * aa))
    if not search[0] <= turnaround <= search[1]:
        raise ValueError("Fitted PDV minimum is outside the declared search window")
    return float(-cc + bb**2 / (4 * aa)), turnaround, x, smooth


def interval(value, sigma, unit, nonnegative=False):
    lower = value - Z90 * sigma
    if nonnegative:
        lower = max(0.0, lower)
    return {"mean": float(value), "lower90": float(lower),
            "upper90": float(value + Z90 * sigma), "unit": unit}


def pdv_estimates(obs, bootstrap=48):
    t, re, im = obs["time_ns"], obs["iq_real"], obs["iq_imag"]
    cal = obs["calibration"]
    search = cal.get("pdv_search_window_ns", (30.0, 60.0))
    t, re, im, step, search = _pdv_inputs(t, re, im, cal["wavelength_nm"], search)
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 2:
        raise ValueError("PDV noise propagation requires at least two draws")
    invalid = (np.hypot(re, im) < 0.3) & (t >= search[0]) & (t <= search[1])
    # Dropout detection does not use truth. A long gap within the target window
    # makes the turnaround non-identifiable to this reduction.
    runs = np.diff(np.flatnonzero(np.concatenate(([True], ~invalid, [True])))) - 1
    dropout = (runs.max(initial=0) * step) > 3
    try:
        depth, turn, _, _ = pdv_point(t, re, im, cal["wavelength_nm"], search)
    except ValueError:
        return {}, {}, ["pdv_minimum_unresolved"]
    # Numerical noise propagation, NOT posterior sampling and NOT SBC.
    # The original observed signal is perturbed with calibrated IQ noise.
    seed = int(hashlib.sha256(obs["case_id"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        try:
            d, tt, _, _ = pdv_point(t, re + rng.normal(0, cal["iq_noise_std"], len(t)),
                                   im + rng.normal(0, cal["iq_noise_std"], len(t)), cal["wavelength_nm"], search)
            draws.append((d, tt))
        except ValueError:
            pass
    if len(draws) < max(2, bootstrap / 2):
        return {}, {}, ["pdv_noise_propagation_failed"]
    sd, st = np.std(draws, axis=0, ddof=1)
    naive = {"compression_nm": interval(depth, sd, "nm"),
             "turnaround_ns": interval(turn, st, "ns")}
    if dropout:
        return naive, {}, ["pdv_return_gap_abstain"]
    calibrated = {
        "compression_nm": interval(depth, math.hypot(sd, depth * cal["scale_relative_sigma"]), "nm"),
        "turnaround_ns": interval(turn, math.hypot(st, cal["trigger_sigma_ns"]), "ns"),
    }
    return naive, calibrated, []


def radiograph_estimates(obs):
    cal, counts = obs["calibration"], obs["counts"]
    dx = np.median(np.diff(obs["x_um"]))
    photons = cal["photons_per_pixel"]
    w = np.sum(1 - counts / photons, axis=1) * dx
    angle = 2 * np.pi * cal["mode"] * obs["z_um"] / cal["z_period_um"]
    design = np.column_stack((np.ones(len(w)), np.cos(angle), np.sin(angle)))
    beta = np.linalg.lstsq(design, w, rcond=None)[0]
    variance_w = np.sum(counts, axis=1) * dx**2 / photons**2
    inv = np.linalg.inv(design.T @ design)
    covariance = inv @ (design.T @ (variance_w[:, None] * design)) @ inv
    # Intercept constrains the radius. Seed amplitude is small relative to R.
    radius_grid = np.linspace(330, 460, 261)
    rad = float(np.interp(beta[0], equivalent_width(radius_grid, cal["absorption_per_um"]), radius_grid))
    slope = float((equivalent_width(rad + 0.02, cal["absorption_per_um"])
                   - equivalent_width(rad - 0.02, cal["absorption_per_um"])) / 0.04)
    amp_w = float(np.hypot(beta[1], beta[2]))
    mode_cov = covariance[1:, 1:]
    chi2 = float(beta[1:] @ np.linalg.solve(mode_cov, beta[1:]))
    # Known-mode 1% false alarm threshold for chi-square with 2 degrees of freedom.
    # p-value is NOT a posterior probability that structure is absent/present.
    detected = bool(chi2 > -2 * np.log(0.01))
    direction = beta[1:] / max(amp_w, 1e-12)
    sd_w = float(np.sqrt(direction @ mode_cov @ direction))
    amplitude = amp_w / slope
    sd_a = sd_w / slope
    naive = interval(amplitude, sd_a, "um", nonnegative=True)
    kpix = 2 * np.pi * cal["mode"] / len(w)
    sigma = cal["sigma_z_pixels"]
    mtf = float(np.exp(-0.5 * (kpix * sigma)**2))
    if mtf < 0.2:
        return naive, None, detected, ["radiograph_mtf_below_0.2_abstain"]
    corrected = amplitude / mtf
    sd = math.hypot(sd_a / mtf, corrected * kpix**2 * sigma * cal["sigma_z_uncertainty_pixels"])
    # At low significance the positive magnitude has a non-Gaussian (Rice)
    # distribution. Use an explicitly conservative upper bound from the 2D
    # Gaussian coefficient radius, NOT a claimed equal-tailed 90% interval.
    if not detected:
        upper = corrected + math.sqrt(-2 * math.log(0.10)) * math.sqrt(np.linalg.eigvalsh(mode_cov).max()) / slope / mtf
        calibrated = {"mean": corrected, "lower90": 0.0, "upper90": upper,
                      "unit": "um", "interval_kind": "conservative_90_percent_bound"}
    else:
        calibrated = interval(corrected, sd, "um", nonnegative=True)
    return naive, calibrated, detected, []


def analyze(obs, bootstrap=48):
    """Reduce observations without a truth argument or explicit scenario label.

    Legacy public case IDs encode generator seeds; they are not a hidden test.
    Opaque IDs and an independent method RNG belong in the new evaluator.
    """
    naive_pdv, cal_pdv, flags = pdv_estimates(obs, bootstrap)
    naive_rad, cal_rad, detected, rflags = radiograph_estimates(obs)
    naive = {"case_id": obs["case_id"], **{key: None for key in UNITS}, **naive_pdv,
             "mode_amplitude_um": naive_rad, "structure_detected": detected,
             "flags": [], "method": "simple_reduction"}
    calibrated = {"case_id": obs["case_id"], **{key: None for key in UNITS}, **cal_pdv,
                  "mode_amplitude_um": cal_rad, "structure_detected": detected,
                  "flags": flags + rflags, "method": "calibration_aware_reduction"}
    return naive, calibrated
