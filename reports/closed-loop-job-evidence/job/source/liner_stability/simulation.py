"""Simulation functions for the synthetic development benchmark."""
import numpy as np
from scipy.ndimage import gaussian_filter
from .constants import SCENARIOS


def simulate(seed, scenario):
    """Return public observation and separate truth. Analysis receives only obs."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    t = np.arange(0, 100.01, 0.25)
    depth = float(rng.uniform(28, 45))
    turn = float(rng.uniform(40, 50))
    a = float(rng.uniform(0.12, 0.20))
    q = np.clip((t - 10) / (turn - 10), 0, 1)
    displacement = -depth * np.sin(np.pi * q / 2) ** 2
    displacement += 0.5 * a * np.maximum(t - turn, 0) ** 2

    trigger_sigma = 2.5 if scenario == "timing_jitter" else 0.15
    trigger_offset = rng.normal(0, trigger_sigma)
    scale_sigma = 0.003
    wavelength_nm = 1550 * (1 + rng.normal(0, scale_sigma))
    optical_drift = 0.10 * np.maximum(t - 10, 0) if scenario == "omitted_optical_drift" else 0
    phase = 4 * np.pi * (displacement + optical_drift) / wavelength_nm
    envelope = np.ones_like(t)
    if scenario == "pdv_dropout":
        envelope[np.abs(t - turn) < 8] = 0
    noise = 0.01
    signal = envelope * np.exp(1j * phase)
    signal += noise * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))

    z = np.arange(64) * 1200.0 / 64
    x = np.linspace(-560, 560, 448)
    mode = 3
    amplitude = 0.0 if scenario == "no_structure" else float(rng.uniform(4, 12))
    spatial_phase = float(rng.uniform(-np.pi, np.pi))
    radius = 400 + float(np.interp(60, t, displacement)) / 1000
    rz = radius + amplitude * np.cos(2 * np.pi * mode * z / 1200 + spatial_phase)
    path_um = 2 * np.sqrt(np.maximum(rz[:, None] ** 2 - x[None, :] ** 2, 0))
    absorption = 0.7 * 2700 * 1e-6  # illustrative kappa [m2/kg] * density [kg/m3]
    transmission = np.exp(-absorption * path_um)
    sigma_z = 5.5 if scenario == "blurred" else 1.0
    sigma_unc = 0.15 if scenario == "blurred" else 0.07
    true_sigma_z = max(0.05, sigma_z + rng.normal(0, sigma_unc))
    blurred = gaussian_filter(transmission, sigma=(true_sigma_z, 1.2), mode=("wrap", "nearest"))
    photons = 5000
    counts = rng.poisson(photons * blurred)
    obs = {
        "case_id": f"demo_{seed}", "time_ns": t + trigger_offset,
        "iq_real": signal.real, "iq_imag": signal.imag,
        "x_um": x, "z_um": z, "counts": counts,
        "calibration": {
            "wavelength_nm": 1550.0, "iq_noise_std": noise,
            "trigger_sigma_ns": trigger_sigma, "scale_relative_sigma": scale_sigma,
            "photons_per_pixel": photons, "absorption_per_um": absorption,
            "sigma_z_pixels": sigma_z, "sigma_z_uncertainty_pixels": sigma_unc,
            "mode": mode, "z_period_um": 1200.0,
            "pdv_search_window_ns": [30.0, 60.0],
        },
    }
    truth = {"case_id": obs["case_id"], "scenario": scenario,
             "compression_nm": depth, "turnaround_ns": turn,
             "mode_amplitude_um": amplitude, "has_structure": bool(amplitude > 0)}
    return obs, truth
