"""Window-contract regression checks; no new diagnostic validation claim."""
import unittest
from unittest.mock import patch
import numpy as np

from liner_stability import diagnostics, simulation


def two_minima():
    t = np.arange(0, 100.01, 0.25)
    displacement = -30*np.exp(-((t-45)/5)**2) - 20*np.exp(-((t-75)/5)**2)
    phase = 4*np.pi*displacement/1550
    return {"case_id": "two-minima-development", "time_ns": t,
            "iq_real": np.cos(phase), "iq_imag": np.sin(phase),
            "calibration": {"wavelength_nm": 1550, "iq_noise_std": 0,
                            "scale_relative_sigma": 0, "trigger_sigma_ns": 0,
                            "pdv_search_window_ns": [65, 85]}}


class PDVWindowTests(unittest.TestCase):
    def test_legacy_default_is_preserved(self):
        obs, _ = simulation.simulate(1701, "nominal")
        args = obs["time_ns"], obs["iq_real"], obs["iq_imag"]
        self.assertEqual(diagnostics.pdv_point(*args)[:2],
                         diagnostics.pdv_point(*args, search_window_ns=(30, 60))[:2])

    def test_declared_window_selects_other_minimum_in_every_noise_draw(self):
        obs = two_minima()
        with patch.object(diagnostics, "pdv_point", wraps=diagnostics.pdv_point) as reducer:
            _, calibrated, flags = diagnostics.pdv_estimates(obs, bootstrap=8)
        self.assertEqual(flags, [])
        self.assertAlmostEqual(calibrated["turnaround_ns"]["mean"], 75, places=5)
        self.assertEqual(reducer.call_count, 9)
        self.assertTrue(all(np.array_equal(call.args[4], [65, 85]) for call in reducer.call_args_list))
        default = diagnostics.pdv_point(obs["time_ns"], obs["iq_real"], obs["iq_imag"])
        self.assertAlmostEqual(default[1], 45, places=5)

    def test_dropout_checks_the_declared_window(self):
        obs = two_minima()
        # A weak return preserves phase but has a six-ns gap within the new window.
        gap = (obs["time_ns"] >= 72) & (obs["time_ns"] <= 78)
        obs["iq_real"][gap] *= 0.1
        obs["iq_imag"][gap] *= 0.1
        _, calibrated, flags = diagnostics.pdv_estimates(obs, bootstrap=4)
        self.assertEqual(calibrated, {})
        self.assertIn("pdv_return_gap_abstain", flags)

    def test_invalid_windows_and_time_grids_are_rejected(self):
        obs = two_minima()
        for window in ([85, 65], [70, 70], [70, float("nan")], [101, 110], [0, 1, 2], [70, 70.1]):
            with self.subTest(window=window), self.assertRaises(ValueError):
                diagnostics.pdv_point(obs["time_ns"], obs["iq_real"], obs["iq_imag"], search_window_ns=window)
        for t in (obs["time_ns"][::-1], np.zeros_like(obs["time_ns"]),
                  np.where(obs["time_ns"] == 50, 50.01, obs["time_ns"])):
            with self.subTest(grid=t[:3]), self.assertRaises(ValueError):
                diagnostics.pdv_point(t, obs["iq_real"], obs["iq_imag"])

    def test_insufficient_local_fit_returns_explicit_error(self):
        t = np.arange(-10, 100, 2.0)  # Baseline is sufficient; local fit has only 3 points.
        phase = 4*np.pi*(-20*np.exp(-((t-44)/5)**2))/1550
        with self.assertRaisesRegex(ValueError, "samples around"):
            diagnostics.pdv_point(t, np.cos(phase), np.sin(phase))

    def test_fit_outside_requested_window_is_unresolved(self):
        obs = two_minima()
        with self.assertRaises(ValueError):
            diagnostics.pdv_point(obs["time_ns"], obs["iq_real"], obs["iq_imag"], search_window_ns=[35, 40])

    def test_noise_variance_needs_two_successful_draws(self):
        obs = two_minima()
        point = (20.0, 75.0, np.zeros(1), np.zeros(1))
        with patch.object(diagnostics, "pdv_point", side_effect=[point, point, ValueError("bad fit")]):
            _, calibrated, flags = diagnostics.pdv_estimates(obs, bootstrap=2)
        self.assertEqual(calibrated, {})
        self.assertIn("pdv_noise_propagation_failed", flags)


if __name__ == "__main__":
    unittest.main()
