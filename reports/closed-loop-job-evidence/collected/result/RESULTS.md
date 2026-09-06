# Synthetic diagnostic evaluation

SYNTHETIC DEVELOPMENT DEMO; no real data or LLM evaluated

12 cases, 2 per scenario. Seed 1701. No fit to experiment.

| Scenario | Amplitude MAE simple / corrected (um) | Corrected amplitude interval coverage | Compression MAE (nm) | Compression interval coverage | PDV answer rate |
| --- | --- | --- | --- | --- | --- |
| nominal | 0.30 / 0.05 | 100% | 0.37 | 50% | 100% |
| blurred | 7.22 / 0.43 | 100% | 0.17 | 100% | 100% |
| pdv_dropout | 0.55 / 0.09 | 100% | abstain | n/a | 0% |
| timing_jitter | 0.34 / 0.03 | 100% | 0.55 | 100% | 100% |
| omitted_optical_drift | 0.32 / 0.10 | 100% | 3.43 | 0% | 100% |
| no_structure | 0.03 / 0.03 | 100% | 0.23 | 100% | 100% |

Coverage is empirical, conditional on answering. Most intervals are approximate 90% intervals; low-significance amplitude results use conservative 90% upper bounds and are separately counted in metrics.json. Wilson intervals and interval widths are in the JSON scorecard. These are not pass/fail certification thresholds.

The optical-drift scenario adds an unmodeled optical-path contribution. Any resulting coverage failure demonstrates a limitation of the reconstruction, not an instability discovery. The mean-displacement minimum is a kinematic turnaround, never a melt label.

Raw examples and calibration metadata are included for six cases. Regenerate all measurements from the manifest. The generator and inverse share ideal material assumptions even though the radiographic integrations differ.
