# Evaluation card: synthetic-development-001

**Status: synthetic development run. No experimental data or LLM was evaluated.**

## What was tested

Prescribed compression/expansion and a known spatial mode were passed through synthetic PDV-like quadratures and radiographic measurements. The reducer received measurements and calibrations, with truth held separately for scoring.

## Results

| Scenario | Cases | Amplitude MAE (um) | Compression MAE (nm) | Compression interval coverage | PDV answer rate |
| --- | --- | --- | --- | --- | --- |
| blurred | 40 | 0.497 | 0.250 | 95.0% | 100% |
| no_structure | 40 | 0.062 | 0.360 | 87.5% | 100% |
| nominal | 40 | 0.059 | 0.235 | 92.5% | 100% |
| omitted_optical_drift | 40 | 0.061 | 3.417 | 0.0% | 100% |
| pdv_dropout | 40 | 0.063 | abstain | n/a | 0% |
| timing_jitter | 40 | 0.052 | 0.366 | 87.5% | 100% |

## Evaluation decision

The implementation is a development benchmark. Test known optical-path drift and realistic raw PDV processing before using its uncertainty intervals on experiments. A lost optical return causes an explicit abstention. Public seeds and keys must not be represented as a hidden AI test.

## Next experiment or calibration

Establish one real-shot record with raw measurements, source/return calibration, clock covariance and target metrology. First distinguish an optical-path change from material displacement. Preserve uncertainty in any model-assisted phase label.

Metrics, interval widths, finite-sample coverage bounds, predictions and run provenance accompany this card.
