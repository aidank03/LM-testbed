# Evaluation card: run_044066356fcb4a4099520a71b2aebec0

**Status: synthetic development run. No experimental data or LLM was evaluated.**

## What was tested

Prescribed compression/expansion and a known spatial mode were passed through synthetic PDV-like quadratures and radiographic measurements. The reducer received measurements and calibrations, with truth held separately for scoring.

## Results

| Scenario | Cases | Amplitude MAE (um) | Compression MAE (nm) | Compression interval coverage | PDV answer rate |
| --- | --- | --- | --- | --- | --- |
| blurred | 2 | 0.425 | 0.167 | 100.0% | 100% |
| no_structure | 2 | 0.032 | 0.225 | 100.0% | 100% |
| nominal | 2 | 0.052 | 0.375 | 50.0% | 100% |
| omitted_optical_drift | 2 | 0.096 | 3.430 | 0.0% | 100% |
| pdv_dropout | 2 | 0.087 | abstain | n/a | 0% |
| timing_jitter | 2 | 0.029 | 0.555 | 100.0% | 100% |

## Evaluation decision

The implementation is a development benchmark. Test known optical-path drift and realistic raw PDV processing before using its uncertainty intervals on experiments. A lost optical return causes an explicit abstention. Public seeds and keys must not be represented as a hidden AI test.

## Next experiment or calibration

Establish one real-shot record with raw measurements, source/return calibration, clock covariance and target metrology. First distinguish an optical-path change from material displacement. Preserve uncertainty in any model-assisted phase label.

Metrics, interval widths, finite-sample coverage bounds, predictions and run provenance accompany this card.
