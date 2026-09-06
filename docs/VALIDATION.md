# Release verification: version 0.2.0

Verified locally on 5 September 2026 with Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8.

| Check | Result |
| --- | --- |
| Editable package installation and CLI | Passed |
| Regression and integration suite | 33 tests passed |
| Full configured synthetic workflow | 240 cases completed across six scenarios |
| Comparison with the version 0.1 numerical scorecard | Identical diagnostic metrics after module refactoring |
| Analytic calculation CLI | Produced magnetic pressure and explicit units/assumptions |
| Experiment-design CLI | Ranked the illustrative candidates and labeled the screening criterion |
| Evidence workflow | Indexed explicit documents and produced an offline evidence packet |
| Public AI tasks | Twelve candidate cases and a separate key exported |
| Provider integration | Completion/refusal/error/schema/reference behavior tested with injected fixtures |
| Distribution | Wheel built, installed in a separate environment and exercised outside the repository; local Git history and a Git bundle included in the release |

The full reference run is in `examples/reference_run/`, including the configuration, evaluation card, raw examples, predictions, scores, plot and package-source hashes. The twelve public candidate tasks are in `examples/ai_cases/`.

**Not established:** a live API-backed model run, remote CI execution, real-shot validation, MHD or phase-model accuracy, an unseen-test generalization result, or a deployed GitHub repository. An API key was not configured. Provider fixtures are software tests and do not provide an AI accuracy score.

The known optical-drift failure is preserved in the release. Passing implementation tests must not be interpreted as passing all physical or uncertainty-calibration tests.
