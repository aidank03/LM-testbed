# World-model course: verification and measured results

Verified on 2026-09-07 in the local LM-testbed checkout. **All ten notebooks executed successfully in fresh Jupyter kernels. All 13 focused tests passed.** The final stored executions total about 33.4 seconds on this machine; startup, model size and hardware affect runtime. Optional local-server inference was disabled and is not claimed as tested.

| Lesson | Code cells | Seconds | Execution |
|---|---:|---:|---|
| 05 | 7 | 1.54 | Passed |
| 06 | 7 | 1.71 | Passed |
| 07 | 8 | 2.11 | Passed |
| 08 | 6 | 14.87 | Passed |
| 09 | 6 | 2.2 | Passed |
| 10 | 6 | 2.33 | Passed |
| 11 | 5 | 2.3 | Passed |
| 12 | 6 | 2.03 | Passed |
| 13 | 6 | 2.06 | Passed |
| 14 | 9 | 2.29 | Passed |

[Machine-readable execution record](notebook_execution.json) · [Test output](world_model_tests.txt)

## Selected observations

- **Solver checks:** maximum autograd/finite-difference difference was 1.96e-10; energy-balance residual was 1.24e-06. These verify the implemented teaching equations, not real liner physics.
- **Joint inference:** the summary Jacobian has an exactly zero growth-sensitivity column. Mean-motion channels cannot identify the passive perturbation parameter in this model.
- **Uncertainty under wrong physics:** matched mass coverage was 89.1% for nominal 90% intervals; changing coupling and thermal-loss physics reduced it to 0.0%. The failed coverage result remains in the notebook.
- **Learned dynamics:** matched normalized rollout RMSE was 0.227, compared with 0.057 for the Euler baseline. The learned model was worse. Under shifted parameters its RMSE was 1.037.
- **Tiny LM:** raw and constrained generation each selected the correct tool for 3/4 held-out semantic cases. All four raw outputs were valid JSON. The one incorrect action prevented an otherwise feasible estimate. This tiny, unbalanced evaluation does not establish broad agent competence.
- **Capstone design:** speed was selected under the authored information-per-cost objective. In 24 independent synthetic shots, the mean mass interval width changed from 0.642 with current only to 0.148 with current plus speed. Augmented mass coverage was 22/24. These are conditional development results for the selected family, not a validation of the entire adaptive policy.

## Artifacts and verification scope

The notebooks contain 27 saved figures, inspected in three contact sheets. Tests cover limiting motion behavior, mass response, seed reproducibility, missing-channel removal, unobservability of growth, configuration errors, noiseless grid recovery, causal masking, split disjointness, tool-call rejection, missing-data abstention, valid numerical estimation and runner scope.

Configuration and learned checkpoints are retained under `runs/world-model/`. The exact ten final run reports are also collected in [world_model_metrics.json](world_model_metrics.json). Earlier development runs remain exposed evidence; no result here is a locked final test.

## Corrected verification-runner issue

The first integration runner selected older notebooks as well as the course. It was stopped. Notebook 01 was restored from its matching VS Code saved version and notebook 02 from the pre-existing matching project copy; interrupted copies were retained under `runs/world-model/runner-correction/`. Source cells matched those saved versions. Other earlier notebooks were not written. The runner now requires both a 05–14 number and course metadata, and failed execution writes a separate failure artifact instead of replacing a notebook. A regression test covers the selection boundary. Hashes confirm all earlier notebooks remained unchanged during the final course execution.

The first scoped run completed all ten lessons but inherited two old failure rows in its combined status report. That report is preserved in the correction folder. Report merging now excludes non-course rows; a further execution of lesson 05 completed with a clean 10/10 combined status.

## Remaining limits

The core is dimensionless and pre-stagnation. Radiation, spectral and perturbation responses are authored proxies. Inverse lessons operate on synthetic reconstructed summaries; uncertainty from the raw diagnostic pipeline is not yet propagated. The LM uses a closed command vocabulary and the agent episode makes one guarded call. Higher-fidelity physics, natural-language generalization, multi-turn acquisition, calibrated real-data inference and a native Factor provider adapter remain later increments documented in the architecture guide.

## Visual expansion

The updated course now has 54 static figures and two animations. The original measurements above are unchanged. See the [visual expansion verification](visuals-v1/verification.md) for the latest execution and preservation checks.
