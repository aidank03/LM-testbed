# Independent verification and practical review

This review records the code state tested during the separate live-model evaluation. No source files were edited for this review.

## Verification evidence

- Full combined suite: **100 tests passed in 11.997 seconds**, with `ResourceWarning` treated as an error.
- This includes the original 33 implementation tests and 67 new tests.
- Python source and test hashes were identical before and after the full run. The exact command, environment, versions and hashes are in `environment.json`; dependency versions are in `requirements-freeze.txt`; full output is in `unittest.txt`.
- Model and Slurm transports in the suite use fixtures. Actual local numerical subprocess execution, cancellation and timeout are tested. This suite is not evidence of live-model performance or a real cluster connection.
- An isolated temporary source copy built a 0.3.0 wheel with no network or dependency installation. From an isolated extraction, the new `factor doctor` entry-point function and a conventional scientific run both completed successfully. See `packaging-smoke.txt` and `packaging.json`. This did not install or modify the shared verification environment.

| Test module | Count |
| --- | ---: |
| Legacy diagnostics/grading | 13 |
| Legacy workflow/assistant | 20 |
| Factor evaluation | 8 |
| Factory | 8 |
| Runtime | 11 |
| HPC | 19 |
| Local model adapter | 8 |
| PDV window regression | 7 |
| Runtime/HPC boundaries | 6 |

## Corrections identified for the release pass

1. **Version provenance is inconsistent.** Distribution metadata and `factor.__version__` report 0.3.0, while `liner_stability.__version__` remains 0.2.0. The isolated wheel confirms the mismatch. Because legacy workflow manifests obtain their version from that module, resolve this before producing final release provenance. Root already plans the correction after the live evaluation.
2. **One offered CLI transport value fails locally.** `factor.cli.common` offers `openai-compatible`; `LocalChatClient` accepts `openai` or `lmstudio-native`. `LocalModel` forwards the value unchanged. Map the CLI alias or standardize the vocabulary, then test that CLI selection reaches the intended adapter without a real network request.
3. **CI does not smoke the new installed entry point.** The current workflow exercises `liner-stability doctor/run` after installation. Add `factor doctor` and one conventional `factor run` to make the newly delivered interface part of installation verification. The isolated wheel check here passed, but it is not a remote CI run.
4. **Expose idempotency at the job CLI before automated cluster use.** The SDK supports `request_id`, and the scientific runtime supplies its run ID. `factor job submit` currently exposes no equivalent option. A shell caller repeating that command creates another request; add a request identifier before recommending automated retry/recovery workflows. Preserve the explicit `submit_unknown` state.

The relevant controls already present include loopback-only local-model destinations with proxies/redirects disabled, a finite tool registry, strict action arguments, foreign-job rejection, explicit remote egress policy, fixed HPC commands, source/artifact hashes, and clear separation of fixture evidence from live validation. These controls are tested application behavior; the project does not claim an operating-system sandbox or a multi-tenant service.
