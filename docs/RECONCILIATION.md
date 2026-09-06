# Factor v0.2.0 — inspection, reproduction and design reconciliation

2026-09-05 · Supersedes the initial “repository missing” assessment

## Finding

The user located the earlier source at `/Users/operator/Desktop/Factor_v0`. It contains the `liner-stability` 0.2.0 source package, tests, configuration, reference results, wheel and Git history bundle. This is the foundation to extend. Our small `factor_motion` example remains a separate analytical illustration; it should not replace the existing diagnostic pipeline.

The Desktop folder was not changed. An exact copy was made at `recovered/Factor_v0-0.2.0` in the authorized workspace. Tests and the full benchmark ran against the supplied wheel installed into a dedicated verification environment. The wheel's 15 Python source files match the source tree.

## What was verified

| Claim or artifact | New evidence | Status |
|---|---|---|
| Release identity | All 68 files listed in `RELEASE.json` match their SHA-256 values | Verified against the supplied manifest; not independent authentication of its author |
| Git identity | Bundle main, v0.2.0 tag and HEAD name `d3f48b01819738dcd1531f904833a61c3868e789`, matching the release | Bundle references inspected; no remote repository inferred |
| Installable wheel | Supplied wheel installed successfully in a dedicated environment; package imports from that installation | Reproduced; not a fresh source build or a completely isolated dependency environment |
| 33 implementation tests | Original unchanged suite: 33 passed | Reproduced |
| 240 synthetic cases | Original config, seed 1701, 40 cases in each of six scenarios | Reproduced |
| Saved truth, predictions and metrics | All numeric fields agree within `rtol=1e-9`, `atol=1e-12`; categorical fields match | Numerical reproduction, not byte-identical output |
| Optical-drift compression coverage | 0/40 intervals contain truth; all 40 cases emit estimates | Failure reproduced and retained |
| Live LLM, remote CI, real-shot validation | No such run performed | Still unverified / not run |

The current runtime uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.11.1. The saved run records Python 3.12.13 with the same NumPy and SciPy; it does not record Matplotlib. The largest numeric difference in the compared JSON artifacts is approximately `7.5e-11` in their respective stored units. Plots, archive timestamps and run timestamps were not expected to match byte for byte. The environment reuses the bundled Python site packages and has a recorded dependency snapshot.

Evidence: [source verification](../reports/original-v0.2.0/source-verification.json), [test output](../reports/original-v0.2.0/tests.txt), [run output](../reports/original-v0.2.0/run.txt), [comparison](../reports/original-v0.2.0/comparison.json), [reproduced evaluation card](../reports/original-v0.2.0/reproduced-run/EVALUATION_CARD.md).

## Failure result, preserved

In the omitted-optical-drift condition, compression MAE is **3.417 nm**, mean interval width is **1.209 nm**, and none of the 40 nominal 90% compression intervals contains the true value. The two-sided Wilson 95% interval for this observed coverage rate is approximately 0–8.76%. All cases produced answers. This is strong evidence of failure in this specified synthetic condition, not a measured failure rate on real shots.

The generator adds a drift of 0.10 nm/ns after 10 ns. The reducer's calibrated compression interval includes selected measurement/calibration terms but no optical-drift term. This is an intentional model-discrepancy test. Passing software checks does not repair that missing uncertainty, and the failure must remain visible when new methods are compared.

## The two motion targets are different

| Property | Original v0.2.0 | Initial design demonstrator |
|---|---|---|
| Observation | Synthetic baseband complex quadratures and calibration | Already reconstructed scalar apparent displacement |
| Target | Positive compression depth at a fitted local minimum, plus turnaround time | Signed negative-inward displacement over a fixed window |
| Window | Minimum search hard-coded to 30–60 ns | Prespecified window represented by one scalar |
| Scale | Prescribed compression depth 28–45 nm | Illustrative decision scale 1 µm |
| Interval | Approximate nominal 90% noise/calibration interval | Nominal 95% noise set widened by a supplied nuisance bound |

The 1 µm illustration is about 22–36 times the original total compression depth. It is **not an acceptance criterion for the rod application**. Distinct task IDs and contracts are required; changing units or signs does not turn minimum depth into fixed-window displacement. Keep the historical examples unchanged and choose the next task's required precision from the actual scientific decision. Reproducing either task does not validate the other.

## What to retain and how to extend it

| Existing component | Treatment in Factor |
|---|---|
| `workflow.run(config, out, override_n)` | Retain as evaluation orchestration. It generates observations and scores against truth, so it must not become the public production inference call. |
| `diagnostics.pdv_estimates(obs)` / `radiograph_estimates(obs)` | Reuse behind separate observation-only task adapters. A motion task should not require a radiograph. |
| `evaluation.py` | Preserve units, missing-case checks, answer denominators, widths and finite-sample coverage bounds; add bias, decision and abstention metrics for the new task contract. |
| `io.py`, manifests and existing CLI | Reuse atomic per-file writes, source identities and compatibility commands; add contract versioning and later run-level concurrency/recovery. |
| Evidence index and optional LLM adapter | Retain as optional capabilities; place actual cloud payloads behind operator policy before general product use. Fixture tests are not live provider validation. |
| Physics, radiography, design ranking and public AI cases | Preserve existing capabilities/tests. Do not expand first-product scope just because these modules exist. |

The revised architecture is one authoritative implementation based on the original package, with a small Factor-facing contract and a clear boundary between domain inference and evaluation. Avoid a broad package rename or framework rewrite during the first migration.

## Specific gaps to resolve

1. **Calibration window is ignored.** `simulation.py:59` supplies `pdv_search_window_ns`, while `diagnostics.py:32,53` use 30–60 ns directly. Matching defaults explain why the current run passes. A caller changing the calibration would silently get the old window. Honor the field or reject unsupported values; cover that behavior with a focused regression.
2. **Public IDs encode generator seeds.** `simulation.py:50` uses `demo_{seed}`, and `reporting.py:53–56` defines scenario seed ranges. The current reducer uses the ID only to seed its noise propagation, but a candidate with the public generator could reconstruct truth. The literal seed-free claim in `diagnostics.py:135` is too strong. Preserve these fixtures as public development data; use opaque IDs, an independent method RNG and an isolated evaluator for any future hidden test.
3. **Abstention and execution failure overlap.** The LLM benchmark runner records certain failures as null answers and `failed_or_abstained`. Separate execution failure from scientifically justified abstention before scoring abstention quality.
4. **Reference/ambiguity tracks are absent.** The six diagnostic scenarios do not test a modeled drift correction, an exact indistinguishable motion/drift pair, or a calibrated witness. Add these as explicit tasks and information tracks, retaining the old failure condition.
5. **The weak-mode interval needs a narrower claim.** In `diagnostics.py:121–128`, the low-significance radiographic bound omits the blur-uncertainty contribution included for detected modes. Its stated conservatism is conditional on the assumed blur; the zero-amplitude null does not test coverage for weak nonzero modes. Defer a new robustness claim until those cases are tested.
6. **Cloud authority belongs outside the method.** Evidence packets can include source paths as well as passages. Inspect the actual outgoing payload, enforce permitted destinations and budgets, and preserve available usage on failed calls. No production egress guarantee is established by the existing fixture tests.

These findings are recorded for the next implementation increment. Original source and reference outputs remain unchanged for reproducibility. We have not silently “fixed” the historical benchmark and then called it a reproduction.

## Next small increment

Keep the old benchmark command as a compatibility path. Introduce a versioned observation-only motion request/result adapter around the existing reducer; make the search-window behavior explicit; separate execution failure from scientific status; and add one exact ambiguity control. Choose the fixed-window or minimum-depth target explicitly before adding drift/reference estimates. The generic Factor runtime, team service and model routing remain later work.

**Plain-language takeaway:** the original setup is useful and reproducible. Factor should build on it, while making clear when its uncertainty assumptions—and therefore its answers—cannot be trusted.
