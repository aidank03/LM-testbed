# Factor motion and agent evaluation card

Contract: `factor-evaluation-contract/1`; generator: `factor-generator/1`; task: `motion.fixed_window/1`.

Status: synthetic development benchmark. No independent real-shot data or measured expert baseline is available. This task measures a fixed-window displacement and decision, separately from the legacy minimum-depth benchmark.

## 1. Scientific question and intended use

Can the allowed reconstructed optical measurement support inward material motion beyond a specified displacement resolution, and can the workflow identify cases where it cannot? This is a diagnostic sufficiency question. It cannot establish melting, instability birth, or an ETI/MRTI mechanism.

The target is signed radial displacement from 0 to 100 ns. Negative means inward. The provisional decision resolution is 10 nm. This value makes a transparent engineering fixture; it is not a facility-approved requirement.

## 2. Allowed inputs

Each public case contains only an opaque case ID, a question, and a validated `motion-observation/1` object. That observation supplies apparent displacement in meters, Gaussian noise standard deviation, a drift bound or explicit absence of a bound, optional reference data, window, resolution, source kind, and calibration identifier. Unknown fields—including truth fields—are rejected.

The signal-only and reference-available cases are separate information tracks. Both conventional baselines and every AI method receive the same public object within a case. Condition names, latent material motion, counterfactual identities, physical decision labels, and the generator seed exist only in `evaluator_truth.json`.

## 3. Required outputs

An agent run returns execution status and a final decision: `compression_supported`, `compression_not_supported`, `ambiguous`, or `out_of_scope`. Optional numerical results carry a point and finite ordered interval in meters. An explicit unresolved numerical result has no finite point/interval. Execution failure and absent output are failures, not scientific abstentions.

The grader checks cited evidence IDs against the supplied artifact map. This measures evidence presence only, not entailment, provenance authenticity, or scientific correctness. Runtime artifact-integrity checks are separate.

## 4. Transparent forward model and reference truth

The generator uses reconstructed displacement, not the recorded PDV waveform:

`y = x + d + epsilon`; optional `r = d + delta + eta`.

Here `x` is material motion, `d` equivalent optical-path drift, `delta` reference-transfer mismatch, and `epsilon/eta` Gaussian measurement errors. Generator noise is independent across channels; covariance is explicitly supplied as zero. The inference interface supports covariance, but this generator does not validate nonzero-covariance behavior.

| Condition | Deliberate challenge |
| --- | --- |
| Clean | 1 nm signal noise; zero drift declared and generated. |
| Bounded drift | Uniform drift within a supplied ±8 nm bound. |
| Unseen drift | Actual drift reverses apparent motion; advertised bound remains zero. |
| Exact ambiguity | Two latent motions, −35 and +20 nm, produce identical allowed measurements, apart from arbitrary IDs; nuisance is unbounded. |
| Added reference | Those counterfactuals receive separate calibrated reference observations; 1 nm reference noise and a 1 nm residual bound. |
| Noisy reference | 25 nm reference noise challenges practical resolution despite additional information. |
| Reference mismatch | An unadvertised transfer error reverses corrected motion, despite an apparently precise reference. |

Latent displacement is exact within this generator only. The model shares additive, Gaussian, fixed-window assumptions with the estimators. Mismatch cases deliberately violate bounds; they are not independent high-fidelity physics. Nonlinear optical response, signal loss, phase reconstruction, and real calibration remain future tests.

## 5. Splits and evaluator integrity

`train`, `development`, and `final` use separate deterministic random streams keyed by split and seed. Opaque hashed IDs do not encode seed, condition, or physical answer. Counterfactual signal-only/reference variants share an evaluator experiment ID; they must never be treated as independent shots.

The complete contract and its hash are written before case generation or candidate execution. Scoring verifies the manifest, public-data hash, exact truth schema, and labels. A separate exposure record marks final scoring; later scores are explicitly nonconfirmatory. Passing dictionaries instead of paths cannot enforce persistent final exposure, so such final results are labeled untracked and nonconfirmatory.

These local files are not an access-control boundary. Untrusted candidate processes require an observation-only mount or equivalent restricted tools. Anyone who sees evaluator keys or uses a scored suite for tuning must treat it as development data. The published generator and this task card also limit claims of unseen-distribution testing. The current baseline report is explicitly development-exposed.

## 6. Baselines and comparison budgets

The naive estimator assumes zero drift and ignores available nuisance/reference information. The bounded estimator uses the supplied reference and residual bound, or the signal-only drift bound. Both receive identical information. The bounded estimator is the stronger practical comparison; beating the naive baseline alone is insufficient.

Compare conventional, local-model, and frontier-model workflows on the same case IDs, prompt contract, tools, maximum attempts, and data access. Record model identifier, provider, prompts, tool versions, latency, tool/model calls, and monetary cost. The scorer sums reported numeric usage but does not independently meter it. Missing or incomplete accounting must not be reported as zero cost. No human result should be invented.

One result per case is accepted in each named method. Stochastic repeats should be separate named result sets, with fixed seeds when supported; this version does not calculate a pooled repeat-aware superiority test. Repeated methods use the same experiment groups and are not independent experiments.

## 7. Metrics and provisional gates

Report bias, MAE, RMSE, interval coverage, mean interval width, completed-run rate, contract-decision accuracy, wrong decisive decisions (both per case and among decisive outputs), abstention precision/recall, and evidence presence. Keep regime reports; there is no weighted total score. Finite intervals count toward coverage even when the decision is ambiguous; missing intervals do not count as covered. Coverage is conditional on numerical output and must be read with completion and usefulness.

The contract's engineering gates are: matched-model MAE ≤10 nm, absolute bias ≤3 nm, empirical coverage ≥90%, wrong decisive decisions ≤5%, contract-decision accuracy ≥95%, abstention on every structurally ambiguous case, and decisive results on ≥90% of clean cases. Clean, bounded-drift, and added-reference regimes also have separate MAE, coverage, and contract-decision gates. The 10 nm error scale matches the fixture's resolution; the bias gate limits systematic error to a minority of that scale. Other cutoffs are provisional tolerances, not statistically proven guarantees or facility requirements.

Nominal Gaussian coverage is 95%; a nuisance bound enlarges the confidence set. Wilson intervals accompany rates when rows are independent. For paired cases, marginal Wilson intervals are omitted; a separately labeled interval describes the event that all associated cases satisfy that metric in an independent experiment group. That group statistic is not a confidence interval for the marginal per-case rate. Small live-model samples will have wide uncertainty.

## 8. Distinguish contractual correctness from physical correctness

The evaluator independently constructs the feasible-motion interval from allowed observations and the frozen assumptions. Unbounded nuisance requires ambiguity. A finite interval crossing −10 nm requires ambiguity because the resolution decision is uncertain. A resolved interval determines the contract decision.

The latent physical decision is scored separately. Undetectable unseen drift may yield a correct conditional decision and an incorrect physical decision. The benchmark preserves that failure instead of rewarding an algorithm for guessing hidden labels or requiring impossible mismatch detection.

## 9. Negative controls and prohibited claims

Tests include known 5 nm errors and 2 nm intervals that always miss truth, always-abstaining outputs, missing runs, duplicate/unknown IDs, invalid units, fabricated evidence IDs, counterfactual equality, altered manifests, and repeat final exposure. The generator is tested while candidate inference is disabled.

Passing these tests verifies evaluator implementation. Calibrated synthetic reconstruction does not validate the optical hardware or physical model. An agent that calls the same bounded estimator cannot claim a new measurement-accuracy improvement merely for restating its output. Any workflow benefit must concern a separately measured function, such as avoiding an unsupported premise or reducing expert time.

## 10. Reproduction

From the repository, use the existing Python environment with `src` on its import path:

```python
from factor.evaluation import create_suite, baseline_results, compare_methods
paths = create_suite("reports/new-development-suite", seed=20260905,
                     n_per_condition=24, split="development")
reports = compare_methods(paths["public_cases"], paths["evaluator_truth"], {
    name: baseline_results(paths["public_cases"], name)
    for name in ("naive", "bounded")
})
```

There are nine observations per requested condition count, including counterfactual/reference counterparts: the example produces 216 cases. Output directories must be new or empty. Run `python -m unittest discover -s tests -p test_factor_evaluation.py -v` for evaluator checks. Record package/source version alongside each candidate run; the generated contract and public-data hashes identify its benchmark inputs.

Plain-language conclusion: more information can resolve ambiguity only when the added measurement is trustworthy. A confident answer can still be wrong when the diagnostic assumptions are wrong.
