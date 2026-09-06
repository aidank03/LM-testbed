# Motion agent evaluation, explicit contract v2

Status: implementation checked; development campaign specified before its candidate runs. The v1 contract, outputs, failures, and interpretation remain unchanged. This card supplements `AGENT_EVALUATION_CARD.md`; the exact public instructions and provisional numerical gates are in the separately hashed v2 `contract.json`.

## Task and intended use

Determine whether signed material displacement over a fixed 100 ns window establishes inward motion beyond a provisional 10 nm resolution. This tests conditional diagnostic reconstruction and agent use of a scientific tool. It does not establish melting, an instability mechanism, performance on raw PDV, or validity for real rods or liners. No facility decision requirement is available, so the resolution and thresholds remain provisional engineering choices.

## Inputs and outputs

Every candidate receives the same public observation and the complete task definition below. Public observations contain a signed apparent displacement, signal noise standard deviation, absolute drift bound or null, resolution, units, window, source kind, and optional reference displacement/noise/covariance/residual bound. They contain no simulator truth, regime label, or seed. Negative displacement is inward. `synthetic_reconstructed` describes provenance; it does not imply zero physical motion.

The final output is `compression_supported`, `compression_not_supported`, `ambiguous`, or `out_of_scope`, with evidence IDs, explanation, and next step. All generated questions are in scope. Completion failure is not scientific abstention. Numerical estimates and intervals remain optional in this inherited interface: report their emission fraction alongside error and coverage; decision-only success cannot establish full reconstruction performance. Evidence presence means an artifact exists, not that it supports the claim.

## Transparent forward model and equal task knowledge

The scalar model is `y = x + d + epsilon`, with zero-mean Gaussian measurement noise. A reference, when present, measures drift: `r = d + eta + residual`. It does not measure material motion. Its residual bound is a calibration assumption whose real physical validity this synthetic exercise cannot establish.

All these rules appear verbatim in each public question, before model execution:

1. Without reference, center is `y`, variance is signal `sigma²`, and the nuisance bound is `drift_bound_m`.
2. With reference, center is `y-r`, variance is `sigma_y² + sigma_r² - 2*cov(y,r)`, and the applicable nuisance bound is `reference.residual_bound_m`, replacing the signal-only bound.
3. Null applicable bound means no finite bound and requires `ambiguous`; null never means zero drift.
4. Otherwise half-width is `1.959963984540054 * sqrt(variance) + bound`. Let `T = -resolution_m`.
5. `compression_supported` requires the upper endpoint to be at or below T. `compression_not_supported` requires the lower endpoint strictly above T. An interval crossing T requires `ambiguous`. Thus absence of convincing compression alone does not justify `compression_not_supported`.
6. Use stated assumptions for the conditional answer. Unseen drift or reference mismatch may be undetectable; do not guess a simulator label. The evaluator separately checks physical error and false confidence against latent truth.

Methods may compute directly or choose any allowed tool. No tool sequence is mandated. The bounded conventional method implements these same rules. The deliberately naive comparator ignores drift/reference corrections while receiving the same observations. Method differences therefore concern processing, not extra measurements.

## Truth, splits, and limitations

V2 reuses the frozen v1 generator and scoring implementation through a validated internal adapter. It changes the public task specification and adds completion/evidence gates, without rewriting v1 artifacts. Evaluator truth is stored separately and includes physical displacement, conditional decision, mismatch status, and experiment pairing. Labels are independently implemented rather than obtained from `infer_motion`.

The campaign uses development seed 20260906 and n=2, yielding 18 observations in 12 independent synthetic experiment groups. Exact ambiguous counterfactual pairs and their reference tracks share group identity. The seven regimes are clean, bounded drift, unseen drift, exact ambiguity, reference, noisy reference, and reference mismatch. Report regimes separately. Separate seeds and split-specific streams support train/development/final separation; no final campaign is used here.

The generator and reconstruction share an additive Gaussian model. Deliberate mismatch is useful but does not make this an independent physical-model validation. A simulator label is true only within this simulator. Reference transfer needs experimental calibration, including sensitivity to mismatch, lag, and covariance.

## Frozen method campaign and budgets

The machine-readable manifest in `reports/agent-v2-suite/campaign.json` specifies the campaign and its hash before candidate runs. Compare naive and bounded conventional outputs with two local models, `factor-qwen35-9b` and `factor-nemotron4b`, each in direct and tool-enabled modes. Assign two runs to every case for each local method; preserve every attempt, including failures. Do not select the best repeat or silently retry failed cases.

Both local modes have the same six-step/six-call maximum, 120-second run limit, 700-token per-response maximum, temperature zero, reasoning disabled, no data egress, and no cloud or HPC job authorization. Direct mode can finish; tool mode can inspect the observation, call motion inference, and finish. Models receive identical case definitions within a mode. Record actual source/model/prompt/tool versions and endpoint metadata per run; an alias alone is not a reproducible weight identity. Hardware use and electricity are unmetered, even if local API charges are zero.

This is a small development comparison. Two repeated analyses do not create twice as many independent scientific cases. Repeats describe method variability; they do not support a population-level reliability guarantee. Frontier and human baselines are absent and must not be invented. V1-v2 changes involve both instructions and a fresh sample, so their difference is not a clean causal estimate of instruction quality.

## Metrics and failure gates

Retain separate conditional-decision accuracy, physical wrong-decisive rate, required-abstention recall and precision, numerical emission, bias, MAE, RMSE, coverage, interval width, completion, evidence presence, latency, calls, and cost/accounting completeness. Keep the v1 provisional matched-regime gates. V2 adds empirical 100% completion and evidence-presence gates. These are checks of the assigned fixture, not claims that the true failure probability is zero. No weighted score or automatic production promotion is allowed.

Wilson intervals apply to independent observations or explicitly identified independent experiment-group quantities. Paired-row marginal rates are descriptive. Include deliberate wrong intervals, fabricated evidence, missing runs, duplicate/unknown prediction IDs, truth leakage, altered manifests/questions, and cross-version truth in evaluator tests. Existing artifact IDs never replace human examination of evidence support.

## Reproduction and next decision

Use `factor.evaluation_v2.create_suite(...)`, `baseline_results(...)`, `score_agent_runs(...)`, and `compare_methods(...)`; the public schema is `factor-cases/2`. Run `PYTHONPATH=src python -m unittest discover -s tests -p 'test_factor_evaluation*.py'`. The v2 adapter rejects altered public questions and mismatched manifests/truth. A final suite, if later created, records exposure and cannot be called a fresh final test after reuse.

After this development campaign, decide whether the interface and deterministic tools are reliable enough for a larger independent check. Fixes motivated by observed failures require a new version and a new evaluation, while preserving these results.

Plain-language takeaway: every method now gets the same scientific rules, but a reliable software answer still depends on whether those rules describe the measurement.
