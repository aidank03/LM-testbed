# Motion-versus-drift baseline report

Status: development-exposed synthetic results. 216 observations; 144 independent synthetic experiment groups. Seed 20260905; 24 cases per ordinary condition and 48 per counterfactual/reference condition. No live AI or real-shot data in this comparison.

The bounded baseline correctly follows the stated inference contract and avoids making a physical claim in every structurally ambiguous case. Both estimators fail under unseen drift. The reference helps only while its transfer assumptions hold.

| Method | Condition | MAE (nm) | Coverage | Mean width (nm) | Wrong decisive / all | Abstention |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| naive | clean | 0.75 | 91.7% | 3.92 | 0.0% | 0.0% |
| naive | bounded_drift | 3.32 | 33.3% | 3.92 | 0.0% | 0.0% |
| naive | unseen_drift | 77.46 | 0.0% | 3.92 | 100.0% | 0.0% |
| naive | ambiguous | 27.50 | 0.0% | 3.92 | 50.0% | 0.0% |
| naive | reference | 27.50 | 0.0% | 3.92 | 50.0% | 0.0% |
| naive | reference_noisy | 26.69 | 4.2% | 3.92 | 12.5% | 0.0% |
| naive | reference_mismatch | 28.89 | 0.0% | 3.92 | 20.8% | 4.2% |
| bounded | clean | 0.75 | 91.7% | 3.92 | 0.0% | 0.0% |
| bounded | bounded_drift | 3.32 | 100.0% | 19.92 | 0.0% | 0.0% |
| bounded | unseen_drift | 77.46 | 0.0% | 3.92 | 100.0% | 0.0% |
| bounded | ambiguous | — | — | — | 0.0% | 100.0% |
| bounded | reference | 1.29 | 97.9% | 7.54 | 0.0% | 0.0% |
| bounded | reference_noisy | 25.00 | 91.7% | 100.08 | 0.0% | 50.0% |
| bounded | reference_mismatch | 80.26 | 0.0% | 7.54 | 100.0% | 0.0% |

## What failed

- In the deliberately adversarial unseen-drift condition, both methods made the wrong decisive physical conclusion in all 24 cases, and every reported interval missed latent material motion. The bounded method still matched the advertised-data contract: the missing physics was not detectable from that observation alone.
- The bounded method also failed in all 24 reference-mismatch cases. These constructed failures establish vulnerability, not the prevalence of this failure in real experiments.
- The naive method gave narrow intervals that missed both material-motion alternatives for every ambiguous pair. It ignored the reference when supplied.
- With a noisy reference, the bounded method had 25.00 nm MAE and abstained on half the decisions. More channels did not automatically provide 10 nm precision.

## What the successful cases establish

- The clean cases yielded 22/24 interval coverage for both methods. This is a small sample with a wide Wilson interval, not evidence of an exact 95% guarantee.
- The added-reference condition yielded 47/48 interval coverage and 1.29 nm MAE for the bounded method. The two counterfactuals and their signal-only counterparts share experiment groups; the report suppresses an invalid marginal Wilson interval for paired rows.
- The bounded method matched all 216 conditional decision labels and passed the provisional matched-model engineering gates. It nevertheless failed the mismatch conditions. There is no overall validated/safe designation.
- All eight evaluator implementation tests passed, including wrong-interval, always-abstain, fabricated-evidence, malformed-unit, duplicate-ID, and exposure controls. This verifies scoring behavior, not experimental physics.

## Reproduction and next decision

`comparison.json` contains per-case scores, regime summaries, rates and uncertainty intervals, and the separate gate results. `contract.json` was hashed before generation/execution. `code_manifest.json` records the evaluated source hashes. Candidate files, public observations, and evaluator truth are separately stored; this directory is published development material.

Use the creation and comparison example in `docs/AGENT_EVALUATION_CARD.md` with a new output directory, seed 20260905, n_per_condition 24, and split development. No final test set was used for this baseline report.

Next: compare bounded conventional inference with local and frontier model workflows using identical public cases and tools, while keeping physical accuracy distinct from tool-use and reporting correctness. A model that returns the same estimator output has not improved the measurement merely by explaining it.

Plain-language takeaway: the system can identify some unanswered questions, but a precise answer is still unreliable when an undetectable diagnostic assumption is wrong.
