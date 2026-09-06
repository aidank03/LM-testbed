# Local agent development comparison: preserved v1 results

The tested Qwen 3.5 9B workflows did not meet the frozen v1 decision contract. The bounded conventional method followed that contract, while remaining confidently wrong under the deliberately unmodeled drift and reference mismatch. These results do **not** establish that local models are intrinsically inferior: the v1 candidate question omitted sign, interval, reference, and decision-label definitions that the conventional methods already encoded. The failed interface and candidate outputs are preserved; v2 supplies those definitions before a separate development campaign.

## What was actually evaluated

The same 18 synthetic observations, representing 12 independent experiment groups, were supplied to naive and bounded conventional methods and to `factor-qwen35-9b` in two local modes. Tool mode could inspect observations and call `infer_motion`; direct mode could only finish. Each local mode ran twice, at temperature zero with reasoning disabled and a 700-token response cap. Both repeats produced identical final decision/failure status case by case. They are repeated analyses of 18 observations, **not 36 independent scientific cases**.

The v1 tool policy allowed six steps/calls; direct allowed two. Tool schemas and available actions also differed. Source manifests show that the v1 evaluator, motion inference, runtime, artifact, contract, and local transport files retained their hashes; CLI/provider hashes changed between arms and new unrelated modules appeared during the campaign. The records therefore describe these concrete interfaces and software versions, rather than a controlled estimate of the causal benefit of tool availability. No frontier model, expert-human workflow, held-out real data, or experimental outcome was measured.

Sources: [tool results](agent-local-tools/comparison.json), [direct results](agent-local-direct/comparison.json), [public cases](agent-development-suite/public_cases.json), [frozen v1 contract](agent-development-suite/contract.json), and [independent trace audit](agent-v1-trace-audit.json).

## Decision and execution results

Local rows below describe **each** repeat; both repeats had the same counts. Failure remains in the denominator. Contract correctness uses the advertised observation assumptions. Physical wrong-decisive counts use the simulator's actual material motion; these are different questions.

| Method | Completed | Contract correct | Ambiguous answers | Required ambiguity recovered | Physical wrong-decisive / all cases | Finite numerical outputs |
|---|---:|---:|---:|---:|---:|---:|
| Naive conventional | 18/18 | 8/18 (44.4%) | 1/18 | 0/6 | 7/18 | 18/18 |
| Bounded conventional | 18/18 | 18/18 (100%) | 6/18 | 6/6 | 4/18 | 14/18 |
| Qwen tools, each repeat | 18/18 | 5/18 (27.8%) | 0/18 | 0/6 | 8/18 | 1/18 |
| Qwen direct, each repeat | 17/18 | 7/18 (38.9%) | 0/18 | 0/6 | 7/18 | 0/18 |

The direct mode's wrong-decisive rate among **emitted decisive answers** is 7/17, not 7/18; the missing answer is a provider failure, not caution. Tool mode is wrong in 8/18 emitted decisive answers. Every completed local result cites an existing artifact. That fact did not ensure correct interpretation or arithmetic.

| Regime | Cases | Bounded contract correct | Qwen tools correct, each repeat | Qwen direct correct, each repeat |
|---|---:|---:|---:|---:|
| Clean | 2 | 2 | 1 | 1 |
| Bounded drift | 2 | 2 | 1 | 2 |
| Unseen drift | 2 | 2 | 0 | 1 |
| Exact ambiguity | 4 | 4 | 0 | 0 |
| Reference | 4 | 4 | 2 | 2 |
| Noisy reference | 2 | 2 | 0 | 0 |
| Reference mismatch | 2 | 2 | 1 | 1 |

Small samples make rates unstable. For example, a separate metric requiring all variants within an independent experiment group to have correct contractual decisions is 12/12 for bounded conventional (Wilson 95% interval 75.8–100%) and 3/12 for Qwen tools (8.9–53.2%). Those are **group all-pass intervals**, not confidence intervals for the paired-row percentages above. Neither two successful clean cases nor two identical model repeats establishes population reliability.

## Numerical performance and preserved physical failure

The matched-model subset excludes unseen drift and reference mismatch. It contains 14 observations, four of which are structurally ambiguous with no finite identified interval. On this subset the bounded estimator emitted 10 finite intervals, with MAE 6.22 nm, bias −3.37 nm, RMSE 10.46 nm, and coverage 10/10. Its empirical bias therefore **failed** the provisional 3 nm gate on this small sample. Naive inference emitted 14 intervals, MAE 22.35 nm, and coverage 2/14.

Qwen tools emitted just **one** numerical interval per repeat: error 8.76 nm and coverage 1/1. This is inadequate for comparing full recovery accuracy; a favorable coverage metric on one selected case is not success on 18 cases. Direct mode has **no registered numerical output field**: `numerical_result` is populated only when `infer_motion` runs, while `finish` accepts a decision and prose. Its zero machine-scored intervals are therefore an interface limitation, not evidence of inability to calculate an interval. Numerical accuracy and calibration of direct calculation are unavailable. This output asymmetry also limits the forthcoming v2 campaign to decision comparisons across direct and tool modes; a future reconstruction comparison needs an equal structured interval interface.

| Bounded conventional regime | MAE (nm) | Finite interval coverage | Scientific implication |
|---|---:|---:|---|
| Clean | 0.55 | 2/2 | Transparent matched case works |
| Bounded drift | 6.96 | 2/2 | Supplied nuisance bound protects these intervals |
| Unseen drift | 84.10 | 0/2 | Correct conditional calculation, false physical confidence |
| Exact ambiguity | Not identified | No finite interval | Same data admit incompatible material motions |
| Reference | 1.37 | 4/4 | Reference helps under the assumed transfer model |
| Noisy reference | 20.88 | 2/2 | Broad uncertainty requires abstention on both cases |
| Reference mismatch | 77.64 | 0/2 | An incorrectly trusted reference can reverse the inference |

The wider [216-case conventional development report](motion-development/BASELINE_REPORT.md) is a different sample and is not pooled into this local comparison. It also preserves complete coverage failure under unseen drift and reference mismatch. No method can be expected to detect an omitted effect that is indistinguishable from motion using the permitted observations. Contractual correctness cannot repair a false physical assumption.

## Examination of observable answers and evidence

The following are manual checks of public final answers and recorded actions, not scores from an LLM judge or examination of private reasoning. All first-repeat local final answers were read; second-repeat decision/failure statuses and integrity were checked.

- Both modes called a **positive 50.6 nm** displacement compression by comparing magnitude with resolution. V1 omitted the inward-negative convention, making this partly an evaluator-interface failure. [Direct example](agent-local-direct/runs/run_5746bbf179a64e0dbee6ade0dc0c62c4/result.json).
- A tool-mode answer described **4.55 nm as approximately 4.5 times 10 nm**. That arithmetic is wrong independently of the missing sign convention. It also acknowledged unknown drift but did not choose ambiguity. [Example](agent-local-tools/runs/run_add3183ff69b4e82a9313622525e0304/result.json).
- A noisy-reference answer treated the signal's 1 nm uncertainty as sufficient while the reference had 25 nm noise. The proper reference variance was absent from the public question, but the data did contain both uncertainty values. [Direct example](agent-local-direct/runs/run_118a75ea03964e8fb9a88109626f6df2/result.json).
- Several direct result fields said `compression_supported` while their final explanations concluded that compression was not supported. This is an output-consistency failure independent of which label is scientifically appropriate. [Example](agent-local-direct/runs/run_1ebcd31bd90c49069f3ba2741cb25512/result.json).
- The same ambiguous case failed with incomplete local output on both direct repeats. It was retained as `provider_failed`; no scientific abstention credit was given. [First failure](agent-local-direct/runs/run_523abb04cd174b2292d22d3cba021e9d/result.json).
- Tool mode chose `infer_motion` only once per repeat; that run produced its sole finite interval and a supported conditional decision. Most runs inspected then answered without the numerical tool. The existence of a correct tool did not guarantee its use. [Numerical example](agent-local-tools/runs/run_e001e04a32b64afab670cc3ff28c27ff/result.json).

All 72 local event chains and stored artifact hashes passed integrity verification. Their historical result files were not yet bound to the event chain; the verifier correctly labels them `unbound_legacy_result`. Integrity does not authenticate a source or validate physics. A later runtime binding fix applies prospectively, and these historical files were not rewritten.

## Cost, calls, and next decision

Tool mode used 36 model calls per repeat: 17 inspections, one inference, and 18 finishes. Direct mode used 18 model calls, yielding 17 finishes and one provider failure. Total elapsed times were 116.97/103.99 seconds for tool repeats and 98.89/86.05 seconds for direct repeats. Timing is descriptive and includes startup/cache/system-state differences; no conventional end-to-end latency comparison was measured.

The original aggregate token counters incorrectly say zero because native provider token keys were not mapped into the runtime counters. Raw event metadata preserves 46,282 input and 3,161 output tokens per tool repeat. Successful direct outputs preserve 14,924 input and 3,576 output tokens per repeat, **excluding the failed call whose usage was unavailable**. Those direct totals are lower bounds. The adapter mapping was corrected for future runs; historical aggregates remain intact. Recorded local API charges are zero; hardware, electricity, and expert time were not measured.

The next comparison uses the separately frozen [v2 task card](../docs/AGENT_EVALUATION_V2_CARD.md) and [campaign manifest](agent-v2-suite/campaign.json), with explicit scientific rules for every method and equal maximum budgets for the two interfaces. V1 results must not be regraded as if those instructions had originally been present. No method is promoted based on this report.

Plain-language takeaway: the conventional calculation is currently more dependable within its stated assumptions; the first local agent test exposed both model mistakes and an unfairly incomplete task definition. Better instructions can be tested, but neither instructions nor tools make hidden diagnostic drift disappear.
