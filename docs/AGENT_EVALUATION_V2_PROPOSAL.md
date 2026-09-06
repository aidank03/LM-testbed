# Proposed next evaluation contract

Status: historical proposal retained. Completion, evidence-presence, explicit task knowledge, and a fixed repeat campaign are implemented for the separately versioned v2 development check; see `AGENT_EVALUATION_V2_CARD.md`. Output-completeness and population-level statistical/accounting promotion criteria remain proposed. This does not change `factor-evaluation-contract/1`, its frozen criteria, the published baseline results, or the active local-model repeats. Those results must retain their original contract hash. No method is approved for production based on those gate booleans.

The v1 contract already prevents an always-abstaining or always-missing method from passing all gates: contract-decision accuracy and clean usefulness fail. However, a low wrong-decisive rate by itself is easy to obtain by producing few answers, and the current gates do not require every run to complete or cite an existing artifact.

Before generating or running a new evaluation campaign, freeze a versioned successor containing:

1. **Completion gate:** require every intended run to terminate with a valid final result. Missing, malformed, provider-failed, cancelled, and budget-exhausted runs remain explicit failures. A different allowed failure rate requires a stated product requirement.
2. **Evidence-presence gate:** require every final scientific decision to cite artifacts actually available in that run. Keep this separate from correctness and entailment; an existing artifact can still fail to support a claim.
3. **Output-completeness contract:** decide before evaluation whether each information track requires a numerical interval or allows a decision-only answer. Report numeric emission rates against the eligible cases. A small selection of accurate intervals must not stand in for complete diagnostic reconstruction.
4. **Selective-answer safeguards:** retain required-abstention recall, abstention precision, useful answer rate, and wrong decisive decisions both among all cases and among emitted decisions. Do not promote from any one favorable rate.
5. **Statistical decision rule:** distinguish passing an empirical fixture threshold from establishing an error-rate bound. Choose sample sizes using independent experiments and the intended confidence criterion. The current small local-model sample cannot substantiate a low population failure rate merely by observing zero failures.
6. **Repeat-aware comparison:** freeze repeat counts and retry budgets, score all assigned repetitions, and preserve grouping by original experiment. Report paired method differences and method variability without treating repeated analyses as new experiments.
7. **Accounting gate:** when making efficiency claims, require complete provider/model, prompt/tool versions, latency, call counts, and monetary accounting. Missing accounting is unknown, not free execution.

Run a new development check to verify these rules, then freeze an independently generated final campaign with the needed access separation. If the new development results lead to another rule change, the final campaign has not yet begun.

Plain-language takeaway: a workflow should not pass by answering only easy cases, omitting evidence, or quietly dropping failed runs.
