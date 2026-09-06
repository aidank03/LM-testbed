# Local candidate factory

The factory separates proposing a model, evaluating it, and accepting a candidate for a specified local use. It does not automatically deploy a model, change the active model, call a provider, or certify scientific correctness.

`factor.factory.assess_candidate(reference, candidate, contract)` applies explicit gates to two completed evaluation envelopes. `write_decision(reference, candidate, contract, path)` saves the resulting local acceptance or rejection record to a new file and refuses to overwrite one. Source reports and failed candidates remain intact.

Freeze a contract **before** running the comparative evaluation. Record its canonical `artifact_hash(contract)` in both reports. Changing a threshold afterwards changes the hash and blocks promotion against those reports. This is a consistency mechanism: a hash does not prove when a contract existed or prevent a dishonest evaluator from inventing evidence. The caller must preserve freeze provenance and use trusted evaluation runners with candidate outputs separated from truth.

A contract specifies:

- `schema_version: 1`, a concrete `task_id`, and an exact `metric_scope`.
- `minimum_independent_units`, using independent families or shots rather than frames or paraphrases.
- A nonempty list of `required_evidence_gates`.
- A nonempty `metrics` mapping. Each dotted metric path has explicit `min`, `max`, `min_delta`, or `max_delta` limits. Delta means candidate minus reference. Every limit must pass; there is no weighted score.

Each evaluation envelope supplies `candidate_id`, `completed: true`, the contract hash, matching task/scope, `independent_units`, `test_set_sha256`, `evidence_gates`, and `metrics`. Required evidence gates need `passed: true`, an artifact SHA-256 and a `verification` value of `deterministic_evaluator` or `independent_human_review`. Candidate self-assessments and an LLM judge are not accepted as the evidence authority. The factory checks declared metadata and hashes; it does not authenticate the declared producer or independently reread artifacts.

For the saved post-training experiment, `metrics` can contain the existing report's `before` or `after` dictionary. Paths include `accuracy`, `invalid_label_rate`, and `per_class.model_inferred.accuracy`. Its scope is `synthetic_provenance_classification`, and it has **four** independent family groups, not 48 independent experiments. It cannot be substituted for a motion reconstruction report. The existing research run did not predeclare a factory release contract; do not retroactively add a binding and call it a production acceptance test. Its measured improvement remains valid within the documented research scope.

An eligible result is only `eligible_for_local_release` under the named contract. Every result explicitly states `scientific_validation_established: false`. Insufficient independent units, incomplete evidence, a mismatched information/task scope, different test sets, a changed contract or any failed metric blocks eligibility. Claims about diagnostic validation, physical mechanisms or prospective decisions require their own independent scientific evidence and review.

The initial tests verify acceptance, single-gate rejection despite other improvements, family-level sample counting, task mismatch, post-hoc threshold changes, rejection of an LLM judge, incomplete evaluations, immutable source reports and refusal to overwrite a decision.

The next step is to predeclare a genuine candidate-release contract for one narrow use, then run the reference and candidate against a fresh evaluation appropriate to that use. This module provides the gate; it does not supply evidence that the gate's thresholds are scientifically sufficient.
