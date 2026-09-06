# Model-driven local scientific workflow

**The local model completed the requested workflow and correctly returned an ambiguous motion result.** It submitted and checked a real local numerical job, collected its outputs, ran the motion estimator, recorded an unresolved hypothesis, and finished with a useful additional-measurement recommendation.

This was one explicitly guided demonstration, not a comparison showing that AI improves scientific decisions. The request specified the workflow, local backend, case count, seed and desired hypothesis update. The model produced the seven tool actions during live inference; the trace is not a scripted model fixture.

## What actually ran

| Item | Observed result |
| --- | --- |
| Model | Qwen3.5-9B Q4_K_M through LM Studio; alias `factor-qwen35-loop` |
| Context/settings | 16,384-token loaded context; temperature 0; reasoning off; maximum 700 output tokens per call |
| Model calls / tool actions | 7 / 7 |
| Action sequence | submit → status → status → collect → infer motion → update hypothesis → finish |
| Numerical jobs | 1 local process; completed with exit code 0 |
| Synthetic cases | 12 unique cases; 2 in each of 6 legacy diagnostic conditions |
| Entire workflow | 56.337 seconds |
| Sum of model-response times | 54.092 seconds |
| Numerical worker duration | 10.052 seconds, overlapping model execution |
| Reported tokens | 23,199 input tokens across repeated contexts; 878 output tokens |
| External cloud cost | Recorded as zero; local hardware/energy cost was not measured |
| Hypothesis updates | 1, retained as unresolved |

The model identity and loaded context are supported by a saved read-only server inventory obtained after the run. The instance metadata and related weight provenance are preserved in the evidence snapshot; model weights were not copied. No remote scheduler or cluster participated. Calling this a successful Slurm/HPC deployment would be incorrect.

## Scientific conclusion and preserved failures

The supplied reconstructed observation was −30 nm with a stated 1 nm noise scale, a 30–60 ns window and a provisional 5 nm decision scale. Its drift bound was unknown, and it had no reference channel. In the stipulated observation model, −30 nm of material motion with zero drift and zero motion with −30 nm of drift produce the same measurement distribution. The final ambiguity decision follows from this non-identifiability.

The model recommended a calibrated drift witness with a defensible residual mismatch bound. It did not assign a material-motion estimate or claim melting, ETI or MRTI. Its unresolved hypothesis update explicitly states that the independent synthetic job cannot establish the true motion of the supplied observation. This is an appropriate conclusion under the task assumptions; it is not validation on a real diagnostic.

The numerical job estimated **positive compression depth near a fitted minimum**, a different target from the supplied signed fixed-window displacement. Its calibration-aware depth results were:

| Synthetic condition | Answered / cases | Compression MAE, nm | Truth covered by nominal 90% interval |
| --- | ---: | ---: | ---: |
| Nominal | 2 / 2 | 0.375 | 1 / 2 |
| Blur | 2 / 2 | 0.167 | 2 / 2 |
| Timing jitter | 2 / 2 | 0.555 | 2 / 2 |
| Omitted optical drift | 2 / 2 | 3.430 | **0 / 2** |
| Lost PDV return | 0 / 2 | — | No interval; abstained |
| No imposed structure | 2 / 2 | 0.225 | 2 / 2 |

The drift failure remains visible. Two cases per condition are insufficient to estimate a reliable population coverage rate; even the nominal condition missed one case. The generator and inverse retain shared simplifying assumptions. These synthetic results do not identify an instability mechanism or validate transfer to imploding liners.

## Independent audit and retained evidence

The audit verified all **17 chained events**, **9 run artifacts**, **16 staged numerical/worker source files**, and **27 collected artifacts totaling 748,792 bytes**. Request/configuration identities, job ownership, final-result integrity and source hashes agree. Re-scoring both saved prediction files against the 12 stored truth records reproduced the saved metrics exactly. That checks numerical record consistency, not independent physical truth.

All 15 controller source files now match the hashes recorded for the run. A changed CLI file was recovered to its exact recorded digest before inclusion. The snapshot preserves numerical and controller source, job records, all collected outputs, and model-instance metadata, while excluding redundant stage outputs, caches and model weights.

- [Original run and result](closed-loop/run_044066356fcb4a4099520a71b2aebec0/result.json)
- [Observable event trace](closed-loop/run_044066356fcb4a4099520a71b2aebec0/events.json)
- [Audit results](closed-loop-job-evidence/AUDIT.json)
- [Snapshot files, original paths and hashes](closed-loop-job-evidence/SNAPSHOT_MANIFEST.json)
- [Saved numerical results](closed-loop-job-evidence/collected/result/RESULTS.md)

The demonstration establishes that the callable environment can connect a live local model to a bounded numerical job and preserve an appropriate scientific limitation through the final answer. It does not establish superiority over a conventional scripted workflow, a repeatable model success rate, remote HPC readiness, or accuracy on real measurements.
