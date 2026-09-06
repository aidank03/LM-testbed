# Factor job execution

Factor now executes a real numerical subprocess locally and implements an opt-in Slurm adapter over SSH. Both use the existing synthetic diagnostic benchmark. The local execution checks are real; Slurm checks use injected command responses. No cluster has been configured, contacted, or validated for this integration.

This is one callable job capability for an agent's bounded workflow. It does not itself interpret results, update scientific hypotheses, operate equipment, install dependencies, or accept generated programs. An agent can submit a permitted experiment, inspect its status, collect artifacts, and then pass those artifacts to a separately evaluated scientific analysis step.

## Run locally

Use a POSIX local host and the Python environment containing Factor, NumPy, SciPy and Matplotlib. From the repository root:

```python
import json
from pathlib import Path
from factor.hpc import HPCManager

manager = HPCManager(**json.loads(Path("configs/hpc_local.json").read_text()))
experiment = json.loads(Path("configs/hpc_smoke.json").read_text())
job = manager.submit(experiment, backend="local", request_id="first-local-check")
job = manager.wait(job["job_id"], timeout_seconds=30)
if job["status"] == "completed":
    job = manager.collect(job["job_id"])
    print(job["artifact_manifest"]["directory"])
```

The smoke configuration executes two cases in each of six existing scenarios. `wait` returns the latest state when its polling budget expires; it does not turn a still-running job into a timeout. The independent worker enforces the configured execution wall time. A manager can be recreated with the same local root to inspect or collect a job after the caller exits.

Public methods are `submit`, `status`, `cancel`, `collect`, and a bounded `wait`. Local job IDs look like `job_<32 hexadecimal characters>`. A Slurm scheduler ID is recorded separately. Callers cannot use a scheduler ID to operate arbitrary jobs through this interface.

## Configure an actual cluster

The supplied `configs/hpc_slurm.json` is deliberately disabled and contains no invented host or account. An operator must provide the authorized SSH host, charging account, partition, existing absolute work directory, and absolute Python executable. That interpreter must already have the numerical dependencies. Resources are fixed by the operator profile: one node, one task, explicit CPUs, memory and wall time. This serial benchmark does not benefit from a multi-node allocation.

```python
from factor.hpc import HPCManager, SlurmConfig

profile = SlurmConfig(**json.loads(Path("configs/hpc_slurm.json").read_text()))
manager = HPCManager("runs/hpc-cluster", slurm=profile, max_jobs=2,
                     max_cases_per_scenario=10)
# This raises while the profile is disabled; enable only after site authorization.
job = manager.submit(experiment, backend="slurm", request_id="approved-cluster-check")
```

SSH uses batch mode and existing verified host keys. It does not bypass host-key checks, prompt for credentials, or install packages. Remote arguments are quoted using `shlex.join`; configuration and source bytes are supplied as data. The only batch script is a fixed invocation of the trusted worker. There is no user/model shell field.

Submission stages numerical Python sources, the worker, and the synthetic configuration in a fresh private job directory. Slurm does not move application inputs for the caller. The adapter requests a parsable job ID, and acceptance means queued/submitted rather than completed. [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html)

Status first checks the active queue and then accounting for terminal jobs. An unavailable scheduler or an empty response is reported as unknown/unavailable, never inferred to mean completion. Accounting must be enabled and accessible at the site for this workflow. Slurm polling is separated by at least five seconds in `wait`. [squeue](https://slurm.schedmd.com/squeue.html), [sacct](https://slurm.schedmd.com/sacct.html)

Cancellation invokes `scancel` only for the recorded numeric job ID. Acceptance is recorded as `cancel_requested`; a subsequent scheduler observation confirms cancellation. A network failure during cancellation becomes `cancel_unknown`. [Slurm scancel documentation](https://slurm.schedmd.com/scancel.html)

## Durable lifecycle and failure handling

Each job has an atomically updated `job.json`, an exact source snapshot, configuration, worker state, logs and, after collection, an artifact manifest. Normal states progress from preparation and submission to queued/running, then completed/failed/cancelled/timed_out. Status history is retained. Terminal execution does not establish scientific validity.

Local cancellation uses a per-job marker that the worker handles. The manager never signals a PID recovered from disk: operating-system PIDs can be reused. A stale heartbeat becomes `worker_unknown`; it is not evidence that the numerical process has stopped. Abrupt machine loss may require operator recovery.

A submission timeout is consequential: the remote scheduler may already have accepted the job. `submit` returns its durable manifest with `submit_unknown`, preserving the caller's ownership of a potentially accepted job. The same `request_id` returns the existing record and never resubmits it. Reusing the key with different configuration is rejected. An uncertain submission currently requires operator reconciliation using the recorded job name before further remote actions; automated orphan discovery is not implemented.

Limits include cumulative jobs in the store, cases per scenario, wall time, fixed Slurm resources, transport timeout, bounded network output, source size and artifact size/file count. Jobs with uncertain outcomes still consume the submission quota. These limits do not estimate scheduler charges. `cost_usd` is unknown; allocation/budget authorization must come from the operator. Do not mistake a CPU allocation or maximum output-token limit elsewhere for a financial ceiling.

The local worker does not enforce a kernel memory limit or sandbox trusted Python code. It limits thread counts and wall time; Slurm memory enforcement depends on site configuration. The manager assumes a single trusted owner of its private job store. It is not a multi-tenant service.

## Artifacts and provenance

The worker verifies the submitted numerical source hashes before execution and records its Python executable/version, platform and installed numerical dependency versions. The controller's source hash is also recorded. These identify the executed numerical source and environment; they do not validate the physics. Virtual-environment executable paths are preserved instead of resolving their symlinks to a different interpreter.

Collection copies allowed regular files into a new directory and verifies byte sizes and SHA-256 hashes. A completed job must also contain the expected workflow outputs, matching configuration/source provenance, and consistent worker/workflow completion states. Traversal paths, symbolic links, duplicates and oversized bundles are rejected. Collection is idempotent once its manifest exists. Interrupted collection directories remain inspectable. Source and remote results are not automatically deleted, and retention cleanup remains an operator task.

The synthetic run includes evaluator truth because this capability runs an evaluation job. Its output is not a hidden inference input. An agent evaluating a diagnostic method must preserve the separate evaluator boundary and cannot treat these public answers as a held-out test.

An execution failure can still have useful logs. Collecting those logs does not change the failure status. An LLM should not interpret a scheduler success alone as a successful diagnostic reconstruction or an accepted hypothesis.

## Verification and remaining work

`tests/test_hpc.py` includes actual local execution of 12 synthetic cases, restart and artifact collection, cancellation and wall timeout. Other tests use fixtures for Slurm submission/status/accounting/cancellation, loss of connectivity, uncertain submission, configuration policy, source hashes and artifact/path checks. No live Slurm claim follows from those tests.

Reproduce with the numerical Python environment:

```text
PYTHONPATH=src python -m unittest discover -s tests -p test_hpc.py -v
```

Before a first live cluster claim: supply and approve the site profile and resource budget; run the small smoke job; compare retrieved hashes and numerical results; check accounting, cancellation and the site's timeout behavior. Cluster-specific modules, containers, GPUs, MPI, automatic retries, cost pricing, remote cleanup and multiple schedulers remain unsupported.

The useful result is a concrete callable execution capability: local work runs today, and the Slurm path has an explicit configuration and verification boundary. Real cluster validation remains unfinished until an authorized cluster is available.
