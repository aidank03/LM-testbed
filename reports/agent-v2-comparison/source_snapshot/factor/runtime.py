"""Bounded question → action → trusted tool → evidence → decision environment.

This is an application capability boundary, not an OS sandbox for arbitrary code.
Only operator-registered numerical tools execute; the model receives JSON data.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

from .artifacts import RunStore, canonical, now
from .contracts import BudgetExceeded, ContractError, PolicyError, RunRequest, RuntimePolicy, validate_schema
from .motion import DECISIONS, infer_motion, validate_observation
from .providers import ProviderError, SYSTEM


def obj(**properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def string(*, choices=None, maximum=2000):
    schema = {"type": "string", "minLength": 1, "maxLength": maximum}
    if choices is not None:
        schema["enum"] = list(choices)
    return schema


EVIDENCE = {"type": "array", "items": string(maximum=100), "maxItems": 20}
TOOLS = {
    "inspect_observation": ("Inspect the supplied reconstructed observation; no additional measurement.", obj()),
    "infer_motion": ("Compute a conditional fixed-window interval using available calibration and drift bounds.",
                     obj(method=string(choices=("naive", "bounded")))),
    "submit_simulation": ("Submit ONE small trusted synthetic development benchmark, not a facility experiment. It is a prescribed-motion model, not an ETI/MRTI solver.",
                          obj(backend=string(choices=("local", "slurm")),
                              cases_per_scenario={"type": "integer", "minimum": 2, "maximum": 10},
                              seed={"type": "integer", "minimum": 0, "maximum": 2**31-1})),
    "job_status": ("Wait briefly for and inspect a job owned by this run.", obj(job_id=string(maximum=100))),
    "job_collect": ("Collect a completed owned job's verified synthetic diagnostic evaluation summary.", obj(job_id=string(maximum=100))),
    "job_cancel": ("Cancel a job owned by this run.", obj(job_id=string(maximum=100))),
    "update_hypothesis": ("Record a conditional hypothesis update with existing evidence; never establishes physical causality.",
                          obj(hypothesis_id=string(maximum=100), statement=string(),
                              state=string(choices=("unresolved", "consistent_with_synthetic_model", "contradicted_within_synthetic_model")),
                              evidence_ids=EVIDENCE)),
    "finish": ("Return the scientific decision, evidence IDs, limits and next measurement. Execution failures are not abstentions.",
               obj(decision=string(choices=DECISIONS), evidence_ids=EVIDENCE,
                   explanation=string(maximum=4000), next_step=string(maximum=2000))),
}


class ScientificRuntime:
    def __init__(self, agent, *, root="runs/agents", policy=None, hpc=None):
        self.agent, self.root = agent, Path(root)
        self.policy = policy or RuntimePolicy()
        self.hpc = hpc
        if set(self.policy.allowed_tools) - TOOLS.keys():
            raise ContractError("Policy contains an unregistered tool")
        if "finish" not in self.policy.allowed_tools:
            raise ContractError("Policy must allow finish")

    def _authorize(self):
        if self.agent.provider not in self.policy.allowed_providers:
            raise PolicyError("Provider is not allowed by the operator policy")
        if self.agent.provider == "openai":
            if not self.policy.data_egress or self.policy.max_cloud_cost_usd <= 0:
                raise PolicyError("Cloud execution requires explicit data egress and a positive budget")
            if self.agent.max_total_cost_usd > self.policy.max_cloud_cost_usd:
                raise PolicyError("Provider budget exceeds the operator policy")

    def run(self, request, *, cancelled=None):
        if isinstance(request, dict):
            try:
                request = RunRequest(**request)
            except TypeError:
                raise ContractError("Request has missing or unknown fields") from None
        if not isinstance(request, RunRequest):
            raise ContractError("Expected RunRequest or its JSON object")
        request.validate()
        validate_observation(request.observation)
        self._authorize()
        if len(canonical(asdict(request))) > self.policy.max_payload_bytes:
            raise PolicyError("Input exceeds the operator payload limit")
        store = RunStore(self.root)
        started = time.monotonic()
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(Path(__file__).parent.glob("*.py"))}
        manifest = {"schema_version": "factor-run/1", "created_at": now(),
                    "provider": self.agent.provider, "model": self.agent.model,
                    "python": platform.python_version(), "platform": platform.platform(),
                    "policy": asdict(self.policy), "source_sha256": source_hashes,
                    "system_prompt": SYSTEM if self.agent.provider in ("local", "openai") else None,
                    "tool_schemas": {name: TOOLS[name][1] for name in self.policy.allowed_tools},
                    "controller_settings": {name: getattr(self.agent, name) for name in
                                            ("temperature", "base_url") if hasattr(self.agent, name)},
                    "science_status": "synthetic_development_unless_independently_validated"}
        store.artifact("run_manifest", manifest)
        observation_id = store.artifact("observation", request.observation)
        store.artifact("request", asdict(request))
        store.event("started", {"observation_artifact_id": observation_id})
        tools = [{"name": name, "description": TOOLS[name][0], "parameters": TOOLS[name][1]}
                 for name in self.policy.allowed_tools]
        history, jobs, hypotheses = [], set(), []
        usage = {"model_calls": 0, "tool_calls": 0, "cloud_cost_usd": 0.0,
                 "cloud_reserved_unknown_usd": 0.0, "accounting_complete": True,
                 "input_tokens": 0, "output_tokens": 0}
        numerical, final, failure = None, None, None
        execution = "budget_exceeded"
        try:
            for step in range(min(request.max_steps or self.policy.max_steps, self.policy.max_steps)):
                remaining = self.policy.max_seconds - (time.monotonic() - started)
                if cancelled and cancelled():
                    execution = "cancelled"
                    break
                if remaining <= 0:
                    raise BudgetExceeded("Run time limit reached")
                context = {"question": request.question, "observation": request.observation,
                           "observation_artifact_id": observation_id, "history": history,
                           "steps_remaining": min(request.max_steps or self.policy.max_steps, self.policy.max_steps)-step}
                if len(canonical({"context": context, "tools": tools})) > self.policy.max_payload_bytes:
                    raise BudgetExceeded("Public context exceeded its payload limit")
                usage["model_calls"] += 1
                try:
                    reply = self.agent.decide(context, tools, max_output_tokens=self.policy.max_output_tokens,
                                              timeout=max(0.01, remaining))
                except ProviderError as exc:
                    self._account(usage, exc.metadata)
                    store.event("provider_error", {"error": str(exc), "metadata": exc.metadata})
                    raise
                self._account(usage, reply.metadata)
                action = reply.action
                store.event("model_action", {"tool": action.tool, "arguments": action.arguments,
                                              "metadata": reply.metadata})
                if time.monotonic() - started >= self.policy.max_seconds:
                    raise BudgetExceeded("Run time limit reached after model response; no tool executed")
                if cancelled and cancelled():
                    execution = "cancelled"
                    break
                if usage["tool_calls"] >= self.policy.max_tool_calls:
                    raise BudgetExceeded("Tool call limit reached")
                usage["tool_calls"] += 1
                try:
                    if action.tool not in self.policy.allowed_tools:
                        raise PolicyError("Action is not in the allowed tool registry")
                    validate_schema(action.arguments, TOOLS[action.tool][1])
                    args = action.arguments
                    if action.tool == "finish":
                        self._evidence(args["evidence_ids"], store)
                        final = args
                        execution = "completed"
                        store.event("decision", final)
                        break
                    if action.tool == "inspect_observation":
                        value = request.observation
                    elif action.tool == "infer_motion":
                        value = numerical = infer_motion(request.observation, args["method"])
                    elif action.tool == "update_hypothesis":
                        self._evidence(args["evidence_ids"], store)
                        value = {**args, "scope": "conditional_synthetic_development", "version": len(hypotheses)+1}
                        hypotheses.append(value)
                    else:
                        value = self._job(action.tool, args, store, jobs,
                                          self.policy.max_seconds-(time.monotonic()-started))
                    if len(canonical(value)) > self.policy.max_payload_bytes // 2:
                        raise ContractError("Tool result exceeded its size limit")
                    artifact_id = store.artifact(action.tool, value)
                    record = {"tool": action.tool, "ok": True, "artifact_id": artifact_id, "result": value}
                except (ContractError, PolicyError, ValueError) as exc:
                    record = {"tool": action.tool, "ok": False, "error": str(exc)}
                history.append(record)
                store.event("tool_result", record)
        except BudgetExceeded as exc:
            execution, failure = "budget_exceeded", str(exc)
        except ProviderError as exc:
            execution, failure = "provider_failed", str(exc)
        except KeyboardInterrupt:
            execution, failure = "cancelled", "Interrupted by operator"
        except Exception as exc:
            # Keep internal paths/provider bodies out of model context and errors.
            execution, failure = "failed", f"Unexpected execution error: {type(exc).__name__}"
        cleanup = []
        # No implicit detached jobs, including a model that finishes too early.
        for job_id in sorted(jobs):
            try:
                cleanup.append({"job_id": job_id, "status": self.hpc.cancel(job_id)["status"]})
            except Exception as exc:
                cleanup.append({"job_id": job_id, "status": "cleanup_uncertain", "error_type": type(exc).__name__})
        if execution == "completed" and cleanup:
            from .hpc_worker import TERMINAL
            if any(item["status"] not in TERMINAL for item in cleanup):
                execution, failure = "cleanup_pending", "Decision returned, but owned job cleanup is not confirmed; inspect the durable job record."
        usage["elapsed_seconds"] = time.monotonic() - started
        store.event("ended", {"execution": execution, "failure": failure, "usage": usage, "cleanup": cleanup})
        return store.result({"schema_version": "factor-result/1", "run_id": store.run_id,
                             "run_path": str(store.path), "execution": execution, "failure": failure,
                             "final": final, "numerical_result": numerical, "usage": usage,
                             "jobs": sorted(jobs), "hypotheses": hypotheses, "cleanup": cleanup,
                             "artifacts": store.artifacts})

    @staticmethod
    def _account(usage, metadata):
        cost = metadata.get("cost_usd")
        if cost is None:
            usage["accounting_complete"] = False
            usage["cloud_reserved_unknown_usd"] += metadata.get("reserved_usd", 0)
        else:
            usage["cloud_cost_usd"] += cost
        for name in ("input_tokens", "output_tokens"):
            if isinstance(metadata.get(name), int):
                usage[name] += metadata[name]

    @staticmethod
    def _evidence(ids, store):
        if not ids or any(identity not in store.artifacts for identity in ids):
            raise ContractError("Cite at least one artifact that actually exists in this run")

    def _job(self, tool, args, store, jobs, remaining):
        if not self.policy.allow_hpc or self.hpc is None:
            raise PolicyError("Job execution is disabled by the operator policy")
        from .hpc import HPCError
        try:
            if tool == "submit_simulation":
                if jobs:
                    raise PolicyError("Only one simulation submission is permitted per run")
                if args["backend"] == "slurm" and not self.policy.data_egress:
                    raise PolicyError("Remote jobs require explicit data egress authorization")
                config = {"schema_version": "1", "experiment_id": store.run_id,
                          "experiment_kind": "prescribed_motion_and_corrugated_cylinder",
                          "cases_per_scenario": args["cases_per_scenario"], "seed": args["seed"],
                          "notes": "Synthetic diagnostic development; no physical mechanism validation."}
                manifest = self.hpc.submit(config, backend=args["backend"], request_id=store.run_id)
                jobs.add(manifest["job_id"])
                return {k: manifest[k] for k in ("job_id", "backend", "status")}
            job_id = args["job_id"]
            if job_id not in jobs:
                raise PolicyError("Job is not owned by this run")
            if tool == "job_status":
                manifest = self.hpc.wait(job_id, timeout_seconds=min(2, max(0.01, remaining)))
                return {k: manifest[k] for k in ("job_id", "backend", "status")}
            if tool == "job_cancel":
                manifest = self.hpc.cancel(job_id)
                return {"job_id": job_id, "status": manifest["status"]}
            manifest = self.hpc.collect(job_id)
            collection = manifest["artifact_manifest"]
            metrics_path = Path(collection["directory"]) / "result/metrics.json"
            if not metrics_path.is_file():
                return {"job_id": job_id, "status": manifest["status"], "diagnostics_available": False}
            metrics = json.loads(metrics_path.read_text())
            summary = {name: {"n_cases": row["n_cases"], "compression_nm": row["compression_nm"]}
                       for name, row in metrics["calibrated"].items() if name != "all"}
            return {"job_id": job_id, "status": manifest["status"], "diagnostics_available": True,
                    "scope": "Independent synthetic development job, not evidence of motion in the supplied observation. Latent truth used by this job's evaluator is only truth within its simulator.",
                    "summary": summary, "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                    "artifact_count": len(collection["files"])}
        except HPCError as exc:
            raise ContractError(str(exc)) from None
