"""Independent ownership and budget checks; no live models or remote cluster."""
import json
from pathlib import Path
import tempfile
import time
import unittest

from factor.contracts import Action, RuntimePolicy
from factor.hpc import CommandResult, HPCManager, SlurmConfig, TransportError
from factor.providers import ModelReply, ScriptedAgent
from factor.runtime import ScientificRuntime


def request():
    return {"question": "Inspect this synthetic motion estimate.", "observation": {
        "schema_version": "motion-observation/1", "observation_id": "boundary-test",
        "apparent_displacement_m": 0, "noise_sigma_m": 1e-9, "drift_bound_m": None,
        "reference": None, "window_s": [30e-9, 60e-9], "resolution_m": 5e-9,
        "source_kind": "synthetic_reconstructed", "calibration_id": "synthetic"}}


def policy(**kwargs):
    values = {"allowed_providers": ("fixture",), "allow_hpc": True,
              "allowed_tools": ("inspect_observation", "submit_simulation", "job_status", "job_cancel", "job_collect", "finish"),
              "max_steps": 2}
    return RuntimePolicy(**(values | kwargs))


SUBMIT = Action("submit_simulation", {"backend": "local", "cases_per_scenario": 2, "seed": 1701})


class NeverCalled:
    def __getattr__(self, name):
        raise AssertionError(f"Unauthorized manager operation: {name}")


class RuntimeHPCBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_agent(self, actions, *, manager=None, operator=None):
        return ScientificRuntime(ScriptedAgent(actions), root=self.root / "agents",
                                 policy=operator or policy(), hpc=manager or NeverCalled()).run(request())

    def tool_errors(self, result):
        events = json.loads((Path(result["run_path"]) / "events.json").read_text())
        return [e["data"]["error"] for e in events if e["kind"] == "tool_result" and not e["data"]["ok"]]

    def test_foreign_job_cannot_be_cancelled_or_collected(self):
        foreign = "job_" + "a"*32
        result = self.run_agent([Action("job_cancel", {"job_id": foreign}), Action("job_collect", {"job_id": foreign})])
        self.assertEqual(len(self.tool_errors(result)), 2)
        self.assertTrue(all("not owned" in error for error in self.tool_errors(result)))
        self.assertEqual(result["jobs"], [])

    def test_disabled_hpc_and_remote_egress_prevent_submission(self):
        result = self.run_agent([SUBMIT], operator=policy(allow_hpc=False, max_steps=1))
        self.assertIn("disabled", self.tool_errors(result)[0])
        remote = Action("submit_simulation", {**SUBMIT.arguments, "backend": "slurm"})
        result = self.run_agent([remote], operator=policy(max_steps=1, data_egress=False))
        self.assertIn("egress", self.tool_errors(result)[0])

    def test_second_submission_is_blocked_and_owned_job_is_cleaned_up(self):
        class Manager:
            submitted, cancelled = [], []
            def submit(self, config, **kwargs):
                self.submitted.append(config)
                return {"job_id": "job_"+"b"*32, "backend": "local", "status": "submitted"}
            def cancel(self, job_id):
                self.cancelled.append(job_id)
                return {"status": "cancel_requested"}
        manager = Manager()
        result = self.run_agent([SUBMIT, SUBMIT], manager=manager)
        self.assertEqual(result["execution"], "budget_exceeded")
        self.assertEqual(len(manager.submitted), 1)
        self.assertEqual(manager.cancelled, result["jobs"])
        self.assertIn("one simulation", self.tool_errors(result)[0])

    def test_tool_budget_prevents_submission(self):
        result = self.run_agent([Action("inspect_observation", {}), SUBMIT],
                                operator=policy(max_tool_calls=1))
        self.assertEqual(result["execution"], "budget_exceeded")
        self.assertEqual(result["usage"]["tool_calls"], 1)
        self.assertEqual(result["jobs"], [])

    def test_expired_time_budget_prevents_submission_after_model_reply(self):
        class SlowFixture:
            provider, model = "fixture", "slow-fixture"
            def decide(self, *args, **kwargs):
                time.sleep(0.12)
                return ModelReply(SUBMIT, {"cost_usd": 0})
        result = ScientificRuntime(SlowFixture(), root=self.root / "agents", policy=policy(max_seconds=0.1),
                                   hpc=NeverCalled()).run(request())
        self.assertEqual(result["execution"], "budget_exceeded")
        self.assertEqual(result["usage"]["tool_calls"], 0)

    def test_uncertain_scheduler_submission_remains_owned(self):
        responses = iter([CommandResult(0, b"staged"), TransportError("fixture submission timeout")])
        def transport(*args, **kwargs):
            value = next(responses)
            if isinstance(value, Exception):
                raise value
            return value
        cluster = SlurmConfig(enabled=True, host="fixture.example", account="research", partition="debug",
                              remote_workdir="/scratch/factor", python_executable="/opt/python/bin/python")
        manager = HPCManager(self.root / "jobs", slurm=cluster, transport=transport)
        remote = Action("submit_simulation", {**SUBMIT.arguments, "backend": "slurm"})
        result = self.run_agent([remote], manager=manager, operator=policy(max_steps=1, data_egress=True))
        self.assertEqual(len(result["jobs"]), 1)
        self.assertEqual(manager.status(result["jobs"][0])["status"], "submit_unknown")
        self.assertEqual(result["cleanup"][0]["status"], "cleanup_uncertain")


if __name__ == "__main__":
    unittest.main()
