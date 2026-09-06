import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from factor.artifacts import verify_run
from factor.contracts import Action, BudgetExceeded, ContractError, PolicyError, RuntimePolicy
from factor.motion import infer_motion
from factor.providers import ConventionalAgent, LocalModel, ModelReply, OpenAIModel, ScriptedAgent
from factor.runtime import ScientificRuntime


def request(**updates):
    observation = {"schema_version": "motion-observation/1", "observation_id": "test",
                   "apparent_displacement_m": -30e-9, "noise_sigma_m": 1e-9,
                   "drift_bound_m": None, "reference": None, "window_s": [30e-9, 60e-9],
                   "resolution_m": 5e-9, "source_kind": "synthetic_reconstructed", "calibration_id": "test"}
    observation.update(updates)
    return {"question": "Does the observation resolve material motion?", "observation": observation}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_real_conventional_loop_and_integrity(self):
        result = ScientificRuntime(ConventionalAgent(), root=self.temp.name).run(request())
        self.assertEqual(result["execution"], "completed")
        self.assertEqual(result["final"]["decision"], "ambiguous")
        self.assertIsNone(result["numerical_result"]["interval_m"])
        self.assertEqual(result["usage"]["tool_calls"], 2)
        self.assertEqual(verify_run(result["run_path"])["integrity"], "verified")
        file = next((Path(result["run_path"]) / "artifacts").glob("*.json"))
        file.write_text("{}")
        with self.assertRaises(ContractError):
            verify_run(result["run_path"])

    def test_truth_and_unrecognized_fields_rejected(self):
        data = request(truth_displacement_m=-30e-9)
        with self.assertRaises(ContractError):
            ScientificRuntime(ConventionalAgent(), root=self.temp.name).run(data)

    def test_equal_observations_do_not_reveal_latent_motion(self):
        a, b = request(), request()
        self.assertEqual(infer_motion(a["observation"]), infer_motion(b["observation"]))
        # Reference can resolve the ambiguity only with a justified mismatch bound.
        a["observation"]["reference"] = {"displacement_m": 0., "sigma_m": 1e-9,
                                         "noise_covariance_m2": 0., "residual_bound_m": 1e-9}
        self.assertEqual(infer_motion(a["observation"])["decision"], "compression_supported")
        a["observation"]["reference"]["residual_bound_m"] = None
        self.assertEqual(infer_motion(a["observation"])["decision"], "ambiguous")

    def test_forbidden_action_can_recover_without_executing(self):
        class Recovering(ConventionalAgent):
            def decide(self, context, tools, **kw):
                if not context["history"]:
                    return ModelReply(Action("shell", {"cmd": "anything"}), {"cost_usd": 0})
                self_record = context["history"][0]
                assert not self_record["ok"]
                return super().decide(context, tools, **kw)
        result = ScientificRuntime(Recovering(), root=self.temp.name).run(request())
        self.assertEqual(result["execution"], "completed")
        self.assertEqual(result["usage"]["tool_calls"], 3)

    def test_fabricated_evidence_does_not_complete(self):
        action = Action("finish", {"decision": "ambiguous", "evidence_ids": ["invented"],
                                   "explanation": "test", "next_step": "test"})
        policy = RuntimePolicy(allowed_providers=("fixture",), max_steps=1)
        result = ScientificRuntime(ScriptedAgent([action]), root=self.temp.name, policy=policy).run(request())
        self.assertEqual(result["execution"], "budget_exceeded")
        self.assertIsNone(result["final"])

    def test_request_cannot_expand_step_limit(self):
        policy = RuntimePolicy(max_steps=1)
        result = ScientificRuntime(ConventionalAgent(), root=self.temp.name, policy=policy).run({**request(), "max_steps": 100})
        self.assertEqual(result["execution"], "budget_exceeded")
        self.assertEqual(result["usage"]["model_calls"], 1)

    def test_cancel_before_first_action(self):
        result = ScientificRuntime(ConventionalAgent(), root=self.temp.name).run(request(), cancelled=lambda: True)
        self.assertEqual(result["execution"], "cancelled")
        self.assertEqual(result["usage"]["model_calls"], 0)

    def test_cloud_disabled_before_transport(self):
        calls = []
        agent = OpenAIModel("fixture", max_total_cost_usd=1, input_usd_per_million=1,
                            output_usd_per_million=1, transport=lambda payload: calls.append(payload))
        with self.assertRaises(PolicyError):
            ScientificRuntime(agent, root=self.temp.name).run(request())
        self.assertFalse(calls)

    def test_cloud_reservation_blocks_call_and_unknown_usage_retained(self):
        agent = OpenAIModel("fixture", max_total_cost_usd=0, input_usd_per_million=1,
                            output_usd_per_million=1, transport=lambda payload: None)
        with self.assertRaises(BudgetExceeded):
            agent.decide({}, [], max_output_tokens=100, timeout=1)
        agent.max_total_cost_usd = 1
        with self.assertRaises(Exception):
            agent.decide({}, [], max_output_tokens=100, timeout=1)
        self.assertGreater(agent.charged_or_reserved_usd, 0)

    def test_impossible_covariance_rejected(self):
        data = request(reference={"displacement_m": 0, "sigma_m": 1e-9,
                                  "residual_bound_m": 0, "noise_covariance_m2": 1})
        with self.assertRaises(ContractError):
            infer_motion(data["observation"])

    def test_native_usage_is_not_silently_zero(self):
        response = {"text": '{"tool":"inspect_observation","arguments":{}}',
                    "usage": {"input_tokens": 23, "total_output_tokens": 7},
                    "finish_reason": "stop", "model": "fixture"}
        with patch("factor.local_models.LocalChatClient.complete", return_value=response):
            reply = LocalModel("fixture").decide({}, [], max_output_tokens=100, timeout=1)
        self.assertEqual(reply.metadata["input_tokens"], 23)
        self.assertEqual(reply.metadata["output_tokens"], 7)

    def test_final_result_is_bound_to_event_chain(self):
        result = ScientificRuntime(ConventionalAgent(), root=self.temp.name).run(request())
        path = Path(result["run_path"])
        self.assertEqual(verify_run(path)["result_integrity"], "verified")
        saved = json.loads((path / "result.json").read_text())
        saved["final"]["decision"] = "compression_supported"
        (path / "result.json").write_text(json.dumps(saved))
        with self.assertRaises(ContractError):
            verify_run(path)

    def test_positive_cloud_budget_requires_actual_prices(self):
        with self.assertRaises(PolicyError):
            OpenAIModel("fixture", max_total_cost_usd=1, input_usd_per_million=0,
                        output_usd_per_million=0)

    def test_provider_list_is_not_a_string(self):
        with self.assertRaises(ContractError):
            RuntimePolicy(allowed_providers="local")

    def test_finishing_does_not_detach_unconfirmed_work(self):
        class EarlyFinish:
            provider, model = "fixture", "early-finish"
            def decide(self, context, tools, **kwargs):
                if not context["history"]:
                    action = Action("submit_simulation", {"backend": "local", "cases_per_scenario": 2, "seed": 1})
                else:
                    action = Action("finish", {"decision": "ambiguous", "evidence_ids": [context["observation_artifact_id"]],
                                               "explanation": "Unknown drift.", "next_step": "Measure drift."})
                return ModelReply(action, {"cost_usd": 0})
        class Manager:
            def submit(self, *args, **kwargs):
                return {"job_id": "job_"+"c"*32, "backend": "local", "status": "running"}
            def cancel(self, job_id):
                return {"status": "cancel_requested"}
        policy = RuntimePolicy(allowed_providers=("fixture",), allow_hpc=True,
                               allowed_tools=("submit_simulation", "finish"))
        result = ScientificRuntime(EarlyFinish(), root=self.temp.name, policy=policy, hpc=Manager()).run(request())
        self.assertEqual(result["execution"], "cleanup_pending")
        self.assertEqual(result["cleanup"][0]["status"], "cancel_requested")

    def test_unknown_job_cannot_be_polled(self):
        class NeverCalled:
            def wait(self, *args, **kwargs):
                raise AssertionError("A foreign job reached the manager")
        action = Action("job_status", {"job_id": "job_" + "a"*32})
        policy = RuntimePolicy(allowed_providers=("fixture",), allowed_tools=("job_status", "finish"),
                               max_steps=1, allow_hpc=True)
        result = ScientificRuntime(ScriptedAgent([action]), root=self.temp.name, policy=policy, hpc=NeverCalled()).run(request())
        events = json.loads((Path(result["run_path"]) / "events.json").read_text())
        self.assertIn("not owned", next(e for e in events if e["kind"] == "tool_result")["data"]["error"])


if __name__ == "__main__":
    unittest.main()
