import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from liner_stability import benchmarks, design, evidence, llm, physics, workflow
from liner_stability.io import read_json, write_json


def fixture_response(answer):
    return {"id":"fixture-response", "model":"fixture-model", "status":"completed",
            "output":[{"type":"message","content":[{"type":"output_text","text":json.dumps(answer)}]}],
            "usage":{"input_tokens":0,"output_tokens":0}}


class PhysicsAndDesign(unittest.TestCase):
    def test_pressure_scale_and_units(self):
        r=physics.magnetic_pressure(1e6,0.5e-3)
        self.assertAlmostEqual(r["field_T"],400)
        self.assertAlmostEqual(r["value"]/1e9,63.66197723675813)
        self.assertEqual(r["unit"],"Pa")
        self.assertAlmostEqual(physics.magnetic_pressure(2e6,0.5e-3)["value"]/r["value"],4)

    def test_invalid_physical_inputs(self):
        for radius in (0,-1,float('nan')):
            with self.assertRaises(ValueError):physics.magnetic_pressure(1,radius)
        with self.assertRaises(ValueError):physics.magnetic_pressure(True,1)

    def test_latent_heat_reference(self):
        r=physics.latent_heat_duration(2700,397000,4e-8,1e12)
        self.assertAlmostEqual(r["value"]/1e-9,26.7975)
        self.assertEqual(r["unit"],"s")

    def test_shared_clock_covariance(self):
        r=physics.event_delay(54,50,2,2,.75)
        self.assertEqual(r["value"],4)
        self.assertAlmostEqual(r["standard_deviation_ns"],math.sqrt(2))
        self.assertEqual(physics.event_delay(54,50,2,2,1)["standard_deviation_ns"],0)
        with self.assertRaises(ValueError):physics.event_delay(1,2,1,1,2)

    def test_design_rewards_usable_discrimination(self):
        config={"observables":[{"name":"amplitude","unit":"um"}],"candidates":[
            {"id":"large_but_unusable","mean_model_a":[0],"mean_model_b":[10],"covariance":[[1]],"usable_probability":.01,"relative_cost":1},
            {"id":"measurable","mean_model_a":[0],"mean_model_b":[3],"covariance":[[1]],"usable_probability":1,"relative_cost":1}]}
        self.assertEqual(design.rank_designs(config)["ranking"][0]["candidate_id"],"measurable")
        config["candidates"][0]["covariance"]=[[-1]]
        with self.assertRaises(ValueError):design.rank_designs(config)


class EvidenceChecks(unittest.TestCase):
    def test_heading_is_not_returned_as_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"notes.md"
            p.write_text("## What would distinguish optical drift?\n\nOptical drift changes phase-derived displacement and needs independent calibration.\n")
            hits=evidence.search(evidence.index_files([p]),"What would distinguish optical drift?")
            self.assertTrue(hits)
            self.assertFalse(hits[0]["text"].startswith("#"))
            self.assertIn("calibration",hits[0]["text"])

    def test_retrieval_and_versioned_ids(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"notes.md"
            p.write_text("Optical drift changes inferred displacement.\n\nMelt timing is model assisted.\n")
            index=evidence.index_files([p])
            hits=evidence.search(index,"optical drift")
            self.assertIn("Optical drift",hits[0]["text"])
            self.assertEqual(hits[0]["line_start"],1)
            self.assertEqual(evidence.search(index,"zebras"),[])
            old=index["chunks"][0]["evidence_id"]
            p.write_text(p.read_text()+"\nCalibration updated.\n")
            self.assertNotEqual(evidence.index_files([p])["chunks"][0]["evidence_id"],old)

    def test_unknown_citation_rejected(self):
        with self.assertRaises(ValueError):
            evidence.verify_references({"evidence_ids":["invented"]},[{"evidence_id":"real"}])
        check=evidence.verify_references({"evidence_ids":["real"]},[{"evidence_id":"real"}])
        self.assertFalse(check["claim_entailment_checked"])


class ProviderChecks(unittest.TestCase):
    def public_case(self):
        c=benchmarks.cases()[0]
        return {k:c[k] for k in ("case_id","kind","prompt","unit")}

    def test_key_is_not_a_candidate_input(self):
        with self.assertRaises(ValueError):llm.public_cases(benchmarks.cases())

    def test_request_schema_and_no_server_storage(self):
        case=self.public_case()
        p=llm.response_payload("chosen-model","instructions",case,llm.benchmark_schema(case),"answer")
        self.assertFalse(p["store"])
        self.assertTrue(p["text"]["format"]["strict"])
        self.assertNotIn("expected",json.loads(p["input"]))

    def test_completed_response(self):
        self.assertEqual(llm.extract_output(fixture_response({"answer":12}))["answer"],12)

    def test_refusal_and_incomplete_do_not_become_answers(self):
        for r in ({"status":"incomplete","output":[]},
                  {"status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"no"}]}]}):
            with self.assertRaises(llm.ModelError):llm.extract_output(r)

    def test_nonfinite_or_malformed_answer_rejected(self):
        with self.assertRaises(llm.ModelError):llm.extract_output(fixture_response({"answer":float('nan')}))
        with self.assertRaises(llm.ModelError):llm.extract_output(fixture_response([1,2]))

    def test_missing_key_prevents_network_request(self):
        with patch.dict(os.environ,{},clear=True):
            with patch("urllib.request.build_opener") as opener:
                with self.assertRaises(llm.ModelError):llm.openai_transport({})
                opener.assert_not_called()

    def test_candidate_run_keeps_failed_cases_in_outputs(self):
        case=self.public_case()
        def wrong_id(payload):
            return fixture_response({"case_id":"wrong","answer":0,"unit":"Pa","explanation":"fixture"})
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"run"
            result=llm.run_benchmark([case],out,"fixture",transport=wrong_id)
            self.assertIsNone(result[0]["answer"])
            self.assertEqual(result[0]["case_id"],case["case_id"])
            self.assertEqual(read_json(out/"manifest.json")["status"],"fixture_test_not_llm_evaluation")
            with self.assertRaises(ValueError):llm.run_benchmark([case],out,"fixture",transport=wrong_id)

    def test_candidate_correct_fixture_and_separate_grading(self):
        case=self.public_case()
        def response(payload):
            self.assertNotIn("rationale",json.loads(payload["input"]))
            return fixture_response({"case_id":case["case_id"],"answer":63.66197723675813e9,"unit":"Pa","explanation":"Analytic fixture result"})
        with tempfile.TemporaryDirectory() as td:
            predictions=llm.run_benchmark([case],Path(td)/"run","fixture",transport=response)
            grade=benchmarks.score_answers([benchmarks.cases()[0]],predictions)
            self.assertEqual(grade["n_correct"],1)

    def test_assistant_rejects_invented_evidence(self):
        index={"schema_version":"1","chunks":[{"evidence_id":"valid","text":"Optical drift affects displacement","source_sha256":"fixture","source_path":"fixture","line_start":1,"line_end":1}]}
        answer={"answer":"A claim","evidence_ids":["invented"],"assumptions":[],"unresolved":[],"next_measurement":None,"status":"provisional"}
        with self.assertRaises(llm.ModelError):
            llm.answer_question("optical drift",index,"fixture",provider="openai",transport=lambda _:fixture_response(answer))
        packet=llm.answer_question("optical drift",index)
        self.assertEqual(packet["status"],"evidence_packet_only_not_llm_answer")


class WorkflowIntegration(unittest.TestCase):
    def test_full_run_and_no_overwrite(self):
        config={"schema_version":"1","experiment_id":"integration","experiment_kind":"prescribed_motion_and_corrugated_cylinder","cases_per_scenario":2,"seed":1900}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"run"
            report=workflow.run(config,out)
            self.assertEqual(report["calibrated"]["all"]["n_cases"],12)
            self.assertEqual(read_json(out/"status.json")["status"],"completed")
            self.assertTrue((out/"EVALUATION_CARD.md").is_file())
            self.assertIn("llm.py",read_json(out/"run_manifest.json")["package_source_sha256"])
            with self.assertRaises(ValueError):workflow.run(config,out)

    def test_config_rejects_unknown_or_unsafe_counts(self):
        config={"schema_version":"1","experiment_id":"x","experiment_kind":"prescribed_motion_and_corrugated_cylinder","cases_per_scenario":-3,"seed":1}
        with self.assertRaises(ValueError):workflow.validate_config(config)
        config["cases_per_scenario"]=2
        config["solver"]="unimplemented"
        with self.assertRaises(ValueError):workflow.validate_config(config)

    def test_cli_works_outside_repository_directory(self):
        with tempfile.TemporaryDirectory() as td:
            conf=Path(td)/"input.json"
            out=Path(td)/"result.json"
            write_json(conf,{"calculation":"magnetic_pressure","parameters":{"current_A":1e6,"radius_m":.0005}})
            process=subprocess.run([sys.executable,"-m","liner_stability","physics","--config",str(conf),"--out",str(out)],cwd=td,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertEqual(read_json(out)["result"]["unit"],"Pa")


if __name__=="__main__":
    unittest.main(verbosity=2)
