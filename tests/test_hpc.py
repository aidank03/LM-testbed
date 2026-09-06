"""Real local numerical jobs; Slurm lifecycle and transport are fixtures only."""
import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

from factor.hpc import CommandResult, HPCError, HPCManager, SlurmConfig, TransportError
from factor.hpc_worker import bundle_artifacts


def experiment(n=2):
    return {"schema_version": "1", "experiment_id": "hpc-development-check",
            "experiment_kind": "prescribed_motion_and_corrugated_cylinder",
            "cases_per_scenario": n, "seed": 1701, "notes": "Synthetic development only"}


def profile():
    return SlurmConfig(enabled=True, host="researcher@cluster.example", account="research",
                       partition="debug", remote_workdir="/scratch/factor jobs",
                       python_executable="/opt/science env/bin/python", wall_minutes=2)


class FixtureTransport:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, host, argv, **kwargs):
        self.calls.append((host, argv, kwargs))
        if not self.replies:
            raise AssertionError("Unexpected external operation")
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


OK = CommandResult(0, b"staged\n")
SUBMITTED = CommandResult(0, b"8123\n")


class HPCFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def manager(self, replies, **kwargs):
        transport = FixtureTransport(replies)
        return HPCManager(self.tmp.name, slurm=profile(), transport=transport, **kwargs), transport

    def test_unconfigured_slurm_is_blocked_without_transport(self):
        transport = FixtureTransport([])
        manager = HPCManager(self.tmp.name, transport=transport)
        with self.assertRaises(HPCError):
            manager.submit(experiment(), backend="slurm")
        self.assertEqual(transport.calls, [])
        self.assertEqual(list(Path(self.tmp.name).glob("job_*")), [])

    def test_explicit_profile_rejects_shell_and_option_injection(self):
        for fields in ({"host": "-oProxyCommand=bad"}, {"host": "cluster;bad"},
                       {"account": "research\n--nodes=100"}, {"partition": "x,y"},
                       {"remote_workdir": "/scratch/../etc"}, {"python_executable": "python"},
                       {"cpus": 0}, {"wall_minutes": True}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(profile(), **fields).validate()

    def test_submit_poll_accounting_and_collect_fixture(self):
        raw = b'{"status":"completed"}'
        replies = [OK, SUBMITTED, CommandResult(0, b"8123|PENDING\n"),
                   CommandResult(0, b""), CommandResult(0, b"8123|COMPLETED|0:0\n")]
        manager, transport = self.manager(replies)
        submitted = manager.submit(experiment(), backend="slurm", request_id="fixture-1")
        content = {"result/status.json": raw, "worker_state.json": raw,
                   "result/metrics.json": b"{}", "experiment.json": json.dumps(experiment()).encode(),
                   "source_manifest.json": json.dumps(submitted["source_manifest"]).encode()}
        files = [{"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                  "data_b64": base64.b64encode(data).decode()} for name, data in content.items()]
        transport.replies.append(CommandResult(0, json.dumps({"files": files}).encode()))
        self.assertEqual(submitted["scheduler_id"], "8123")
        self.assertEqual(manager.status(submitted["job_id"])["status"], "queued")
        self.assertEqual(manager.status(submitted["job_id"])["status"], "completed")
        collected = manager.collect(submitted["job_id"])
        location = Path(collected["artifact_manifest"]["directory"]) / "result/status.json"
        self.assertEqual(location.read_bytes(), raw)
        self.assertEqual(manager.collect(submitted["job_id"]), collected)
        self.assertEqual(transport.replies, [])
        submit_call = transport.calls[1]
        self.assertIn("--account=research", submit_call[1])
        self.assertIn("--export=NONE", submit_call[1])
        command_line = submit_call[2]["stdin"].decode().splitlines()[1]
        tokens = shlex.split(command_line)
        self.assertEqual(tokens[0:2], ["exec", profile().python_executable])
        self.assertEqual(tokens[-1], submitted["remote_directory"])
        self.assertIn(" ", tokens[-1])  # Paths containing spaces remain single arguments.

    def test_uncertain_submission_is_durable_and_never_retried(self):
        manager, transport = self.manager([OK, TransportError("fixture timeout")])
        uncertain = manager.submit(experiment(), backend="slurm", request_id="one-request")
        self.assertEqual(uncertain["status"], "submit_unknown")
        restored = HPCManager(self.tmp.name, slurm=profile(), transport=transport)
        manifest = restored.submit(experiment(), backend="slurm", request_id="one-request")
        self.assertEqual(manifest["status"], "submit_unknown")
        self.assertEqual(len(transport.calls), 2)
        with self.assertRaises(HPCError):
            restored.cancel(manifest["job_id"])

    def test_missing_scheduler_state_is_not_completed(self):
        manager, _ = self.manager([OK, SUBMITTED, CommandResult(0), CommandResult(0)])
        job = manager.submit(experiment(), backend="slurm")
        self.assertEqual(manager.status(job["job_id"])["status"], "scheduler_unknown")

    def test_scheduler_timeout_preserves_unknown_execution(self):
        manager, _ = self.manager([OK, SUBMITTED, TransportError("offline")])
        job = manager.submit(experiment(), backend="slurm")
        self.assertEqual(manager.status(job["job_id"])["status"], "scheduler_unavailable")

    def test_scheduler_timeout_and_cancel_are_distinct(self):
        manager, transport = self.manager([OK, SUBMITTED, CommandResult(0),
                                          CommandResult(0, b"8123|CANCELLED by 42\n")])
        job = manager.submit(experiment(), backend="slurm")
        self.assertEqual(manager.cancel(job["job_id"])["status"], "cancel_requested")
        self.assertEqual(transport.calls[2][1], ["scancel", "8123"])
        self.assertEqual(manager.status(job["job_id"])["status"], "cancelled")

    def test_scheduler_terminal_failure_states_and_nonzero_exit(self):
        for state, exit_code, expected in (("TIMEOUT", "0:15", "timed_out"),
                                          ("OUT_OF_MEMORY", "0:9", "failed"),
                                          ("COMPLETED", "1:0", "failed")):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as root:
                transport = FixtureTransport([OK, SUBMITTED, CommandResult(0),
                                              CommandResult(0, f"8123|{state}|{exit_code}\n".encode())])
                manager = HPCManager(root, slurm=profile(), transport=transport)
                job = manager.submit(experiment(), backend="slurm")
                self.assertEqual(manager.status(job["job_id"])["status"], expected)

    def test_scheduler_success_does_not_bypass_required_artifacts(self):
        manager, _ = self.manager([OK, SUBMITTED, CommandResult(0),
                                   CommandResult(0, b"8123|COMPLETED|0:0\n"),
                                   CommandResult(0, b'{"files":[]}')])
        job = manager.submit(experiment(), backend="slurm")
        with self.assertRaisesRegex(HPCError, "missing required"):
            manager.collect(job["job_id"])

    def test_profile_change_cannot_retarget_existing_job(self):
        manager, transport = self.manager([OK, SUBMITTED])
        job = manager.submit(experiment(), backend="slurm")
        manager.slurm = replace(profile(), host="other.example")
        with self.assertRaises(HPCError):
            manager.status(job["job_id"])
        self.assertEqual(len(transport.calls), 2)

    def test_case_count_idempotency_and_job_limits(self):
        manager, transport = self.manager([OK, SUBMITTED], max_jobs=1, max_cases_per_scenario=2)
        with self.assertRaises(HPCError):
            manager.submit(experiment(3), backend="slurm")
        job = manager.submit(experiment(), backend="slurm", request_id="bounded")
        self.assertEqual(manager.submit(experiment(), backend="slurm", request_id="bounded")["job_id"], job["job_id"])
        altered = experiment()
        altered["seed"] = 99
        with self.assertRaises(HPCError):
            manager.submit(altered, backend="slurm", request_id="bounded")
        with self.assertRaises(HPCError):
            manager.submit(experiment(), backend="slurm")
        self.assertEqual(len(transport.calls), 2)

    def test_source_snapshot_hashes_match_staged_bytes(self):
        manager, transport = self.manager([OK, SUBMITTED])
        job = manager.submit(experiment(), backend="slurm")
        staged = json.loads(transport.calls[0][2]["stdin"])
        self.assertIn("worker.py", [f["path"] for f in staged["files"]])
        self.assertTrue(any(f["path"].endswith("simulation.py") for f in staged["files"]))
        for item in staged["files"]:
            raw = base64.b64decode(item["data_b64"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
        self.assertIsNotNone(job["source_manifest"]["manager_sha256"])

    def test_local_artifact_cap_and_symlinks(self):
        directory = Path(self.tmp.name)
        (directory / "result").mkdir()
        (directory / "result" / "large.bin").write_bytes(b"x" * 1025)
        with self.assertRaises(ValueError):
            bundle_artifacts(directory, 1024)
        (directory / "result" / "large.bin").unlink()
        (directory / "result" / "link").symlink_to("/etc/passwd")
        with self.assertRaises(ValueError):
            bundle_artifacts(directory, 1024)

    def test_remote_artifact_traversal_is_rejected(self):
        evil = {"path": "../escape", "size_bytes": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(), "data_b64": "eA=="}
        manager, _ = self.manager([OK, SUBMITTED, CommandResult(0),
                                   CommandResult(0, b"8123|COMPLETED|0:0\n"),
                                   CommandResult(0, json.dumps({"files": [evil]}).encode())])
        job = manager.submit(experiment(), backend="slurm")
        with self.assertRaises(HPCError):
            manager.collect(job["job_id"])
        self.assertFalse((Path(self.tmp.name) / "escape").exists())

    def test_unknown_job_ids_cannot_escape_store(self):
        manager = HPCManager(self.tmp.name)
        for job_id in ("../other", "--all", "8123", "job_" + "a" * 32):
            with self.subTest(job_id=job_id), self.assertRaises((HPCError, ValueError)):
                manager.cancel(job_id)

    def test_stale_worker_does_not_claim_failure_or_signal_pid(self):
        manager = HPCManager(self.tmp.name)
        with patch("factor.hpc.subprocess.Popen", return_value=Mock(pid=999999)):
            job = manager.submit(experiment())
        state = Path(self.tmp.name) / job["job_id"] / "worker_state.json"
        state.write_text(json.dumps({"status": "running", "heartbeat_at": time.time()-60}))
        self.assertEqual(manager.status(job["job_id"])["status"], "worker_unknown")


class HPCLocalExecutionTests(unittest.TestCase):
    """These execute actual numerical subprocesses; they do not use a cluster."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_actual_local_synthetic_run_and_collection_survive_manager_restart(self):
        manager = HPCManager(self.tmp.name, python_executable=sys.executable, max_cases_per_scenario=2)
        job = manager.submit(experiment(), request_id="local-real")
        restored = HPCManager(self.tmp.name, python_executable=sys.executable)
        finished = restored.wait(job["job_id"], timeout_seconds=60)
        self.assertEqual(finished["status"], "completed", finished)
        collected = restored.collect(job["job_id"])
        artifacts = Path(collected["artifact_manifest"]["directory"])
        metrics = json.loads((artifacts / "result/metrics.json").read_text())
        self.assertEqual(metrics["calibrated"]["all"]["n_cases"], 12)
        self.assertEqual(collected["execution_validation"], "local_process")
        self.assertIn("numpy", collected["worker"]["dependencies"])
        for item in collected["artifact_manifest"]["files"]:
            self.assertEqual(hashlib.sha256((artifacts / item["path"]).read_bytes()).hexdigest(), item["sha256"])

    def test_actual_local_cancellation(self):
        manager = HPCManager(self.tmp.name, python_executable=sys.executable)
        job = manager.submit(experiment(40))
        manager.cancel(job["job_id"])
        self.assertEqual(manager.wait(job["job_id"], timeout_seconds=15)["status"], "cancelled")

    def test_actual_local_wall_timeout(self):
        manager = HPCManager(self.tmp.name, python_executable=sys.executable, wall_seconds=1)
        job = manager.submit(experiment(40))
        self.assertEqual(manager.wait(job["job_id"], timeout_seconds=15)["status"], "timed_out")


if __name__ == "__main__":
    unittest.main()
