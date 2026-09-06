"""Durable local/Slurm execution of one trusted synthetic benchmark template.

Slurm is opt-in operator configuration, never a hostname supplied by a model.
Only regular numerical source/configuration files are staged. No facility access,
package installation, arbitrary commands, MPI, or automatic submission retries.
"""
import base64
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .hpc_worker import TERMINAL, atomic_json, bundle_artifacts


class HPCError(RuntimeError):
    """An explicit lifecycle, policy, transport or artifact error."""


class TransportError(HPCError):
    """Outcome may be unknown after a submission transport failure."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _duration(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0.01 <= value <= 60):
        raise ValueError("timeout_seconds must be finite and in [0.01, 60]")


def _remote_path(value, name):
    if (not isinstance(value, str) or not value.startswith("/") or value == "/"
            or any(ord(c) < 32 for c in value) or "%" in value or ".." in PurePosixPath(value).parts
            or str(PurePosixPath(value)) != value):
        raise ValueError(f"{name} must be a normalized absolute non-root POSIX path")
    return value


@dataclass(frozen=True)
class SlurmConfig:
    enabled: bool = False
    host: str = ""
    account: str = ""
    partition: str = ""
    remote_workdir: str = ""
    python_executable: str = ""
    cpus: int = 1
    memory_mb: int = 2048
    wall_minutes: int = 10

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not self.enabled:
            raise HPCError("Slurm is disabled; explicit cluster configuration is required")
        if not re.fullmatch(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9.-]*", self.host):
            raise ValueError("Invalid SSH host; use one explicit host or user@host")
        for name in ("account", "partition"):
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", getattr(self, name)):
                raise ValueError(f"Invalid or missing Slurm {name}")
        _remote_path(self.remote_workdir, "remote_workdir")
        _remote_path(self.python_executable, "python_executable")
        _integer(self.cpus, "cpus", 1, 64)
        _integer(self.memory_mb, "memory_mb", 128, 262144)
        _integer(self.wall_minutes, "wall_minutes", 1, 1440)


class SSHTransport:
    """Bounded SSH subprocess; remote argv is quoted once using shlex.join."""

    def __call__(self, host, argv, *, stdin=b"", timeout=30, max_output_bytes=1_000_000):
        command = ["ssh", "-T", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes",
                   "-oConnectTimeout=10", host, shlex.join(argv)]
        # A regular input file avoids a blocked writer when a remote process fails.
        with tempfile.TemporaryFile() as source:
            source.write(stdin)
            source.seek(0)
            process = subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            buffers = {"stdout": bytearray(), "stderr": bytearray()}
            deadline = time.monotonic() + timeout
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TransportError("SSH command timed out; remote outcome may be unknown")
                        for key, _ in selector.select(min(remaining, 0.2)):
                            data = os.read(key.fd, 65536)
                            if not data:
                                selector.unregister(key.fileobj)
                            else:
                                buffers[key.data].extend(data)
                                if sum(map(len, buffers.values())) > max_output_bytes:
                                    raise TransportError("SSH output exceeded its size limit")
                process.wait(timeout=max(0.01, deadline-time.monotonic()))
            except BaseException as exc:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise TransportError("SSH command timed out; remote outcome may be unknown") from None
                raise
            finally:
                process.stdout.close()
                process.stderr.close()
        return CommandResult(process.returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"]))


# Fixed staging program. Its data arrives on stdin, never interpolated into code.
_STAGE = r'''
import base64, hashlib, json, pathlib, sys
data = json.load(sys.stdin)
root = pathlib.Path(sys.argv[1])
if not root.is_dir() or root.is_symlink():
    raise ValueError("Approved work directory must already exist and not be a symlink")
target = root / sys.argv[2]
target.mkdir(mode=0o700, exist_ok=False)
for item in data["files"]:
    relative = pathlib.PurePosixPath(item["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid staged path")
    path = target / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(item["data_b64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("Staging hash mismatch")
    with path.open("xb") as handle:
        handle.write(raw)
print("staged")
'''


class HPCManager:
    """Single-owner durable manager; job IDs resolve only under local_root.

    transport is an injectable fixture boundary, not an agent-settable capability.
    max_jobs caps cumulative submissions in this root, including uncertain ones.
    Resource ceilings belong to operator configuration, not job requests.
    """

    def __init__(self, local_root, *, source_root=None, python_executable=sys.executable,
                 slurm=None, transport=None, max_jobs=4, max_cases_per_scenario=40,
                 wall_seconds=300, artifact_max_bytes=50_000_000):
        self.root = Path(local_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.source_root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
        # Preserve a virtual environment's executable symlink: resolving it would
        # silently select the base interpreter and lose its numerical packages.
        self.python_executable = os.path.abspath(python_executable)
        if not Path(self.python_executable).is_file() or not os.access(self.python_executable, os.X_OK):
            raise ValueError("python_executable must name an existing executable")
        self.slurm = slurm or SlurmConfig()
        self.transport = transport or SSHTransport()
        for value, name, low, high in ((max_jobs, "max_jobs", 1, 100),
                                      (max_cases_per_scenario, "max_cases_per_scenario", 2, 1000),
                                      (wall_seconds, "wall_seconds", 1, 86400),
                                      (artifact_max_bytes, "artifact_max_bytes", 1024, 200_000_000)):
            _integer(value, name, low, high)
        self.max_jobs = max_jobs
        self.max_cases_per_scenario = max_cases_per_scenario
        self.wall_seconds = wall_seconds
        self.artifact_max_bytes = artifact_max_bytes

    @contextmanager
    def _lock(self):
        with (self.root / ".manager.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _directory(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"job_[0-9a-f]{32}", job_id):
            raise ValueError("Invalid Factor job ID")
        directory = self.root / job_id
        if directory.is_symlink() or directory.resolve().parent != self.root or not directory.is_dir():
            raise HPCError("Unknown or invalid job directory")
        return directory

    def _read(self, job_id):
        return json.loads((self._directory(job_id) / "job.json").read_text())

    def _save(self, manifest, status=None, **updates):
        if status is not None and status != manifest["status"]:
            manifest["history"].append({"status": status, "at": time.time()})
            manifest["status"] = status
        manifest.update(updates, updated_at=time.time())
        atomic_json(self._directory(manifest["job_id"]) / "job.json", manifest)
        return manifest

    def _remote(self, manifest, argv, *, stdin=b"", limit=1_000_000, timeout=30):
        # Do not allow a persisted job to silently move to a newly configured host.
        self.slurm.validate()
        if manifest["slurm"] != asdict(self.slurm):
            raise HPCError("Current Slurm configuration differs from the submitted job")
        return self.transport(self.slurm.host, argv, stdin=stdin, timeout=timeout,
                              max_output_bytes=limit)

    def _snapshot(self, directory):
        paths = sorted((self.source_root / "liner_stability").rglob("*.py"))
        if not paths or not (self.source_root / "liner_stability" / "__main__.py").is_file():
            raise HPCError("The installed liner_stability numerical source is required")
        files, total = [], 0
        for path in paths + [Path(__file__).with_name("hpc_worker.py")]:
            if path.is_symlink():
                raise HPCError("Symbolic source files are prohibited")
            raw = path.read_bytes()
            total += len(raw)
            if total > 10_000_000 or len(files) >= 500:
                raise HPCError("Source snapshot exceeds its size limit")
            relative = ("worker.py" if path.name == "hpc_worker.py" else
                        "source/" + path.relative_to(self.source_root).as_posix())
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(raw)
            files.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                          "size_bytes": len(raw)})
        manifest = {"files": files, "manager_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        atomic_json(directory / "source_manifest.json", manifest)
        return manifest

    def submit(self, experiment_config, *, backend="local", request_id=None):
        from liner_stability.workflow import validate_config
        config = dict(experiment_config)
        validate_config(config)
        if config["cases_per_scenario"] > self.max_cases_per_scenario:
            raise HPCError("Requested case count exceeds operator policy")
        if backend not in {"local", "slurm"}:
            raise ValueError("backend must be local or slurm")
        if backend == "slurm":
            self.slurm.validate()
        if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", request_id)):
            raise ValueError("request_id must be a short identifier")
        encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode()
        if len(encoded) > 100_000:
            raise HPCError("Experiment configuration is too large")
        digest = hashlib.sha256(encoded).hexdigest()
        with self._lock():
            existing = sorted(self.root.glob("job_*/job.json"))
            for path in existing:
                prior = json.loads(path.read_text())
                if request_id is not None and prior["request_id"] == request_id:
                    if prior["config_sha256"] != digest or prior["backend"] != backend:
                        raise HPCError("Idempotency key already names a different request")
                    return prior  # Never re-dispatch a request with an uncertain outcome.
            if len(existing) >= self.max_jobs:
                raise HPCError("Cumulative job limit reached for this run store")
            job_id = "job_" + uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir(mode=0o700)
            manifest = {"schema_version": "1", "job_id": job_id, "request_id": request_id,
                        "backend": backend, "status": "preparing", "config_sha256": digest,
                        "created_at": time.time(), "history": [], "scheduler_id": None,
                        "science_status": "synthetic_development_only",
                        "execution_validation": "local_process" if backend == "local" else "slurm_not_site_validated",
                        "artifact_limit_bytes": self.artifact_max_bytes,
                        "slurm": asdict(self.slurm) if backend == "slurm" else None,
                        "cost_usd": None, "automatic_retry": False}
            self._save(manifest)
            try:
                atomic_json(directory / "experiment.json", config)
                source = self._snapshot(directory)
                atomic_json(directory / "worker_config.json", {
                    "wall_seconds": self.wall_seconds if backend == "local" else self.slurm.wall_minutes*60,
                    "cpus": 1 if backend == "local" else self.slurm.cpus})
                self._save(manifest, source_manifest=source)
                if backend == "local":
                    self._save(manifest, "starting")
                    with (directory / "supervisor.log").open("xb") as log:
                        env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
                        env["PYTHONNOUSERSITE"] = "1"
                        process = subprocess.Popen([self.python_executable, str(directory / "worker.py"),
                                                    "run", str(directory)], stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=log, env=env, start_new_session=True)
                    threading.Thread(target=process.wait, daemon=True).start()
                    self._save(manifest, "submitted", worker_pid=process.pid,
                               wall_seconds=self.wall_seconds)
                else:
                    self._submit_slurm(manifest, directory)
            except (OSError, ValueError, HPCError) as exc:
                if manifest["status"] not in {"submitting", "submit_unknown"}:
                    self._save(manifest, "failed", error_type=type(exc).__name__)
                else:
                    # Preserve ownership in callers even when acceptance is unknown.
                    # Raising here could leave an accepted external job untracked.
                    return self._save(manifest, "submit_unknown", error_type=type(exc).__name__)
                raise
            return manifest

    def _submit_slurm(self, manifest, directory):
        remote = self.slurm.remote_workdir + "/" + manifest["job_id"]
        self._save(manifest, remote_directory=remote)
        files = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "job.json":
                raw = path.read_bytes()
                files.append({"path": path.relative_to(directory).as_posix(),
                              "sha256": hashlib.sha256(raw).hexdigest(),
                              "data_b64": base64.b64encode(raw).decode("ascii")})
        staged = self._remote(manifest, [self.slurm.python_executable, "-c", _STAGE,
                                       self.slurm.remote_workdir, manifest["job_id"]],
                              stdin=json.dumps({"files": files}).encode())
        if staged.returncode != 0:
            raise HPCError("Remote source staging failed; no sbatch request was made")
        argv = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1",
                "--account=" + self.slurm.account, "--partition=" + self.slurm.partition,
                "--cpus-per-task=" + str(self.slurm.cpus), "--mem=" + str(self.slurm.memory_mb) + "M",
                "--time=" + str(self.slurm.wall_minutes), "--export=NONE",
                "--job-name=" + manifest["job_id"], "--chdir=" + remote,
                "--output=" + remote + "/scheduler.log"]
        script = "#!/bin/sh\nexec " + shlex.join([self.slurm.python_executable,
                                                  remote + "/worker.py", "run", remote]) + "\n"
        self._save(manifest, "submitting", submission_argv=argv,
                   submission_script_sha256=hashlib.sha256(script.encode()).hexdigest())
        result = self._remote(manifest, argv, stdin=script.encode())
        match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", result.stdout.decode().strip())
        if result.returncode != 0 or not match:
            self._save(manifest, "submit_unknown")
            raise HPCError("Submission outcome is uncertain; inspect scheduler by recorded job name, do not retry")
        self._save(manifest, "submitted", scheduler_id=match.group(1))

    def status(self, job_id, *, timeout_seconds=30):
        _duration(timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        with self._lock():
            manifest = self._read(job_id)
            if manifest["status"] in TERMINAL or manifest["status"] in {"preparing", "submit_unknown", "submitting"}:
                return manifest
            if manifest["backend"] == "local":
                path = self._directory(job_id) / "worker_state.json"
                if path.exists():
                    worker = json.loads(path.read_text())
                    status = worker["status"]
                    if status not in TERMINAL and time.time() - worker["heartbeat_at"] > 15:
                        status = "worker_unknown"
                    return self._save(manifest, status, worker=worker)
                if time.time() - manifest["created_at"] > 15:
                    return self._save(manifest, "worker_unknown")
                return manifest
            scheduler_id = manifest["scheduler_id"]
            if not isinstance(scheduler_id, str) or not re.fullmatch(r"[1-9][0-9]*", scheduler_id):
                raise HPCError("Missing or invalid scheduler ID")
            try:
                result = self._remote(manifest, ["squeue", "--noheader", "--states=all",
                                               "--jobs="+scheduler_id, "--format=%i|%T"],
                                      timeout=max(0.05, deadline-time.monotonic()))
                if result.returncode != 0:
                    return self._save(manifest, "scheduler_unavailable")
                rows = [r.split("|") for r in result.stdout.decode().splitlines() if r.strip()]
                state = next((r[1].strip() for r in rows if len(r) == 2 and r[0].strip() == scheduler_id), None)
                exit_code = None
                if state is None:
                    result = self._remote(manifest, ["sacct", "--noheader", "--parsable2", "--allocations",
                                                   "--jobs="+scheduler_id, "--format=JobIDRaw,State%40,ExitCode"],
                                          timeout=max(0.05, deadline-time.monotonic()))
                    if result.returncode != 0:
                        return self._save(manifest, "scheduler_unavailable")
                    for row in result.stdout.decode().splitlines():
                        values = row.split("|")
                        if len(values) >= 3 and values[0].strip() == scheduler_id:
                            state, exit_code = values[1].strip(), values[2].strip()
                            break
                if state is None:
                    return self._save(manifest, "scheduler_unknown")
                normalized = state.split()[0].rstrip("+")
                states = {"PENDING": "queued", "CONFIGURING": "queued", "RUNNING": "running",
                          "COMPLETING": "running", "SUSPENDED": "suspended", "COMPLETED": "completed",
                          "CANCELLED": "cancelled", "TIMEOUT": "timed_out", "FAILED": "failed",
                          "NODE_FAIL": "failed", "OUT_OF_MEMORY": "failed", "PREEMPTED": "failed",
                          "BOOT_FAIL": "failed", "DEADLINE": "timed_out", "REVOKED": "failed"}
                resolved = states.get(normalized, "scheduler_unknown")
                if resolved == "completed" and exit_code not in {None, "0:0"}:
                    resolved = "failed"
                return self._save(manifest, resolved, scheduler_state=state, scheduler_exit_code=exit_code)
            except (TransportError, OSError, UnicodeError) as exc:
                return self._save(manifest, "scheduler_unavailable", error_type=type(exc).__name__)

    def cancel(self, job_id):
        with self._lock():
            manifest = self._read(job_id)
            if manifest["status"] in TERMINAL:
                return manifest
            if manifest["backend"] == "local":
                marker = self._directory(job_id) / "cancel_requested"
                if not marker.exists():
                    marker.touch(exist_ok=False)
            else:
                sid = manifest["scheduler_id"]
                if not isinstance(sid, str) or not re.fullmatch(r"[1-9][0-9]*", sid):
                    raise HPCError("Submission has no confirmed job ID; reconcile it before cancellation")
                try:
                    result = self._remote(manifest, ["scancel", sid])
                    if result.returncode != 0:
                        return self._save(manifest, "cancel_unknown")
                except (TransportError, OSError):
                    return self._save(manifest, "cancel_unknown")
            return self._save(manifest, "cancel_requested")

    def wait(self, job_id, *, timeout_seconds=30):
        """Bounded wait; Slurm queries are at least five seconds apart."""
        _duration(timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        manifest = self._read(job_id)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return manifest
            manifest = self.status(job_id, timeout_seconds=max(0.01, min(30, remaining)))
            if manifest["status"] in TERMINAL or manifest["status"] in {"submit_unknown", "worker_unknown", "scheduler_unavailable"}:
                return manifest
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return manifest
            time.sleep(min(remaining, 5 if manifest["backend"] == "slurm" else 0.2))

    def collect(self, job_id):
        manifest = self.status(job_id)
        if manifest["status"] not in TERMINAL:
            raise HPCError("Artifacts require a confirmed terminal execution state")
        with self._lock():
            manifest = self._read(job_id)
            if manifest.get("artifact_manifest"):
                return manifest
            directory = self._directory(job_id)
            cap = min(self.artifact_max_bytes, manifest["artifact_limit_bytes"])
            if manifest["backend"] == "local":
                bundle = bundle_artifacts(directory, cap)
            else:
                remote = manifest["remote_directory"]
                result = self._remote(manifest, [self.slurm.python_executable, remote+"/worker.py",
                                               "collect", remote, str(cap)], limit=2*cap+1_000_000)
                if result.returncode != 0:
                    raise HPCError("Remote artifact collection failed or exceeded its limits")
                bundle = json.loads(result.stdout)
            files = bundle.get("files")
            if not isinstance(files, list) or len(files) > 500:
                raise HPCError("Invalid artifact list")
            verified, names, total = [], set(), 0
            for item in files:
                name = item["path"]
                if not isinstance(name, str):
                    raise HPCError("Artifact path must be a string")
                rel = PurePosixPath(name)
                allowed = name in {"experiment.json", "source_manifest.json", "worker_state.json",
                                   "stdout.log", "stderr.log", "scheduler.log"} or name.startswith("result/")
                if (not allowed or rel.is_absolute() or ".." in rel.parts or not rel.parts
                        or str(rel) != name or name in names or any(ord(c) < 32 for c in name)):
                    raise HPCError("Invalid or duplicate artifact path")
                raw = base64.b64decode(item["data_b64"], validate=True)
                total += len(raw)
                if total > cap or len(raw) != item["size_bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                    raise HPCError("Artifact size or hash validation failed")
                names.add(name)
                verified.append((item, raw))
            if manifest["status"] == "completed":
                required = {"experiment.json", "source_manifest.json", "worker_state.json",
                            "result/metrics.json", "result/status.json"}
                if not required <= names:
                    raise HPCError("Completed job is missing required workflow artifacts")
                payloads = {item["path"]: raw for item, raw in verified}
                if json.loads(payloads["source_manifest.json"]) != manifest["source_manifest"]:
                    raise HPCError("Collected source provenance differs from submitted source")
                actual_config = json.dumps(json.loads(payloads["experiment.json"]), sort_keys=True,
                                           allow_nan=False).encode()
                if hashlib.sha256(actual_config).hexdigest() != manifest["config_sha256"]:
                    raise HPCError("Collected experiment differs from submitted configuration")
                if (json.loads(payloads["worker_state.json"]).get("status") != "completed"
                        or json.loads(payloads["result/status.json"]).get("status") != "completed"):
                    raise HPCError("Scheduler success is inconsistent with worker/workflow status")
            # Create a fresh collection directory; interrupted copies remain for inspection.
            destination = directory / ("collected_" + uuid.uuid4().hex)
            destination.mkdir(mode=0o700)
            artifacts = []
            for item, raw in verified:
                target = destination / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(raw)
                artifacts.append({k: item[k] for k in ("path", "size_bytes", "sha256")})
            record = {"directory": str(destination), "files": artifacts, "total_bytes": total,
                      "collected_at": time.time(), "job_id": job_id, "execution_status": manifest["status"]}
            atomic_json(destination / "ARTIFACT_MANIFEST.json", record)
            return self._save(manifest, artifact_manifest=record)
