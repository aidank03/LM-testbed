"""Trusted, standalone supervisor for the fixed synthetic benchmark template.

Copied with the exact numerical source into each job; no model-written commands.
This worker is process isolation for reliability, not an untrusted-code sandbox.
"""
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time

TERMINAL = {"completed", "failed", "cancelled", "timed_out"}


def atomic_json(path, data):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, allow_nan=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def bundle_artifacts(directory, cap):
    """Return bounded regular-file artifacts. Never follow links or include source."""
    directory = Path(directory).resolve()
    names = ["experiment.json", "source_manifest.json", "worker_state.json",
             "stdout.log", "stderr.log", "scheduler.log"]
    result = directory / "result"
    if result.is_symlink():
        raise ValueError("Symbolic artifact directories are prohibited")
    if result.exists():
        for parent, dirs, files in os.walk(result, followlinks=False):
            if any((Path(parent) / d).is_symlink() for d in dirs):
                raise ValueError("Symbolic artifact directories are prohibited")
            names.extend(str((Path(parent) / f).relative_to(directory)) for f in files)
    if len(names) > 500:
        raise ValueError("Too many artifact files")
    files, total = [], 0
    for name in sorted(names):
        path = directory / name
        if path.is_symlink():
            raise ValueError("Symbolic artifact files are prohibited")
        if not path.exists():
            continue
        if not path.is_file() or not path.resolve().is_relative_to(directory):
            raise ValueError("Invalid artifact path")
        if path.stat().st_size > cap - total:
            raise ValueError("Artifact size limit exceeded")
        with path.open("rb") as handle:
            raw = handle.read(cap - total + 1)
        total += len(raw)
        if total > cap:
            raise ValueError("Artifact size limit exceeded")
        files.append({"path": name, "size_bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "data_b64": base64.b64encode(raw).decode("ascii")})
    return {"files": files, "total_bytes": total}


def run(directory):
    directory = Path(directory).resolve(strict=True)
    state_path = directory / "worker_state.json"
    state = {"status": "running", "started_at": time.time(), "worker_pid": os.getpid(),
             "python": sys.version, "executable": sys.executable,
             "platform": platform.platform(), "dependencies": {}}
    process = None
    interrupted = [False]
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    try:
        config = json.loads((directory / "worker_config.json").read_text())
        for item in json.loads((directory / "source_manifest.json").read_text())["files"]:
            path = directory / item["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise ValueError("Source path escaped job directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Source digest mismatch")
        for name in ("numpy", "scipy", "matplotlib"):
            try:
                state["dependencies"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                state["dependencies"][name] = "unavailable"
        command = [sys.executable, "-m", "liner_stability", "run", "--config",
                   str(directory / "experiment.json"), "--out", str(directory / "result")]
        # No inherited API keys, SSH agent socket, PYTHONPATH or user site packages.
        env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
        env.update(PYTHONPATH=str(directory / "source"), PYTHONNOUSERSITE="1",
                   MPLCONFIGDIR=str(directory / "matplotlib-cache"),
                   OMP_NUM_THREADS=str(config["cpus"]), OPENBLAS_NUM_THREADS=str(config["cpus"]),
                   MKL_NUM_THREADS=str(config["cpus"]))
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=directory, env=env, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            state["child_pid"] = process.pid
            deadline = time.monotonic() + config["wall_seconds"]
            while process.poll() is None:
                state["heartbeat_at"] = time.time()
                atomic_json(state_path, state)
                if interrupted[0] or (directory / "cancel_requested").exists():
                    state["status"] = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    state["status"] = "timed_out"
                    break
                time.sleep(0.2)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            state["exit_code"] = process.wait()
            if state["status"] == "running":
                state["status"] = "completed" if process.returncode == 0 else "failed"
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    finally:
        state["finished_at"] = time.time()
        state["heartbeat_at"] = time.time()
        atomic_json(state_path, state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        raise SystemExit(run(sys.argv[2]))
    if len(sys.argv) == 4 and sys.argv[1] == "collect":
        print(json.dumps(bundle_artifacts(sys.argv[2], int(sys.argv[3])), allow_nan=False))
    else:
        raise SystemExit("Only the fixed run and collect operations are supported")
