"""Append-only local records for Learning Lab experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
import uuid


def _write_new(path: Path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class LearningRun:
    def __init__(self, root, kind, config):
        self.id = "learn_" + uuid.uuid4().hex[:16]
        self.path = Path(root).resolve() / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.state = {"id": self.id, "kind": kind, "status": "running",
                      "created_at": time.time(), "events": [], "run_path": str(self.path)}
        _write_new(self.path / "request.json", {"kind": kind, "config": config})

    def emit(self, event):
        if not isinstance(event, dict):
            raise ValueError("event must be an object")
        with self.lock:
            record = {"sequence": len(self.state["events"]), **event}
            self.state["events"].append(record)
            with (self.path / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    def complete(self, result):
        body = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with self.lock:
            self.state.update(status="completed", result=result,
                              result_sha256=hashlib.sha256(body).hexdigest(), completed_at=time.time())
            _write_new(self.path / "result.json", result)
            _write_new(self.path / "run.json", {k: v for k, v in self.state.items() if k != "events"})

    def fail(self, error):
        with self.lock:
            self.state.update(status="failed", error=str(error), completed_at=time.time())
            _write_new(self.path / "run.json", {k: v for k, v in self.state.items() if k != "events"})

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))
