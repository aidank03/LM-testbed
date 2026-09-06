"""Local run artifacts with exclusive ownership and a verifiable event chain."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

from liner_stability.io import write_json
from .contracts import ContractError


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(canonical(data)).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, root, *, run_id=None):
        self.run_id = run_id or "run_" + uuid.uuid4().hex
        if not re.fullmatch(r"run_[a-f0-9]{32}", self.run_id):
            raise ContractError("Invalid run ID")
        self.path = Path(root).resolve() / self.run_id
        self.path.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.events = []
        self.artifacts = {}
        self._previous_hash = "0" * 64

    def artifact(self, kind, data):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,50}", kind):
            raise ContractError("Invalid artifact kind")
        identity = "artifact_" + digest({"kind": kind, "data": data})[:24]
        if identity not in self.artifacts:
            relative = f"artifacts/{identity}.json"
            write_json(self.path / relative, data)
            self.artifacts[identity] = {"id": identity, "kind": kind, "path": relative,
                                        "sha256": hashlib.sha256((self.path / relative).read_bytes()).hexdigest()}
            write_json(self.path / "artifacts.json", self.artifacts)
        return identity

    def event(self, kind, data):
        record = {"sequence": len(self.events), "timestamp": now(), "kind": kind,
                  "data": data, "previous_hash": self._previous_hash}
        record["hash"] = digest(record)
        self.events.append(record)
        self._previous_hash = record["hash"]
        # Exclusive per-run ownership; atomic snapshot leaves a parseable prefix.
        write_json(self.path / "events.json", self.events)
        return record

    def result(self, data):
        write_json(self.path / "result.json", data)
        self.event("result_written", {"result_sha256": digest(data)})
        return data


def verify_run(path):
    """Verify integrity, not scientific truth, signatures, or source authenticity."""
    path = Path(path)
    events = json.loads((path / "events.json").read_text())
    previous = "0" * 64
    for index, event in enumerate(events):
        fields = {k: v for k, v in event.items() if k != "hash"}
        if event["sequence"] != index or event["previous_hash"] != previous or digest(fields) != event["hash"]:
            raise ContractError("Event chain verification failed")
        previous = event["hash"]
    artifacts = json.loads((path / "artifacts.json").read_text())
    for record in artifacts.values():
        candidate = (path / record["path"]).resolve()
        if not candidate.is_relative_to(path.resolve()) or hashlib.sha256(candidate.read_bytes()).hexdigest() != record["sha256"]:
            raise ContractError("Artifact integrity verification failed")
    bindings = [e for e in events if e["kind"] == "result_written"]
    if bindings and digest(json.loads((path / "result.json").read_text())) != bindings[-1]["data"]["result_sha256"]:
        raise ContractError("Result integrity verification failed")
    return {"events": len(events), "artifacts": len(artifacts), "integrity": "verified",
            "result_integrity": "verified" if bindings else "unbound_legacy_result"}
