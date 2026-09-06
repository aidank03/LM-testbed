"""Io functions for the synthetic development benchmark."""
import json
from pathlib import Path
import os
import tempfile


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant is not allowed: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, obj):
    write_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    write_text(path, "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
