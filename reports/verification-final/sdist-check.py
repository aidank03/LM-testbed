"""Build and test the source distribution offline without changing the wheel."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/verification-final"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    record_path = OUT / "sdist-verification.json"
    target = ROOT / "dist/liner_stability-0.3.0.tar.gz"
    if record_path.exists() or target.exists():
        raise SystemExit("Refusing to overwrite a source archive or its verification")
    wheel = ROOT / "dist/liner_stability-0.3.0-py3-none-any.whl"
    before_wheel = sha(wheel)
    required = [ROOT / "requirements-core.lock", ROOT / "docs/third-party/Qwen3-0.6B-LICENSE.txt"]
    experiment = ROOT / "experiments/local_model"
    required += sorted(experiment.glob("*.py")) + sorted(experiment.glob("*.json"))
    required += [experiment / "requirements.lock", experiment / "posttrain_run/manifest.json"]
    required += sorted((experiment / "frozen_dataset").glob("*.json"))
    required += sorted((experiment / "frozen_dataset").glob("*.jsonl"))
    required += sorted((experiment / "posttrain_run/source_snapshot").glob("*.py"))
    required += sorted((experiment / "posttrain_run/source_snapshot").glob("*.md"))
    checked = required + [ROOT / "MANIFEST.in", ROOT / "pyproject.toml"]
    checked += sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    before = {str(p.relative_to(ROOT)): sha(p) for p in checked}
    record = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "python_executable": sys.executable, "source_sha256_before": before,
              "wheel_sha256_before": before_wheel,
              "scope": "Offline source-distribution build, exact source/fixture hashes, excluded-artifact probes, complete packaged test suite; no model or remote HPC calls."}
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    for key in ("OPENAI_API_KEY", "HF_TOKEN"):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(OUT / "matplotlib-cache"))
    try:
        with tempfile.TemporaryDirectory(prefix="factor-sdist-") as directory:
            work = Path(directory)
            stage = work / "project"
            stage.mkdir()
            for name in ("pyproject.toml", "README.md", "CHANGELOG.md", "MANIFEST.in", "Makefile", "requirements-core.lock"):
                shutil.copy2(ROOT / name, stage / name)
            for name in ("src", "configs", "docs", "templates", "tests"):
                shutil.copytree(ROOT / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            for source in required:
                destination = stage / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            # Tiny non-model sentinels prove exclusions without copying real weights.
            probes = [".venv/exclusion_probe.py", "experiments/local_model/base_model/exclusion_probe.json",
                      "src/factor/exclusion_probe.safetensors", "src/factor/exclusion_probe.gguf"]
            for name in probes:
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("packaging exclusion probe; not a model")
            command = [sys.executable, "-c", "from setuptools.build_meta import build_sdist; import sys; build_sdist(sys.argv[1])", str(work / "dist")]
            with (OUT / "sdist-build.txt").open("w") as log:
                built = subprocess.run(command, cwd=stage, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90)
            record.update(build_command=command, build_exit_code=built.returncode)
            if built.returncode:
                raise RuntimeError("Source-distribution build failed")
            archive = next((work / "dist").glob("*.tar.gz"))
            with tarfile.open(archive) as tar:
                members = tar.getmembers()
                prefixes = {Path(m.name).parts[0] for m in members}
                if len(prefixes) != 1:
                    raise RuntimeError("Unexpected source archive layout")
                prefix = prefixes.pop()
                files = {Path(m.name).relative_to(prefix).as_posix(): m for m in members if m.isfile()}
                forbidden = [n for n in files if n.endswith((".safetensors", ".gguf", ".pyc"))
                             or any(part in (".venv", ".venv-local", "base_model", "cache", "checkpoints") for part in Path(n).parts)]
                if forbidden:
                    raise RuntimeError(f"Forbidden archive entries: {forbidden}")
                missing = sorted(set(before) - files.keys())
                if missing:
                    raise RuntimeError(f"Missing packaged test/source inputs: {missing}")
                mismatched = [name for name, expected in before.items()
                              if hashlib.sha256(tar.extractfile(files[name]).read()).hexdigest() != expected]
                if mismatched:
                    raise RuntimeError(f"Packaged bytes differ: {mismatched}")
                record.update(archive_files={n: m.size for n, m in sorted(files.items())},
                              required_fixture_sha256={str(p.relative_to(ROOT)): before[str(p.relative_to(ROOT))] for p in required},
                              missing_required_files=missing, mismatched_source_files=mismatched,
                              excluded_probe_paths=probes, forbidden_archive_files=forbidden)
                extracted = work / "extracted"
                tar.extractall(extracted, filter="data")
            extracted_root = extracted / prefix
            test_env = env | {"PYTHONPATH": str(extracted_root / "src")}
            command = [sys.executable, "-W", "error::ResourceWarning", "-m", "unittest", "discover", "-s", "tests", "-v"]
            with (OUT / "sdist-unittest.txt").open("w") as log:
                tested = subprocess.run(command, cwd=extracted_root, env=test_env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            record.update(test_command=command, test_exit_code=tested.returncode)
            if tested.returncode:
                raise RuntimeError("Tests from extracted source archive failed")
            shutil.copy2(archive, target)
            record.update(archive_path=str(target), archive_sha256=sha(target), archive_bytes=target.stat().st_size,
                          checks_passed=True)
    except Exception as error:
        record.update(checks_passed=False, failure=str(error))
        raise
    finally:
        record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      wheel_sha256_after=sha(wheel), source_sha256_after={str(p.relative_to(ROOT)): sha(p) for p in checked})
        record["wheel_unchanged"] = before_wheel == record["wheel_sha256_after"]
        record["source_unchanged"] = before == record["source_sha256_after"]
        if not record["wheel_unchanged"] or not record["source_unchanged"]:
            record["checks_passed"] = False
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks_passed": record["checks_passed"], "archive": record["archive_path"], "sha256": record["archive_sha256"]}))
    return 0 if record["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
