"""Run only after the live campaign finishes and root freezes release sources.

No network, model calls, training, shared-environment installs, or source edits.
Creates a fresh report directory and builds a wheel from a temporary source copy.
"""
import argparse
from datetime import datetime, timezone
import email
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", default="0.3.0")
    parser.add_argument("--out", default="reports/verification/release-candidate")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = (root / args.out).resolve()
    if not out.is_relative_to(root / "reports" / "verification"):
        raise ValueError("Final verification output must stay under reports/verification")
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "src"))
    import factor
    import liner_stability
    declared = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M).group(1)
    versions = {"expected": args.expected_version, "distribution": declared,
                "factor": factor.__version__, "liner_stability": liner_stability.__version__}
    if set(versions.values()) != {args.expected_version}:
        (out / "version-mismatch.json").write_text(json.dumps(versions, indent=2))
        raise SystemExit("Release version mismatch; see version-mismatch.json")

    def sources():
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for base in ("src", "tests") for path in sorted((root / base).rglob("*.py"))}

    before = sources()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1",
               MPLCONFIGDIR=str(out / "matplotlib-cache"))
    command = [sys.executable, "-W", "error::ResourceWarning", "-m", "unittest", "discover", "-s", "tests", "-v"]
    started = time.monotonic()
    record = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "versions": versions,
              "python": sys.version, "python_executable": sys.executable,
              "platform": platform.platform(), "test_command": command,
              "source_sha256_before": before}
    with (out / "unittest.txt").open("w") as log:
        tested = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    record["test_exit_code"] = tested.returncode
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True, timeout=30)
    (out / "requirements-freeze.txt").write_text(freeze.stdout)
    record["distributions"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if tested.returncode == 0:
        with tempfile.TemporaryDirectory(prefix="factor-final-package-") as directory, (out / "packaging-smoke.txt").open("w") as log:
            work = Path(directory)
            stage = work / "project"
            stage.mkdir()
            for name in ("pyproject.toml", "README.md", "CHANGELOG.md", "MANIFEST.in", "Makefile"):
                shutil.copy2(root / name, stage / name)
            for name in ("src", "configs", "docs", "templates"):
                shutil.copytree(root / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            build = [sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation",
                     "--wheel-dir", str(out / "wheels"), str(stage)]
            built = subprocess.run(build, cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=90)
            record["wheel_build_exit_code"] = built.returncode
            if built.returncode == 0:
                wheel = next((out / "wheels").glob("*.whl"))
                unpacked = work / "unpacked"
                with zipfile.ZipFile(wheel) as archive:
                    metadata = email.message_from_bytes(archive.read(next(p for p in archive.namelist() if p.endswith(".dist-info/METADATA"))))
                    record["wheel_version"] = metadata["Version"]
                    record["wheel_contents"] = archive.namelist()
                    archive.extractall(unpacked)
                record["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
                isolated = ('import sys; sys.path.insert(0,sys.argv[1]); import factor,liner_stability; '
                            'assert factor.__version__==liner_stability.__version__==sys.argv[2]; '
                            'from factor.cli import main; raise SystemExit(main(sys.argv[3:]))')
                for name, options in [("doctor", ["doctor"]),
                                      ("conventional_run", ["run", "--request", str(stage / "configs/motion_request.json"), "--out", str(work / "runs")])]:
                    outcome = subprocess.run([sys.executable, "-I", "-c", isolated, str(unpacked), args.expected_version, *options],
                                             cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=30)
                    record[name + "_exit_code"] = outcome.returncode
    record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                  source_sha256_after=sources())
    record["source_changed_during_checks"] = before != record["source_sha256_after"]
    record["evidence_scope"] = "Implementation/fixture tests and actual local numerical jobs; isolated wheel smoke, no live models or remote cluster."
    success = (all(record.get(key) == 0 for key in ("test_exit_code", "wheel_build_exit_code", "doctor_exit_code", "conventional_run_exit_code"))
               and record.get("wheel_version") == args.expected_version and not record["source_changed_during_checks"])
    record["checks_passed"] = success
    (out / "environment.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"checks_passed": success, "report_directory": str(out), "elapsed_seconds": record["elapsed_seconds"]}))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
