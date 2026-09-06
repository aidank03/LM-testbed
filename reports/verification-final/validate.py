"""Offline release validation; does not edit source or the historical environment.

Run with the historical numerical verification Python. Creates the repository's
new .venv once, using the installed wheel and copies of its verified numerical
dependency distributions. No network, model, remote scheduler or training calls.
"""
from datetime import datetime, timezone
import email
import hashlib
import importlib.metadata as metadata
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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "verification-final"
BASE = Path("/Users/operator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")
CORE = ROOT / ".venv"
EXPECTED = "0.3.0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    return {str(p.relative_to(ROOT)): digest(p)
            for base in ("src", "tests") for p in sorted((ROOT / base).rglob("*.py"))} | {
                "pyproject.toml": digest(ROOT / "pyproject.toml")}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    if (OUT / "environment.json").exists() or CORE.exists():
        raise SystemExit("Refusing to replace final verification evidence or an existing core environment")
    OUT.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    started = time.monotonic()
    record = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "python_executable": sys.executable,
              "platform": platform.platform(), "sha256_before": before,
              "scope": "Implementation and fixture tests, actual local subprocess tests, offline installed-wheel checks; no model or remote HPC calls."}
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               MPLCONFIGDIR=str(OUT / "matplotlib-cache"))
    # Hide any inherited provider credentials in child environments; none are needed.
    for key in ("OPENAI_API_KEY", "HF_TOKEN"):
        env.pop(key, None)
    checks = []

    def run(label, argv, *, cwd=ROOT, environ=env, timeout=180):
        with (OUT / (label + ".txt")).open("w") as stream:
            result = subprocess.run([str(v) for v in argv], cwd=cwd, env=environ,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        checks.append({"label": label, "argv": [str(v) for v in argv], "cwd": str(cwd), "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect its saved output")
        return result

    try:
        sys.path.insert(0, str(ROOT / "src"))
        import factor
        import liner_stability
        declared = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
        record["versions"] = {"distribution_declared": declared, "factor": factor.__version__,
                              "liner_stability": liner_stability.__version__}
        if set(record["versions"].values()) != {EXPECTED}:
            raise RuntimeError("Source versions disagree")
        run("unittest", [sys.executable, "-W", "error::ResourceWarning", "-m", "unittest", "discover", "-s", "tests", "-v"])
        run("requirements-freeze", [sys.executable, "-m", "pip", "freeze", "--all"])
        record["verification_distributions"] = {d.metadata["Name"]: d.version for d in metadata.distributions()}

        with tempfile.TemporaryDirectory(prefix="factor-final-wheel-") as directory:
            work = Path(directory)
            stage = work / "project"
            stage.mkdir()
            for name in ("pyproject.toml", "README.md", "CHANGELOG.md", "MANIFEST.in", "Makefile"):
                shutil.copy2(ROOT / name, stage / name)
            for name in ("src", "configs", "docs", "templates"):
                shutil.copytree(ROOT / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            run("wheel-build", [sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation", "--wheel-dir", work / "wheels", stage], cwd=work)
            built = next((work / "wheels").glob("*.whl"))
            target = ROOT / "dist" / built.name
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                raise RuntimeError("Refusing to overwrite a deliverable wheel")
            shutil.copy2(built, target)
            with zipfile.ZipFile(target) as archive:
                package = email.message_from_bytes(archive.read(next(p for p in archive.namelist() if p.endswith(".dist-info/METADATA"))))
                entry_points = archive.read(next(p for p in archive.namelist() if p.endswith(".dist-info/entry_points.txt"))).decode()
                record["wheel"] = {"path": str(target), "sha256": digest(target), "bytes": target.stat().st_size,
                                   "version": package["Version"], "entry_points": entry_points,
                                   "contents": archive.namelist()}
                if package["Version"] != EXPECTED:
                    raise RuntimeError("Wheel metadata version mismatch")
            run("venv-create", [BASE, "-m", "venv", "--system-site-packages", CORE], cwd=work)
            core_env = env.copy()
            core_env.pop("PYTHONPATH", None)
            core_env.pop("PYTHONHOME", None)
            core_env["PATH"] = str(CORE / "bin") + os.pathsep + core_env.get("PATH", "")
            core_python = CORE / "bin" / "python"
            run("wheel-install", [core_python, "-m", "pip", "install", "--no-index", "--no-deps", target], cwd=work, environ=core_env)
            # The bundled base runtime lacks scipy/matplotlib. Copy only the core
            # numerical dependency closure from inspected distributions, never the
            # historical Factor distribution or provider packages. No shared writes.
            site = CORE / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            copied = []
            dependencies = ("numpy", "scipy", "matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver", "packaging", "pillow", "pyparsing", "python-dateutil", "six")
            for name in dependencies:
                distribution = metadata.distribution(name)
                origin = Path(distribution.locate_file("")).resolve()
                info = {"name": name, "version": distribution.version, "source_site_packages": str(origin), "files": []}
                for relative in distribution.files or ():
                    relative = Path(relative)
                    if relative.is_absolute() or ".." in relative.parts or relative.suffix == ".pyc":
                        continue
                    source = origin / relative
                    if not source.is_file():
                        continue
                    destination = site / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() and digest(source) != digest(destination):
                        raise RuntimeError(f"Dependency copy would overwrite different file: {relative}")
                    shutil.copy2(source, destination)
                    info["files"].append({"relative_path": str(relative), "sha256": digest(destination)})
                copied.append(info)
            write("core-dependency-copy-manifest.json", copied)
            record["core_environment"] = {"path": str(CORE), "configuration": (CORE / "pyvenv.cfg").read_text(),
                                          "numerical_dependency_source": sys.executable,
                                          "copied_dependencies": [{k: v for k, v in d.items() if k != "files"} for d in copied],
                                          "note": "Uses bundled runtime system-site-packages plus local copies of the verified numerical dependency closure. Historical numerical environment is unchanged. Dependency command-line scripts outside site-packages were not copied."}
            run("core-requirements-freeze", [core_python, "-m", "pip", "freeze", "--all"], cwd=work, environ=core_env)
            package_check = '''import factor,liner_stability,importlib.metadata as m,json,sys,pathlib
root=pathlib.Path(sys.argv[1]).resolve()
assert factor.__version__==liner_stability.__version__==m.version("liner-stability")=="0.3.0"
assert pathlib.Path(factor.__file__).is_relative_to(root)
assert pathlib.Path(liner_stability.__file__).is_relative_to(root)
print(json.dumps({"factor":factor.__file__,"liner_stability":liner_stability.__file__,"version":m.version("liner-stability"),"dependencies":{n:m.version(n) for n in ("numpy","scipy","matplotlib")},"sys_path":sys.path},indent=2))'''
            run("installed-package-origins", [core_python, "-I", "-c", package_check, CORE], cwd=work, environ=core_env)
            run("factor-doctor", [CORE / "bin" / "factor", "doctor"], cwd=work, environ=core_env)
            run("legacy-version", [CORE / "bin" / "liner-stability", "--version"], cwd=work, environ=core_env)
            run("legacy-doctor", [CORE / "bin" / "liner-stability", "doctor"], cwd=work, environ=core_env)
            run("conventional-run", [CORE / "bin" / "factor", "run", "--request", stage / "configs" / "motion_request.json", "--out", work / "request-run"], cwd=work, environ=core_env)
            run("v2-suite", [CORE / "bin" / "factor", "suite", "--out", work / "suite", "--n", "2"], cwd=work, environ=core_env)
            run("v2-evaluate-conventional", [CORE / "bin" / "factor", "evaluate", "--provider", "conventional", "--suite", work / "suite", "--out", work / "evaluation"], cwd=work, environ=core_env)
            shutil.copytree(work / "suite", OUT / "v2-suite")
            shutil.copy2(work / "evaluation" / "comparison.json", OUT / "v2-conventional-comparison.json")
            run("corrupt-contract-preflight", [core_python, "-I", "-c", '''from pathlib import Path
from unittest.mock import patch
import json,sys
from factor.cli import main
suite,out=map(Path,sys.argv[1:])
path=suite/"public_cases.json"
public=json.loads(path.read_text())
assert public["schema_version"]=="factor-cases/2"
public["contract_hash"]="0"*64
path.write_text(json.dumps(public))
with patch("factor.cli.provider",side_effect=AssertionError("provider must not be invoked")) as provider:
    result=main(["evaluate","--provider","local","--model","never-contacted","--suite",str(suite),"--out",str(out)])
    assert result==2,result
    provider.assert_not_called()
assert not out.exists()
print(json.dumps({"exit_code":result,"provider_factory_calls":0,"model_calls":0,"output_created":False}))''', work / "suite", work / "must-not-exist"], cwd=work, environ=core_env)
        record["checks_passed"] = True
    except Exception as exc:
        record["checks_passed"] = False
        record["failure"] = str(exc)
        raise
    finally:
        record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                      commands=checks, sha256_after=snapshot())
        record["source_changed_during_checks"] = {k: v for k, v in before.items() if not k.startswith("tests/")} != {k: v for k, v in record["sha256_after"].items() if not k.startswith("tests/")}
        record["tests_changed_during_checks"] = {k: v for k, v in before.items() if k.startswith("tests/")} != {k: v for k, v in record["sha256_after"].items() if k.startswith("tests/")}
        if record["source_changed_during_checks"]:
            record["checks_passed"] = False
        write("environment.json", record)
    print(json.dumps({"checks_passed": record["checks_passed"], "elapsed_seconds": record["elapsed_seconds"], "out": str(OUT)}))
    return 0 if record["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
