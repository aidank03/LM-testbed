"""Execute a predeclared local campaign; no search, tuning, best-of, or cloud."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from factor.artifacts import digest
from factor.contracts import RuntimePolicy
from factor.evaluation_v2 import _validate_public, baseline_results, compare_methods
from factor.providers import LocalModel
from factor.runtime import ScientificRuntime
from liner_stability.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.campaign)
    envelope = json.loads(manifest_path.read_text())
    plan = envelope["campaign"]
    if digest(plan) != envelope["campaign_hash"]:
        raise ValueError("Campaign hash mismatch")
    for name, expected in plan["files"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen campaign file changed: {name}")
    suite = manifest_path.parent
    public = _validate_public(suite / "public_cases.json")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, out / "campaign.json")
    for package in ("factor", "liner_stability"):
        shutil.copytree(Path("src") / package, out / "source_snapshot" / package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    methods = {name: baseline_results(public, name) for name in plan["baselines"]}
    for method in plan["local_methods"]:
        policy = RuntimePolicy(**json.loads(Path(method["policy"]).read_text()))
        if policy.data_egress or policy.allow_hpc or policy.max_cloud_cost_usd:
            raise ValueError("This campaign runner only permits local observation tasks")
        model = LocalModel(method["model"], temperature=plan["common_settings"]["temperature"],
                           transport=plan["common_settings"]["transport"])
        runtime = ScientificRuntime(model, root=out / "runs", policy=policy)
        for repeat in range(method["repeats"]):
            label = f"{method['model']}:{method['mode']}:repeat-{repeat+1}"
            predictions = []
            for index, case in enumerate(public["cases"]):
                result = runtime.run({"question": case["question"], "observation": case["observation"]})
                predictions.append({"case_id": case["case_id"], "result": result})
                write_json(out / f"{label.replace(':', '-')}.json", predictions)
                print(f"{label} {index+1}/{len(public['cases'])}: {result['execution']}", flush=True)
            methods[label] = predictions
    # The grader first reads evaluator truth after every assigned model run.
    report = compare_methods(suite / "public_cases.json", suite / "evaluator_truth.json", methods)
    write_json(out / "comparison.json", report)
    changes = []
    for file in (out / "source_snapshot").rglob("*.py"):
        original = Path("src") / file.relative_to(out / "source_snapshot")
        if not original.is_file() or original.read_bytes() != file.read_bytes():
            changes.append(str(original))
    write_json(out / "completion.json", {"completed": True, "model_attempts": len(public["cases"])*sum(m["repeats"] for m in plan["local_methods"]),
                                        "source_changes_during_run": changes, "campaign_hash": envelope["campaign_hash"]})


if __name__ == "__main__":
    main()
