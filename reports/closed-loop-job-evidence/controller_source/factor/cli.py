"""Factor's command-line surface uses the same callable environment as an SDK."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from liner_stability.io import write_json
from .artifacts import verify_run
from .contracts import ContractError, RuntimePolicy
from .providers import ConventionalAgent, LocalModel, OpenAIModel
from .runtime import ScientificRuntime, TOOLS


def read(path):
    return json.loads(Path(path).read_text())


def provider(args):
    if args.provider == "conventional":
        return ConventionalAgent()
    if not args.model:
        raise ContractError("Supply an explicit --model")
    if args.provider == "local":
        return LocalModel(args.model, base_url=args.base_url, transport=args.local_transport)
    if not args.cloud_profile:
        raise ContractError("Cloud execution requires --cloud-profile with explicit budget and prices")
    return OpenAIModel(args.model, **read(args.cloud_profile))


def policy(args):
    return RuntimePolicy(**read(args.policy)) if args.policy else RuntimePolicy()


def manager(args):
    if not args.hpc_profile:
        return None
    from .hpc import HPCManager, SlurmConfig
    settings = read(args.hpc_profile)
    if args.slurm_profile:
        settings["slurm"] = SlurmConfig(**read(args.slurm_profile))
    return HPCManager(**settings)


def common(parser):
    parser.add_argument("--provider", choices=("conventional", "local", "openai"), default="conventional")
    parser.add_argument("--model")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--local-transport", choices=("lmstudio-native", "openai"), default="lmstudio-native")
    parser.add_argument("--cloud-profile")
    parser.add_argument("--policy")
    parser.add_argument("--hpc-profile")
    parser.add_argument("--slurm-profile")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="factor", description="A bounded scientific environment with separate evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Execute one observation-only scientific request")
    run.add_argument("--request", required=True)
    run.add_argument("--out", required=True)
    common(run)
    suite = sub.add_parser("suite", help="Freeze a new synthetic benchmark; does not run models")
    suite.add_argument("--out", required=True)
    suite.add_argument("--n", type=int, default=24)
    suite.add_argument("--seed", type=int, default=20260905)
    suite.add_argument("--split", choices=("train", "development", "final"), default="development")
    evaluate = sub.add_parser("evaluate", help="Run public cases, then score separately; records test exposure")
    evaluate.add_argument("--suite", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--repeats", type=int, default=1)
    common(evaluate)
    doctor = sub.add_parser("doctor", help="Read capability presence without showing credentials")
    verify = sub.add_parser("verify-run", help="Verify local event/artifact integrity, not scientific truth")
    verify.add_argument("path")
    hpc = sub.add_parser("job", help="Manage configured jobs, with explicit profile")
    hpc.add_argument("action", choices=("submit", "status", "collect", "cancel"))
    hpc.add_argument("--hpc-profile", required=True)
    hpc.add_argument("--slurm-profile")
    hpc.add_argument("--backend", choices=("local", "slurm"), default="local")
    hpc.add_argument("--config")
    hpc.add_argument("--id")
    hpc.add_argument("--request-id", help="Stable operator submission ID for explicit retry reconciliation")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = {"python": platform.python_version(), "platform": platform.platform(),
                      "executables": {name: shutil.which(name) is not None for name in ("ssh", "sbatch", "lms", "ollama")},
                      "credential_present": {name: bool(os.environ.get(name)) for name in ("OPENAI_API_KEY", "HF_TOKEN")},
                      "note": "Presence is not verified access. No server was contacted, model loaded, or job submitted."}
        elif args.command == "verify-run":
            result = verify_run(args.path)
        elif args.command == "suite":
            from .evaluation import create_suite
            result = {k: str(v) for k, v in create_suite(args.out, args.seed, args.n, args.split).items()}
        elif args.command == "job":
            mgr = manager(args)
            if args.action == "submit":
                if not args.config:
                    raise ContractError("Submission requires --config")
                result = mgr.submit(read(args.config), backend=args.backend, request_id=args.request_id)
            else:
                if not args.id:
                    raise ContractError("Job operation requires --id")
                result = getattr(mgr, args.action)(args.id)
        elif args.command == "run":
            runtime = ScientificRuntime(provider(args), root=args.out, policy=policy(args), hpc=manager(args))
            result = runtime.run(read(args.request))
        else:
            from .evaluation import baseline_results, compare_methods, _validate_public
            if not 1 <= args.repeats <= 10:
                raise ContractError("repeats must be 1–10")
            directory = Path(args.suite)
            # Candidate execution gets only this public structure, never truth.
            public = _validate_public(directory / "public_cases.json")
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=False)
            candidate = provider(args)  # Shared total reservation budget across cases/repeats.
            runtime = ScientificRuntime(candidate, root=out / "runs", policy=policy(args), hpc=manager(args))
            methods = {"naive": baseline_results(public, "naive"), "bounded": baseline_results(public, "bounded")}
            for repetition in range(args.repeats):
                results = []
                label = f"{args.provider}:{candidate.model}:repeat-{repetition+1}"
                for index, case in enumerate(public["cases"]):
                    result = runtime.run({"question": case["question"], "observation": case["observation"]})
                    results.append({"case_id": case["case_id"], "result": result})
                    write_json(out / f"candidate-{repetition+1}.json", results)
                    print(f"{label}: {index+1}/{len(public['cases'])} {result['execution']}", file=sys.stderr, flush=True)
                methods[label] = results
            # Evaluator truth is first read by the evaluator after all runs.
            result = compare_methods(directory / "public_cases.json", directory / "evaluator_truth.json", methods)
            write_json(out / "comparison.json", result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "run" and result["execution"] != "completed":
            return 2
        return 0
    except (ContractError, ValueError, OSError, RuntimeError) as exc:
        print(f"Factor: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
