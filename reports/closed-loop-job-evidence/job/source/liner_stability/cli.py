"""Command-line entry point for liner-stability."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from . import __version__, benchmarks, design, evidence, evaluation, llm, physics, workflow
from .io import read_json, read_jsonl, write_json


def main(argv=None):
    parser=argparse.ArgumentParser(prog="liner-stability",description="Experiments, synthetic diagnostics, evaluation and evidence-backed AI tools.")
    parser.add_argument("--version",action="version",version=__version__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("run",help="Run a configured synthetic experiment and write an evaluation card")
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--n",type=int,help="Override cases per scenario")
    p=commands.add_parser("score",help="Score numerical predictions against separate truth")
    p.add_argument("--truth",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p=commands.add_parser("physics",help="Run an explicit-unit analytic check")
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p=commands.add_parser("design",help="Rank candidate experiments from supplied model predictions")
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    e=commands.add_parser("evidence",help="Index and search explicitly named local text files")
    es=e.add_subparsers(dest="action",required=True)
    p=es.add_parser("index")
    p.add_argument("--files",type=Path,nargs="+",required=True)
    p.add_argument("--out",type=Path,required=True)
    p=es.add_parser("search")
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--query",required=True)
    p.add_argument("--out",type=Path,required=True)
    p=commands.add_parser("ask",help="Create an evidence packet or request a live provisional model answer")
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--question",required=True)
    p.add_argument("--provider",choices=("offline","openai"),default="offline")
    p.add_argument("--model",default=os.environ.get("LINER_LLM_MODEL"))
    p.add_argument("--calculation",type=Path,help="Explicit analytic request to calculate and include in model context")
    p.add_argument("--out",type=Path,required=True)
    a=commands.add_parser("ai",help="Export public example tasks, run a candidate, or grade separate answers")
    aa=a.add_subparsers(dest="action",required=True)
    p=aa.add_parser("export")
    p.add_argument("--out",type=Path,required=True)
    p=aa.add_parser("run")
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--model",default=os.environ.get("LINER_LLM_MODEL"))
    p.add_argument("--out",type=Path,required=True)
    p=aa.add_parser("score")
    p.add_argument("--key",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    commands.add_parser("doctor",help="Report local dependencies and whether a live API key is configured")
    args=parser.parse_args(argv)
    try:
        if args.command=="run":
            workflow.run(read_json(args.config),args.out,args.n)
        elif args.command=="score":
            write_json(args.out,evaluation.score(read_jsonl(args.truth),read_jsonl(args.predictions)))
        elif args.command=="physics":
            write_json(args.out,physics.calculate(read_json(args.config)))
        elif args.command=="design":
            write_json(args.out,design.rank_designs(read_json(args.config)))
        elif args.command=="evidence":
            result=evidence.index_files(args.files) if args.action=="index" else evidence.search(read_json(args.index),args.query)
            write_json(args.out,result)
        elif args.command=="ask":
            calculation=physics.calculate(read_json(args.calculation)) if args.calculation else None
            write_json(args.out,llm.answer_question(args.question,read_json(args.index),args.model,calculation,args.provider))
        elif args.command=="ai":
            if args.action=="export":
                benchmarks.export(args.out)
            elif args.action=="run":
                llm.run_benchmark(read_json(args.inputs),args.out,args.model)
            else:
                write_json(args.out,benchmarks.score_answers(read_json(args.key),read_json(args.predictions)))
        else:
            print(json.dumps({"version":__version__,"python":platform.python_version(),
                              "dependencies":{n:importlib.metadata.version(n) for n in ("numpy","scipy","matplotlib")},
                              "openai_api_key_configured":bool(os.environ.get("OPENAI_API_KEY")),
                              "live_model_validation":"not established by doctor", "core_workflow":"offline"},indent=2))
            return
    except (ValueError,OSError,KeyError,llm.ModelError) as exc:
        parser.exit(2,f"Error: {exc}\n")
    print(f"Output: {args.out}")
