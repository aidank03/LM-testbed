"""Summarize saved agent scores and audit traces after the scored report exists.

Never reads evaluator truth, regenerates data, calls a model, or regrades answers.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import tempfile


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def integrity(run, candidate):
    """Independent byte/hash audit; hashes do not prove scientific correctness."""
    run = Path(run)
    events, artifacts, result = read(run / "events.json"), read(run / "artifacts.json"), read(run / "result.json")
    previous = "0" * 64
    for i, event in enumerate(events):
        fields = {k: v for k, v in event.items() if k != "hash"}
        if event["sequence"] != i or event["previous_hash"] != previous or digest(fields) != event["hash"]:
            raise ValueError(f"Event chain failed: {run}")
        previous = event["hash"]
    for record in artifacts.values():
        path = (run / record["path"]).resolve()
        if not path.is_relative_to(run.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Artifact hash failed: {run}")
        expected_id = "artifact_" + digest({"kind": record["kind"], "data": read(path)})[:24]
        if expected_id != record["id"]:
            raise ValueError(f"Artifact identity failed: {run}")
    bindings = [event for event in events if event["kind"] == "result_written"]
    if not bindings or bindings[-1]["data"]["result_sha256"] != digest(result):
        raise ValueError(f"Missing or inconsistent final result binding: {run}")
    if result != candidate or result["artifacts"] != artifacts:
        raise ValueError(f"Candidate/result/artifact index disagreement: {run}")
    return events, artifacts


def count(value):
    return {k: value[k] for k in ("successes", "total", "rate")}


def locate_run(root, result):
    """Use the evidence copy being audited, preserving original recorded paths."""
    root = Path(root).resolve()
    run_id = result.get("run_id", "")
    if not re.fullmatch(r"run_[0-9a-f]{32}", run_id):
        raise ValueError("Invalid recorded run identity")
    expected = root / "runs" / run_id
    if expected.is_dir() and expected.resolve().is_relative_to(root):
        return expected
    matches = [p for p in root.rglob(run_id)
               if p.is_dir() and p.resolve().is_relative_to(root)]
    if len(matches) != 1:
        raise ValueError(f"Expected one local evidence directory for {run_id}")
    return matches[0]


def summarize(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Wait for the completed scored comparison before opening run outcomes")
    report = read(path)
    if report.get("schema_version") != "factor-method-comparison/2":
        raise ValueError("Expected the explicit v2 saved comparison")
    root = path.parent
    campaign = read(root / "campaign.json")
    if digest(campaign["campaign"]) != campaign["campaign_hash"]:
        raise ValueError("Campaign integrity failed")
    summary = {"schema_version": "factor-agent-audit/2", "source": str(path),
               "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "campaign_hash": campaign["campaign_hash"], "methods": {},
               "paired_comparisons": [], "repeat_agreement": [],
               "limitations": ["Saved deterministic scores, not an LLM judge.",
                               "Repeated analyses do not increase independent scientific cases.",
                               "Artifact existence and hash integrity do not establish evidence support.",
                               "Direct mode cannot submit a machine-scored interval in this interface.",
                               "Native finish status is inferred from output token count, not a server termination reason."]}
    for name, report_method in report["methods"].items():
        overall = report_method["overall"]
        info = {"cases": overall["cases"],
                "independent_groups": overall["contract_correct"]["experiment_all_pass"]["total"],
                "metrics": {key: count(overall[key]) for key in ("completed", "contract_correct", "wrong_decisive", "wrong_decisive_given_decisive", "required_abstention_recall", "abstained", "evidence_present")},
                "numerical_outputs": overall["numerical_outputs"],
                "finite_numerical_completed": sum(r["completed"] and r["coverage"] is not None for r in report_method["rows"]),
                "provisional_gates": report_method["provisional_gates"],
                "regimes": {regime: {"cases": r["cases"], "correct": count(r["contract_correct"]),
                                      "wrong_decisive": count(r["wrong_decisive"]), "abstained": count(r["abstained"]),
                                      "numerical_outputs": r["numerical_outputs"]}
                            for regime, r in report_method["regimes"].items()}}
        summary["methods"][name] = info
        if name in ("naive", "bounded"):
            continue
        model, mode, repeat = name.rsplit(":", 2)
        info.update(model=model, mode=mode, repeat=int(repeat.removeprefix("repeat-")))
        results = read(root / (name.replace(":", "-") + ".json"))
        score_rows = {row["case_id"]: row for row in report_method["rows"]}
        if len(results) != len(score_rows) or {r["case_id"] for r in results} != set(score_rows):
            raise ValueError(f"Incomplete or duplicate candidate IDs: {name}")
        actions, tool_errors, provider_errors, finishes, sources = Counter(), Counter(), Counter(), Counter(), defaultdict(set)
        tokens, executions, integrity_counts = Counter(), Counter(), Counter()
        inputs, outputs, elapsed, run_records = [], [], [], []
        recorded_calls = 0
        missing_token_calls = 0
        for record in results:
            result, cid = record["result"], record["case_id"]
            run = locate_run(root, result)
            events, artifacts = integrity(run, result)
            integrity_counts["bound_result_chain_and_artifacts"] += 1
            executions[result["execution"]] += 1
            elapsed.append(result["usage"]["elapsed_seconds"])
            issues = []
            for event in events:
                if event["kind"] in ("model_action", "provider_error"):
                    recorded_calls += 1
                    metadata = event["data"].get("metadata", {})
                    native = metadata.get("usage") or {}
                    inp = metadata.get("input_tokens", native.get("input_tokens"))
                    out = metadata.get("output_tokens", native.get("total_output_tokens"))
                    if isinstance(inp, (int, float)):
                        inputs.append(inp); tokens["input_tokens"] += inp
                    if isinstance(out, (int, float)):
                        outputs.append(out); tokens["output_tokens"] += out
                    if inp is None or out is None:
                        missing_token_calls += 1
                    finishes[str(metadata.get("finish_reason"))] += 1
                    if event["kind"] == "provider_error":
                        provider_errors[event["data"]["error"]] += 1
                        issues.append({"kind": "provider_error", "error": event["data"]["error"],
                                       "finish_reason": metadata.get("finish_reason"), "input_tokens": inp, "output_tokens": out})
                    else:
                        actions[event["data"]["tool"]] += 1
                if event["kind"] == "tool_result" and not event["data"]["ok"]:
                    key = f'{event["data"]["tool"]}: {event["data"]["error"]}'
                    tool_errors[key] += 1
                    issues.append({"kind": "tool_error", "error": key})
            manifest_record = next(a for a in artifacts.values() if a["kind"] == "run_manifest")
            manifest = read(run / manifest_record["path"])
            for filename, sha in manifest["source_sha256"].items():
                sources[filename].add(sha)
                snapshot = root / "source_snapshot" / "factor" / filename
                if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != sha:
                    raise ValueError(f"Run source differs from frozen snapshot: {filename}: {run}")
            final = result.get("final") or {}
            evidence = final.get("evidence_ids") or []
            run_records.append({"case_id": cid, "run_path": str(run), "recorded_run_path": result["run_path"],
                                "execution": result["execution"],
                                "failure": result.get("failure"), "decision": final.get("decision"),
                                "contract_correct": score_rows[cid]["contract_correct"],
                                "condition": score_rows[cid]["condition"], "issues": issues,
                                "evidence_exists": bool(evidence) and all(e in artifacts for e in evidence),
                                "evidence_kinds": [artifacts[e]["kind"] for e in evidence if e in artifacts]})
        reported = report_method["usage_totals_reported"]
        info.update(executions=dict(executions), action_counts=dict(actions), tool_errors=dict(tool_errors),
                    provider_errors=dict(provider_errors), finish_statuses=dict(finishes),
                    integrity=dict(integrity_counts), source_versions={k: sorted(v) for k, v in sources.items()},
                    usage_reported=reported, audited_tokens=dict(tokens), recorded_model_events=recorded_calls,
                    calls_missing_token_usage=missing_token_calls,
                    maximum_input_tokens=max(inputs, default=None), maximum_output_tokens=max(outputs, default=None),
                    elapsed_sum_seconds=sum(elapsed), elapsed_mean_seconds=statistics.mean(elapsed),
                    elapsed_median_seconds=statistics.median(elapsed), runs=run_records,
                    aggregate_token_fields_match_events=all(reported.get(k, 0) == v for k, v in tokens.items()))
    for model in sorted({v["model"] for v in summary["methods"].values() if "model" in v}):
        for repeat in (1, 2):
            a, b = f"{model}:tools:repeat-{repeat}", f"{model}:direct:repeat-{repeat}"
            if a not in report["methods"] or b not in report["methods"]:
                continue
            rows_a = {r["case_id"]: r for r in report["methods"][a]["rows"]}
            rows_b = {r["case_id"]: r for r in report["methods"][b]["rows"]}
            if rows_a.keys() != rows_b.keys():
                raise ValueError("Paired methods have unequal cases")
            cells = {name: [] for name in ("both_correct", "tools_only_correct", "direct_only_correct", "neither_correct")}
            for cid, row in rows_a.items():
                x, y = row["contract_correct"], rows_b[cid]["contract_correct"]
                key = "both_correct" if x and y else "tools_only_correct" if x else "direct_only_correct" if y else "neither_correct"
                cells[key].append(cid)
            summary["paired_comparisons"].append({"model": model, "repeat": repeat, "cases": len(rows_a),
                                                   "counts": {k: len(v) for k, v in cells.items()}, "case_ids": cells})
        for mode in ("tools", "direct"):
            names = [f"{model}:{mode}:repeat-{r}" for r in (1, 2)]
            if not all(name in summary["methods"] for name in names):
                continue
            first, second = [{r["case_id"]: (r["execution"], r["decision"]) for r in summary["methods"][name]["runs"]} for name in names]
            changes = [cid for cid in first if first[cid] != second[cid]]
            summary["repeat_agreement"].append({"model": model, "mode": mode, "same_decision_and_execution": len(first)-len(changes), "cases": len(first), "changed_case_ids": changes})
    return summary


def figure(summary, target):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "factor-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    groups = [("naive", ["naive"]), ("bounded", ["bounded"])]
    models = []
    for name, info in summary["methods"].items():
        if "model" in info and info["model"] not in models:
            models.append(info["model"])
    for model in models:
        for mode in ("tools", "direct"):
            names = [n for n, m in summary["methods"].items() if m.get("model") == model and m.get("mode") == mode]
            groups.append((f"{model}\n{mode}", names))
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 8), sharey=True)
    fig.subplots_adjust(left=.20, right=.984, top=.77, bottom=.22, wspace=.24)
    specs = [("completed", "A  Completed"), ("contract_correct", "B  Contract correct"),
             ("wrong_decisive", "C  Wrong decisive"), ("required_abstention_recall", "D  Required ambiguity")]
    for ax, (metric, title) in zip(axes, specs):
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=17)
        ax.set_xlim(-1, 19); ax.set_xticks([0, 6, 12, 18]); ax.set_xlabel("Case count", labelpad=12)
        ax.xaxis.grid(True, color="#DEE4E8", linewidth=.65)
        ax.tick_params(length=0, pad=7)
        for spine in ax.spines.values(): spine.set_visible(False)
        for y, (_, names) in enumerate(groups):
            ax.axhspan(y-.42, y+.42, color="#F5F7F8" if y % 2 == 0 else "white", zorder=-2)
            for name in names:
                info = summary["methods"][name]; value = info["metrics"][metric]
                repeat = info.get("repeat")
                color, marker, offset = ("#4C5D68", "D", 0) if repeat is None else ("#0072B2", "o", -.15) if repeat == 1 else ("#D55E00", "s", .15)
                x, total = value["successes"], value["total"]
                if total == 0:
                    ax.text(6, y+offset, "Not applicable", fontsize=8, color=color); continue
                ax.plot(x, y+offset, marker=marker, markersize=6.5, color=color, markeredgecolor="white", markeredgewidth=.6, zorder=3)
                dx, ha = (-6, "right") if x >= 14 else (6, "left")
                ax.annotate(f"{x}/{total}", (x, y+offset), xytext=(dx, 0), textcoords="offset points", fontsize=9, ha=ha, va="center", color=color)
    labels = [label.replace("naive", "Naive conventional").replace("bounded", "Bounded conventional").replace("factor-qwen35-9b", "Qwen 3.5 9B").replace("factor-nemotron4b", "Nemotron 4B") for label, _ in groups]
    axes[0].set_yticks(range(len(groups)), labels); axes[0].set_ylim(len(groups)-.48, -.50)
    for ax in axes[1:]: ax.tick_params(labelleft=False)
    fig.suptitle("Explicit motion task: local agents and conventional methods", x=.035, y=.962, ha="left", fontsize=19, fontweight="bold", color="#152935")
    fig.text(.035, .914, "18 observations / 12 independent experiment groups per repeat  ·  synthetic development only", fontsize=12, color="#44545E")
    fig.text(.035, .878, "Same public scientific rules and equal maximum budgets for local direct/tool modes; every assigned attempt retained.", fontsize=10.5, color="#44545E")
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=6.5, label=l) for c,m,l in [("#4C5D68","D","Conventional"),("#0072B2","o","Repeat 1"),("#D55E00","s","Repeat 2")]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.03,.85), ncols=3, frameon=False)
    fig.text(.035,.136,"A/B: higher counts are desirable. C: lower is desirable, but failures and abstention can lower this count; inspect A and D too.",fontsize=10,color="#253B47")
    fig.text(.035,.10,"D counts required ambiguous answers recovered, with its eligible denominator printed. Repeats do not create new independent cases.",fontsize=10,color="#253B47")
    fig.text(.035,.064,"Exact descriptive counts; no sampling confidence intervals. Contract correctness is conditional on supplied diagnostic assumptions.",fontsize=9.7,color="#526773")
    fig.text(.035,.031,"Direct mode has no structured interval field; these are decision comparisons, not a fair numerical-reconstruction comparison.",fontsize=9.7,color="#526773")
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".png", ".pdf"): fig.savefig(target.with_suffix(extension), dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=Path("reports/agent-v2-comparison/comparison.json"))
    parser.add_argument("--audit", type=Path, default=Path("reports/agent-v2-trace-audit.json"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/agent-v2-comparison"))
    args = parser.parse_args()
    output = summarize(args.comparison)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(output, indent=2) + "\n")
    figure(output, args.figure)
    print(json.dumps({"methods": len(output["methods"]), "audited_local_runs": sum(len(m.get("runs", [])) for m in output["methods"].values()), "audit": str(args.audit), "figure": str(args.figure)}, indent=2))
