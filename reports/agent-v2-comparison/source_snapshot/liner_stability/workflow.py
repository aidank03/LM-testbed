"""One reproducible command from configuration to a scored evaluation card."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
from . import __version__
from .io import read_json, write_json, write_text
from .reporting import demo


def validate_config(config):
    allowed = {"schema_version","experiment_id","experiment_kind","cases_per_scenario","seed","notes"}
    if not isinstance(config,dict) or set(config)-allowed:
        raise ValueError("Unknown experiment configuration fields")
    if config.get("schema_version") != "1" or config.get("experiment_kind") != "prescribed_motion_and_corrugated_cylinder":
        raise ValueError("Unsupported experiment schema or model")
    if not isinstance(config.get("experiment_id"),str) or not config["experiment_id"].strip():
        raise ValueError("experiment_id is required")
    n, seed = config.get("cases_per_scenario"), config.get("seed")
    if not isinstance(n,int) or isinstance(n,bool) or not 2<=n<=1000:
        raise ValueError("cases_per_scenario must be an integer between 2 and 1000")
    if not isinstance(seed,int) or isinstance(seed,bool) or seed<0:
        raise ValueError("seed must be a nonnegative integer")


def evaluation_card(report, experiment_id):
    lines=[f"# Evaluation card: {experiment_id}","",
           "**Status: synthetic development run. No experimental data or LLM was evaluated.**", "",
           "## What was tested", "",
           "Prescribed compression/expansion and a known spatial mode were passed through synthetic PDV-like quadratures and radiographic measurements. The reducer received measurements and calibrations, with truth held separately for scoring.","",
           "## Results", "",
           "| Scenario | Cases | Amplitude MAE (um) | Compression MAE (nm) | Compression interval coverage | PDV answer rate |",
           "| --- | --- | --- | --- | --- | --- |"]
    for group, rows in report["calibrated"].items():
        if group=="all":continue
        a,c=rows["mode_amplitude_um"],rows["compression_nm"]
        ae=f"{a['mae']:.3f}" if a['n_answered'] else 'abstain'
        ce=f"{c['mae']:.3f}" if c['n_answered'] else 'abstain'
        coverage=f"{c['coverage90']:.1%}" if c['n_answered'] else 'n/a'
        lines.append(f"| {group} | {rows['n_cases']} | {ae} | {ce} | {coverage} | {c['answer_rate']:.0%} |")
    lines += ["","## Evaluation decision", "",
              "The implementation is a development benchmark. Test known optical-path drift and realistic raw PDV processing before using its uncertainty intervals on experiments. A lost optical return causes an explicit abstention. Public seeds and keys must not be represented as a hidden AI test.","",
              "## Next experiment or calibration", "",
              "Establish one real-shot record with raw measurements, source/return calibration, clock covariance and target metrology. First distinguish an optical-path change from material displacement. Preserve uncertainty in any model-assisted phase label.","",
              "Metrics, interval widths, finite-sample coverage bounds, predictions and run provenance accompany this card."]
    return "\n".join(lines)+"\n"


def run(config, out, override_n=None):
    config=dict(config)
    if override_n is not None:
        config["cases_per_scenario"]=override_n
    validate_config(config)
    out=Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Choose a new empty output directory; previous runs are preserved")
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"experiment_config.json",config)
    write_json(out/"status.json",{"status":"running","experiment_id":config["experiment_id"]})
    try:
        demo(out,config["cases_per_scenario"],config["seed"])
        report=read_json(out/"metrics.json")
        write_text(out/"EVALUATION_CARD.md",evaluation_card(report,config["experiment_id"]))
        manifest=read_json(out/"run_manifest.json")
        manifest.update(package_version=__version__,created_at_utc=datetime.now(timezone.utc).isoformat(),
                        experiment_id=config["experiment_id"],
                        package_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted(Path(__file__).parent.glob("*.py"))})
        manifest["command"]="liner-stability run --config experiment_config.json --out NEW_OUTPUT_DIRECTORY"
        manifest.pop("reducer_sha256",None)
        write_json(out/"run_manifest.json",manifest)
        write_json(out/"status.json",{"status":"completed","experiment_id":config["experiment_id"],
                                      "science_status":"synthetic_development_only"})
    except BaseException as exc:
        write_json(out/"status.json",{"status":"failed","error_type":type(exc).__name__})
        raise
    return report
