"""Reporting functions for the synthetic development benchmark."""
import hashlib
import platform
from pathlib import Path
import numpy as np
import scipy
from .constants import VERSION, SCENARIOS
from .diagnostics import pdv_point, analyze
from .evaluation import score
from .simulation import simulate
from .io import write_json, write_jsonl


def plot_examples(examples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.3), layout="constrained")
    nominal = examples["nominal"][0]
    t = nominal["time_ns"]
    _, _, x, smooth = pdv_point(t, nominal["iq_real"], nominal["iq_imag"])
    ax[0, 0].plot(t, x, color="#b2c2cf", lw=0.8, label="Noisy phase-derived displacement")
    ax[0, 0].plot(t, smooth, color="#087e8b", lw=2, label="Smoothed estimate")
    ax[0, 0].set(xlabel="Recorded time (ns)", ylabel="Displacement (nm)", title="Prescribed-motion PDV-like example")
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].imshow(nominal["counts"] / 5000, aspect="auto", origin="lower",
                    extent=[-560, 560, 0, 1200], cmap="gray", vmin=0, vmax=1)
    ax[0, 1].set(xlabel="Transverse position (um)", ylabel="Axial position (um)", title="Synthetic single-energy transmission")
    for label, color in [("nominal", "#087e8b"), ("blurred", "#dd7f39")]:
        ob, truth = examples[label]
        naive, cal = analyze(ob)
        vals = [truth["mode_amplitude_um"], naive["mode_amplitude_um"]["mean"], cal["mode_amplitude_um"]["mean"]]
        ax[1, 0].plot([0, 1, 2], vals, "o-", color=color, label=label)
    ax[1, 0].set(xticks=[0, 1, 2], xticklabels=["Known truth", "Simple reduction", "PSF corrected"],
                 ylabel="Preselected mode amplitude (um)", title="Blur can hide a real perturbation")
    ax[1, 0].legend(fontsize=8)
    ob, _ = examples["pdv_dropout"]
    ax[1, 1].plot(ob["time_ns"], np.hypot(ob["iq_real"], ob["iq_imag"]), color="#71459b")
    ax[1, 1].axhline(0.3, color="#888888", ls="--", lw=1)
    ax[1, 1].set(xlabel="Recorded time (ns)", ylabel="Quadrature return amplitude", title="A lost return requires an abstention")
    fig.suptitle("Liner evaluation starter | Synthetic development examples only", fontsize=14, fontweight="bold")
    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def demo(out, n, base_seed):
    if n < 2:
        raise ValueError("Use at least two cases per scenario")
    out.mkdir(parents=True, exist_ok=True)
    truth, naive, calibrated, examples = [], [], [], {}
    for j, scenario in enumerate(SCENARIOS):
        for i in range(n):
            seed = base_seed + j * 100000 + i
            obs, actual = simulate(seed, scenario)
            a, b = analyze(obs)
            truth.append(actual)
            naive.append(a)
            calibrated.append(b)
            if i == 0:
                examples[scenario] = (obs, actual)
                np.savez_compressed(out / f"example_{scenario}.npz",
                                    **{k: v for k, v in obs.items() if isinstance(v, np.ndarray)})
                write_json(out / f"example_{scenario}_metadata.json",
                           {k: v for k, v in obs.items() if not isinstance(v, np.ndarray)})
        print(f"Completed {scenario}: {n} synthetic cases", flush=True)
    write_jsonl(out / "truth.jsonl", truth)
    write_jsonl(out / "predictions_simple.jsonl", naive)
    write_jsonl(out / "predictions_calibrated.jsonl", calibrated)
    report = {"status": "SYNTHETIC DEVELOPMENT DEMO; no real data or LLM evaluated",
              "simple": score(truth, naive), "calibrated": score(truth, calibrated)}
    write_json(out / "metrics.json", report)
    manifest = {"version": VERSION, "base_seed": base_seed, "cases_per_scenario": n,
                "scenarios": list(SCENARIOS), "split": "public_development_only",
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "reducer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "command": f"python liner_eval.py demo --out {out.name} --n {n} --seed {base_seed}",
                "limitations": ["prescribed motion, no MHD or melt", "baseband IQ, not raw PDV digitizer",
                                "known seeded mode, not blind spatial discovery", "single-energy uniform-cylinder opacity",
                                "noise propagation, not Bayesian posterior or SBC", "public truth and seed, not a hidden evaluation",
                                "no real data or trained language model", "omitted optical drift deliberately absent from reducer"]}
    write_json(out / "run_manifest.json", manifest)
    lines = ["# Synthetic diagnostic evaluation", "", report["status"], "",
             f"{len(truth)} cases, {n} per scenario. Seed {base_seed}. No fit to experiment.", "",
             "| Scenario | Amplitude MAE simple / corrected (um) | Corrected amplitude interval coverage | Compression MAE (nm) | Compression interval coverage | PDV answer rate |",
             "| --- | --- | --- | --- | --- | --- |"]
    for scenario in SCENARIOS:
        a, b = report["simple"][scenario], report["calibrated"][scenario]
        c = b["compression_nm"]
        cmae = f"{c['mae']:.2f}" if c["n_answered"] else "abstain"
        ccov = f"{c['coverage90']:.0%}" if c["n_answered"] else "n/a"
        lines.append(f"| {scenario} | {a['mode_amplitude_um']['mae']:.2f} / {b['mode_amplitude_um']['mae']:.2f} | {b['mode_amplitude_um']['coverage90']:.0%} | {cmae} | {ccov} | {c['answer_rate']:.0%} |")
    lines += ["", "Coverage is empirical, conditional on answering. Most intervals are approximate 90% intervals; low-significance amplitude results use conservative 90% upper bounds and are separately counted in metrics.json. Wilson intervals and interval widths are in the JSON scorecard. These are not pass/fail certification thresholds.",
              "", "The optical-drift scenario adds an unmodeled optical-path contribution. Any resulting coverage failure demonstrates a limitation of the reconstruction, not an instability discovery. The mean-displacement minimum is a kinematic turnaround, never a melt label.",
              "", "Raw examples and calibration metadata are included for six cases. Regenerate all measurements from the manifest. The generator and inverse share ideal material assumptions even though the radiographic integrations differ."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    plot_examples(examples, out / "diagnostic_examples.png")
    print(f"Wrote {len(truth)} case evaluations to {out}")
