"""Plot frozen conventional results; does not regenerate or rescore observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "factor-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


REGIMES = (
    ("clean", "Clean"),
    ("bounded_drift", "Bounded drift"),
    ("ambiguous", "Ambiguous, no reference"),
    ("reference", "Calibrated reference"),
    ("reference_noisy", "Noisy reference"),
    ("unseen_drift", "Unseen drift"),
    ("reference_mismatch", "Reference mismatch"),
)
METHODS = (("naive", "Naive", "#D55E00", "o", -.16),
           ("bounded", "Bounded uncertainty", "#0072B2", "s", .16))


def plot(source, output):
    source, output = Path(source), Path(output)
    data = json.loads(source.read_text())
    if data.get("schema_version") != "factor-method-comparison/1":
        raise ValueError("Expected the preserved v1 comparison schema")
    methods = data["methods"]
    if not {"naive", "bounded"} <= methods.keys():
        raise ValueError("Both conventional methods are required")
    for regime, _ in REGIMES:
        if methods["naive"]["regimes"][regime]["cases"] != methods["bounded"]["regimes"][regime]["cases"]:
            raise ValueError("Methods must have the same assigned case counts")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10.5, "axes.titlesize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 8.9), sharey=True,
                             gridspec_kw={"wspace": .16})
    fig.subplots_adjust(left=.205, right=.985, top=.77, bottom=.22)
    total = methods["bounded"]["overall"]["cases"]
    groups = methods["bounded"]["overall"]["contract_correct"]["experiment_all_pass"]["total"]
    fig.suptitle("Motion and diagnostic drift: coverage and false confidence", x=.04,
                 y=.963, ha="left", fontsize=19, fontweight="bold", color="#152935")
    fig.text(.04, .918, f"Synthetic development  ·  {total} observations / {groups} independent experiment groups",
             fontsize=12, color="#44545E")
    fig.text(.04, .887, "Fixed 100 ns displacement window  ·  10 nm decision resolution  ·  truth is exact only within this simulator",
             fontsize=10.5, color="#44545E")
    legend = [Line2D([], [], linestyle="none", color=color, marker=marker,
                     markersize=7, label=label) for _, label, color, marker, _ in METHODS]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.035, .862),
               frameon=False, ncols=2, columnspacing=2, handletextpad=.4, fontsize=11)
    specs = (("coverage", "A  Interval coverage", "Truth inside emitted finite interval (%)"),
             ("wrong_decisive", "B  Wrong decisive answers", "Physically wrong decision / all cases (%)"),
             ("abstained", "C  Scientific abstention", "Ambiguous answer / all cases (%)"))
    for ax, (metric, title, xlabel) in zip(axes, specs):
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel(xlabel, labelpad=12)
        ax.set_title(title, loc="left", pad=19, fontweight="bold")
        ax.xaxis.grid(True, color="#E1E5E8", linewidth=.65, zorder=0)
        ax.tick_params(axis="both", length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axhspan(4.55, 6.48, color="#FDECE6", zorder=-3)
        ax.axhspan(1.57, 2.43, color="#F0F3F5", zorder=-2)
        for position, (regime, _) in enumerate(REGIMES):
            for method, _, color, marker, offset in METHODS:
                result = methods[method]["regimes"][regime]
                value = result[metric]
                y = position + offset
                if value["total"] == 0:
                    if value["rate"] is not None:
                        raise ValueError("A missing denominator must have null rate")
                    ax.text(31, y, "No finite interval", va="center", ha="left",
                            color=color, fontsize=9.4, fontstyle="italic")
                    continue
                rate = 100 * value["successes"] / value["total"]
                if abs(rate / 100 - value["rate"]) > 1e-12:
                    raise ValueError("Stored rate and counts disagree")
                ax.plot(rate, y, marker=marker, markersize=6.5, markeredgewidth=.7,
                        markeredgecolor="white", color=color, zorder=3)
                dx, align = (-6, "right") if rate >= 80 else (6, "left")
                ax.annotate(f'{value["successes"]}/{value["total"]}', (rate, y),
                            xytext=(dx, 0), textcoords="offset points", va="center",
                            ha=align, color=color, fontsize=9.4)
        if metric == "coverage":
            ax.axvline(95, color="#8496A3", linestyle=(0, (3, 3)), linewidth=.8, zorder=1)
            ax.text(.98, 1.013, "95% nominal guide*", transform=ax.transAxes,
                    fontsize=8.7, color="#526773", ha="right")
    labels = []
    for regime, label in REGIMES:
        r = methods["bounded"]["regimes"][regime]
        paired = r["contract_correct"]["experiment_all_pass"]["total"]
        count = f'n = {r["cases"]}'
        if paired != r["cases"]:
            count += f"; {paired} paired groups"
        labels.append(f"{label}\n{count}")
    axes[0].set_yticks(range(len(REGIMES)), labels)
    axes[0].set_ylim(6.5, -.55)
    axes[0].tick_params(axis="y", labelsize=10.4)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.text(.04, .138, "Counts label every point. Coverage excludes cases with no finite interval; missing coverage is not plotted as zero.",
             fontsize=10, color="#253B47")
    fig.text(.04, .107, "Gray row: structurally ambiguous information. Peach rows: undeclared model mismatch. Zero wrong decisions may reflect abstention.",
             fontsize=10, color="#253B47")
    fig.text(.04, .076, "*95% refers to the Gaussian component, conditional on supplied noise and nuisance bounds. Enlarged sets may be conservative.",
             fontsize=9.6, color="#526773")
    fig.text(.04, .045, "Descriptive paired-case proportions; no sampling confidence intervals shown. The ambiguous and calibrated-reference tracks share experiment groups.",
             fontsize=9.6, color="#526773")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=220)
    plt.close(fig)
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "contract_hash": methods["bounded"]["contract_hash"],
                  "methods": [m[0] for m in METHODS], "observations": total,
                  "independent_experiment_groups": groups,
                  "display": "Counts and descriptive rates; no marginal confidence intervals; missing coverage remains missing.",
                  "matplotlib_version": matplotlib.__version__}
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/motion-development/comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/baseline-evaluation"))
    args = parser.parse_args()
    print(json.dumps(plot(args.source, args.output), indent=2))
