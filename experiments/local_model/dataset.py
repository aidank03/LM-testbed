"""Synthetic provenance classification, with family-disjoint frozen partitions.

Labels describe the evidence explicitly stipulated by each vignette. They are
not labels of real physical events. Family variants are statistically dependent.
"""
import hashlib
import json
from pathlib import Path
import random

LABELS = ["recorded", "reconstructed", "model_inferred", "unresolved"]
SYSTEM = ("Classify the epistemic status of the TARGET claim using only the supplied record. "
          "recorded: stored instrument signal; reconstructed: quantity extracted from signals by calibration or reduction; "
          "model_inferred: physical state inferred by matching data to a chosen physical model; "
          "unresolved: causal or physical claim not distinguished by the available evidence. "
          "A caption or author's assertion cannot upgrade evidence. Return exactly one label: " + ", ".join(LABELS) + ".")

# Each family has distinct wording and a distinct measurement setting. Numeric
# variants never cross partitions. All four labels occur in every family.
FAMILIES = [
 ("train", "pdv", "quadrature samples", "radial displacement", "a liquid fraction", "electrothermal seeding"),
 ("train", "radiography", "detector pixel counts", "areal mass", "a density profile", "Rayleigh-Taylor growth"),
 ("train", "spectroscopy", "spectrometer intensities", "a line intensity ratio", "an electron temperature", "turbulent heating"),
 ("train", "electrical", "digitizer voltages", "a current waveform", "a resistivity history", "a plasma bridge"),
 ("train", "interferometry", "camera fringe pixels", "a phase shift", "an electron density profile", "an ionization front"),
 ("train", "thermography", "infrared detector values", "a calibrated brightness", "a surface temperature", "a hot-spot feedback loop"),
 ("dev", "acoustics", "microphone voltage samples", "an arrival delay", "an elastic modulus", "microcrack nucleation"),
 ("dev", "diffraction", "diffraction pixel counts", "a fitted peak position", "a crystal phase fraction", "a defect-mediated transition"),
 ("test", "strain", "bridge output voltages", "a strain history", "a plastic strain fraction", "local shear banding"),
 ("test", "neutrons", "scintillator pulse records", "a time-of-flight distribution", "an ion temperature", "beam-target fusion"),
 ("test", "reflectometry", "returned optical intensities", "a reflectance curve", "a layer thickness", "surface delamination"),
 ("test", "velocimetry", "heterodyne voltage samples", "a speed trajectory", "a pressure history", "shock-induced damage"),
]

FORMATS = [
 "The archive stores {raw}. A reduction computes {rec} from that record. Matching {rec} with an assumed constitutive model yields {inf}. The team proposes {hyp}, but alternatives remain compatible. TARGET: {target}.",
 "TARGET: {target}. Available evidence: {raw} were saved by the instrument; calibration and signal processing produced {rec}; a simulation fit to those estimates supplied {inf}. No measurement separates {hyp} from competing explanations.",
 "A report lists {raw} as the detector output and {rec} as its numerical reduction. It obtains {inf} only through a model fit. The caption claims {hyp} is proven, although no discriminating data were collected. TARGET: {target}.",
 "In this experiment, {raw} exist in the acquisition file. Analysts extracted {rec}. A selected physical model, constrained by that extraction, predicts {inf}. The attribution to {hyp} is still conjectural. TARGET: {target}.",
 "Instrument output: {raw}. Derived diagnostic result: {rec}. Fitting a chosen simulation to this result supports {inf} conditionally. Proposed cause: {hyp}; it has not been isolated from alternatives. TARGET: {target}.",
 "The instrument saved {raw}. Its processing pipeline calculated {rec}. The estimate of {inf} depends on a physical model fitted to the pipeline output. The proposed {hyp} lacks a test against rivals. TARGET: {target}.",
 "Classify {target}. The saved detector trace contains {raw}; processing it gives {rec}. Researchers matched a mechanical model to those estimates to obtain {inf}. A claim of {hyp} has no discriminating evidence.",
 "A headline says 'direct proof', but the chain is: stored {raw}; signal-derived {rec}; {inf} obtained by fitting a material model; {hyp} suggested without excluding alternatives. TARGET: {target}.",
 "TARGET: {target}. Before any analysis the recording system saved {raw}. Later a conversion procedure yielded {rec}. Researchers then conditioned a constitutive simulation on that estimate to deduce {inf}. They also speculated about {hyp}, with other causes still possible.",
 "The proposed {hyp} has not been distinguished from rivals. The inference of {inf} requires fitting a physical model to {rec}. That latter quantity came from reducing {raw}, the instrument's stored output. TARGET: {target}.",
 "What status belongs to {target}? The acquired file contains {raw}. A numerical diagnostic inversion produces {rec}; agreement with a selected material simulation suggests {inf}. A confident caption attributes everything to {hyp}, unsupported by an intervention or distinctive prediction.",
 "An analyst calls all four results 'measured'. In fact the acquisition captured {raw}; processing recovered {rec}; comparison with a constitutive calculation inferred {inf}; and {hyp} was merely conjectured. TARGET: {target}.",
]


def build():
    partitions = {name: [] for name in ("train", "dev", "test")}
    for index, (split, family, raw, rec, inf, hyp) in enumerate(FAMILIES):
        count = 8 if split == "train" else 3
        for variant in range(count):
            descriptors = [raw, rec, inf, hyp]
            for label, target in zip(LABELS, descriptors):
                body = FORMATS[index].format(raw=raw, rec=rec, inf=inf, hyp=hyp, target=target)
                # Irrelevant metadata is balanced across labels and partitions.
                text = f"Synthetic record, replicate {variant+1}, acquisition time {17+variant*7} ns. " + body
                partitions[split].append({"id": f"{family}-{variant}-{label}", "family": family,
                                          "text": text, "label": label})
    rng = random.Random(20260905)
    for rows in partitions.values():
        rng.shuffle(rows)
    return partitions


def freeze(directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Do not overwrite a frozen dataset")
    directory.mkdir(parents=True)
    manifest = {"task": "synthetic_evidence_provenance_v1", "system": SYSTEM,
                "labels": LABELS, "split_unit": "measurement_and_wording_family",
                "seed": 20260905, "status": "FROZEN_BEFORE_MODEL_EVALUATION",
                "limitations": "Author-constructed labels; no independent annotation; family variants are dependent; public after this run.",
                "planned_training": {"base": "Qwen/Qwen3-0.6B", "method": "causal-LM LoRA q_proj,v_proj rank8",
                     "seed": 20260905, "epochs": 3, "learning_rate": 0.0005, "batch_size": 8,
                     "selection": "lowest dev label-token NLL after each epoch; final test once after selection",
                     "decoding": "greedy, non-thinking, max_new_tokens12",
                     "success": "Positive paired test accuracy difference with per-family reporting; descriptive only, no certification"},
                "partitions": {}}
    all_text = set()
    for split, rows in build().items():
        body = "".join(json.dumps(row, sort_keys=True)+"\n" for row in rows)
        (directory/f"{split}.jsonl").write_text(body)
        assert not any(row["text"] in all_text for row in rows)
        all_text.update(row["text"] for row in rows)
        manifest["partitions"][split] = {"n": len(rows), "families": sorted({r["family"] for r in rows}),
            "sha256": hashlib.sha256(body.encode()).hexdigest()}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    print(json.dumps(freeze(Path(__file__).parent/"frozen_dataset"), indent=2))
