"""Public development tasks and a separate deterministic answer grader."""
import json
import math
from pathlib import Path

def cases():
    mu0 = 4 * math.pi * 1e-7
    return [
        {"case_id": "A01", "category": "units_and_physics", "kind": "number", "unit": "Pa", "relative_tolerance": 1e-6,
         "prompt": "For a long cylindrical conductor, use B=mu0*I/(2*pi*R) and p=B^2/(2*mu0). I=1.0e6 A, R=0.5e-3 m, mu0=4*pi*1e-7 H/m. Compute the exterior azimuthal magnetic pressure in Pa; neglect all other fields.",
         "expected": mu0 * 1e12 / (8*math.pi**2*(0.5e-3)**2),
         "rationale": "Radius must be in metres and output is pressure, not magnetic field. This is an ideal long-cylinder calculation."},
        {"case_id": "A02", "category": "units_and_physics", "kind": "number", "unit": "m", "relative_tolerance": 1e-6,
         "prompt": "Use the explicitly defined transient diffusion length ell=sqrt(eta*t/mu0), not the sinusoidal skin-depth convention. eta=4e-8 ohm m; t=100 ns; mu0=4*pi*1e-7 H/m. Return ell in m.",
         "expected": math.sqrt(4e-8*100e-9/mu0),
         "rationale": "Convert nanoseconds to seconds. eta denotes electrical resistivity, not magnetic diffusivity."},
        {"case_id": "A03", "category": "units_and_physics", "kind": "number", "unit": "s", "relative_tolerance": 1e-6,
         "prompt": "A uniform idealized sample is already at melt temperature. With fixed density rho=2700 kg/m^3, latent heat L=397000 J/kg, resistivity eta=4e-8 ohm m and uniform fixed J=1e12 A/m^2, neglect work and all losses. Use dt=rho*L/(eta*J^2). Return the latent-heat absorption duration in seconds. This is an analytic toy, not a rod prediction.",
         "expected": 2700*397000/(4e-8*1e24),
         "rationale": "Latent heat per unit volume divided by volumetric Joule power has units of time."},
        {"case_id": "A04", "category": "diagnostic_inversion", "kind": "number", "unit": "kg/m^2", "relative_tolerance": 1e-6,
         "prompt": "An ideal monochromatic transmission is 0.50, with known constant mass attenuation coefficient 0.70 m^2/kg and no blur, scatter or self-emission. Return the line-integrated areal mass in kg/m^2 using T=exp(-kappa*Sigma). No thickness is supplied.",
         "expected": -math.log(0.5)/0.7,
         "rationale": "Transmission constrains areal mass under these assumptions; bulk density is not separately identified."},
        {"case_id": "A05", "category": "uncertainty", "kind": "number", "unit": "ns", "relative_tolerance": 1e-6,
         "prompt": "Two Gaussian time estimates each have standard deviation 2 ns and correlation 0.75. What is the standard deviation of their difference, in ns? Include covariance.",
         "expected": math.sqrt(4+4-2*0.75*2*2),
         "rationale": "A positive shared timing component partly cancels in the difference."},
        {"case_id": "A06", "category": "phase_evidence", "kind": "label",
         "prompt": "A measured PDV velocity trace is interpreted through a chosen 1D MHD/EOS calculation to infer a 4.5 ns melt interval. No independent phase diagnostic is present. Classify the phase interval. Allowed answers: direct_phase_measurement, model_assisted_inference, pure_simulator_output.",
         "expected": "model_assisted_inference",
         "rationale": "Measured motion constrains a phase inference but is not an independent direct phase measurement."},
        {"case_id": "A07", "category": "cross_shot_reasoning", "kind": "label",
         "prompt": "PDV from one rod campaign and a finger radiograph from a different campaign have similar diameters. There is no shared timing record or matched drive/coating calibration. Has same-shot temporal ordering between melt and finger birth been identified? Allowed answers: identified, not_identified.",
         "expected": "not_identified",
         "rationale": "Cross-campaign similarity motivates a hypothesis but does not create a simultaneous timing measurement."},
        {"case_id": "A08", "category": "diagnostic_identifiability", "kind": "label",
         "prompt": "The only optical signal is a real cosine of Doppler phase with zero known frequency offset and no quadrature reference. Can its spectrum alone identify the sign of a constant line-of-sight velocity? Allowed answers: identified, not_identified.",
         "expected": "not_identified",
         "rationale": "The real zero-offset cosine is invariant to changing the phase sign."},
        {"case_id": "A09", "category": "mechanism_abstention", "kind": "label",
         "prompt": "One radiograph shows finger-shaped material. There is no motion history, seed map, acceleration history, phase constraint, or diagnostic forward model. Which mechanism is established? Allowed answers: classical_ETI, MRT, boundary_feedback, unresolved.",
         "expected": "unresolved",
         "rationale": "Morphology by itself is insufficient to assign a unique causal mechanism."},
        {"case_id": "A10", "category": "leakage", "kind": "label",
         "prompt": "Frames, crops and alternative reductions share the same shots, and shots share campaign-specific calibration. For a new-campaign generalization test, what is the appropriate grouping unit? Allowed answers: pixel, frame, campaign_and_lineage.",
         "expected": "campaign_and_lineage",
         "rationale": "Related frames and reductions must not appear on both sides of a generalization test."},
        {"case_id": "A11", "category": "uncertainty", "kind": "label",
         "prompt": "A model answers 10 out of 100 cases. Nine of its ten 90% intervals cover truth. Does this alone establish reliable 90% performance across all 100 cases? Allowed answers: established, not_established.",
         "expected": "not_established",
         "rationale": "Conditional coverage needs its answer rate, interval width, case mix and finite-sample uncertainty."},
        {"case_id": "A12", "category": "experiment_design", "kind": "label",
         "prompt": "In this explicitly illustrative design screen, utility is expected information gain times usable-diagnostic probability divided by cost. A: 1.0 nat, probability 0.2, cost 1. B: 0.7 nat, probability 0.9, cost 1. C: 1.5 nat, probability 0.8, cost 3. Which feasible candidate maximizes this score? Allowed answers: A, B, C.",
         "expected": "B",
         "rationale": "Scores are 0.20, 0.63 and 0.40. Largest raw information gain need not give the best usable result per cost."},
    ]


def export(out):
    out.mkdir(parents=True, exist_ok=True)
    cs = cases()
    public = [{k: c[k] for k in ("case_id", "category", "kind", "prompt")}
              | ({"unit": c["unit"]} if "unit" in c else {}) for c in cs]
    for name, content in [("candidate_inputs.json", public), ("evaluator_key.json", cs)]:
        (out/name).write_text(json.dumps(content, indent=2, allow_nan=False)+"\n")
    template = [{"case_id": c["case_id"], "answer": None,
                 "unit": c.get("unit"), "explanation": "", "evidence_ids": [], "tool_artifacts": []} for c in cs]
    (out/"candidate_output_template.json").write_text(json.dumps(template, indent=2)+"\n")
    print(f"Exported {len(cs)} public example cases. Key must be kept outside candidate workspace.")


def score_answers(key, predictions):
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a JSON array")
    ids = [c["case_id"] for c in key]
    pids = [p["case_id"] for p in predictions]
    if len(set(ids)) != len(ids) or len(set(pids)) != len(pids) or set(ids) != set(pids):
        raise ValueError("Exactly one response per case is required; null is an abstention")
    mapped = {p["case_id"]: p for p in predictions}
    results = []
    for c in key:
        p = mapped[c["case_id"]]
        answer = p.get("answer")
        correct = False
        if c["kind"] == "number":
            if isinstance(answer, (int, float)) and not isinstance(answer, bool) and math.isfinite(answer):
                correct = p.get("unit") == c["unit"] and math.isclose(answer, c["expected"], rel_tol=c["relative_tolerance"], abs_tol=0)
        else:
            correct = isinstance(answer, str) and answer == c["expected"]
        results.append({"case_id": c["case_id"], "category": c["category"], "correct": correct, "abstained": answer is None})
    return {"status": "Automatic label/number checks only; explanations, sources and tool lineage require separate review",
            "n_cases": len(results), "n_correct": sum(r["correct"] for r in results),
            "accuracy_all_cases": sum(r["correct"] for r in results)/len(results),
            "answer_rate": 1-sum(r["abstained"] for r in results)/len(results), "results": results}
