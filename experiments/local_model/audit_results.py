"""Independent, standard-library re-scoring of saved predictions, without inference."""
import json
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    truth = [json.loads(line) for line in (ROOT/"frozen_dataset/test.jsonl").read_text().splitlines()]
    before = json.loads((ROOT/"posttrain_run/test_predictions_before.json").read_text())
    after = json.loads((ROOT/"posttrain_run/test_predictions_after.json").read_text())
    key = {r["id"]: r for r in truth}
    a, b = [{r["id"]: r for r in records} for records in (before, after)]
    assert len(a) == len(before) == len(b) == len(after) == len(key)
    assert set(a) == set(b) == set(key)
    for case_id, case in key.items():
        assert a[case_id]["label"] == b[case_id]["label"] == case["label"]
    family_scores, regressions, remaining_errors = [], [], []
    for family in sorted({r["family"] for r in truth}):
        ids = [r["id"] for r in truth if r["family"] == family]
        old = sum(a[i]["prediction"] == key[i]["label"] for i in ids)
        new = sum(b[i]["prediction"] == key[i]["label"] for i in ids)
        family_scores.append({"family": family, "n_dependent_variants": len(ids),
                              "before_correct": old, "after_correct": new, "difference": (new-old)/len(ids)})
    for i, case in key.items():
        if b[i]["prediction"] != case["label"]:
            remaining_errors.append({"id": i, "family": case["family"], "expected": case["label"],
                                     "after": b[i]["prediction"]})
            if a[i]["prediction"] == case["label"]:
                regressions.append({"id": i, "expected": case["label"], "before": a[i]["prediction"],
                                    "after": b[i]["prediction"], "text": case["text"]})
    result = {"audit": "Saved predictions re-scored independently; no model run or retuning",
              "family_scores": family_scores, "n_family_groups": len(family_scores),
              "all_family_means_improved": all(r["difference"] > 0 for r in family_scores),
              "regressions": regressions, "remaining_errors": remaining_errors,
              "uncertainty": "Four authored families do not support a reliable population or seed-variability claim; do not treat48 variants as48 independent experiments"}
    (ROOT/"posttrain_run/independent_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "remaining_errors"}, indent=2))


if __name__ == "__main__":
    main()
