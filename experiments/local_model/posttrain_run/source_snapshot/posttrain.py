"""Actual PyTorch causal-language-model LoRA training; no classifier head.

Run once per fresh output directory. Hyperparameters and split hashes were
frozen before evaluation. Test outputs are produced only after dev selection.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import LABELS, SYSTEM

ROOT = Path(__file__).parent
DATA = ROOT/"frozen_dataset"
OUT = ROOT/"posttrain_run"
SEED = 20260905


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False))


def rows(split):
    return [json.loads(line) for line in (DATA/f"{split}.jsonl").read_text().splitlines()]


def metrics(records):
    correct = [r["prediction"] == r["label"] for r in records]
    return {"n": len(records), "accuracy": sum(correct)/len(records),
            "invalid_label_rate": sum(r["prediction"] not in LABELS for r in records)/len(records),
            "per_class": {label: {"n": sum(r["label"] == label for r in records),
                "accuracy": sum(r["prediction"] == label and r["label"] == label for r in records)
                    / sum(r["label"] == label for r in records)} for label in LABELS},
            "per_family": {family: sum(r["prediction"] == r["label"] for r in records if r["family"] == family)
                                / sum(r["family"] == family for r in records)
                           for family in sorted({r["family"] for r in records})}}


def main():
    if OUT.exists():
        raise ValueError("Refusing to overwrite a run or repeat its final test")
    OUT.mkdir()
    manifest = json.loads((DATA/"manifest.json").read_text())
    for split, info in manifest["partitions"].items():
        actual = hashlib.sha256((DATA/f"{split}.jsonl").read_bytes()).hexdigest()
        if actual != info["sha256"]:
            raise ValueError("Frozen dataset changed: " + split)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(ROOT/"base_model", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(ROOT/"base_model", local_files_only=True,
                torch_dtype=dtype, attn_implementation="sdpa").to(device)
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                    target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_manifest = {"base_model": json.loads((ROOT/"base_model_manifest.json").read_text()),
        "dataset_manifest_sha256": hashlib.sha256((DATA/"manifest.json").read_bytes()).hexdigest(),
        "training": manifest["planned_training"], "device": device, "dtype": str(dtype),
        "hardware": "Apple M4 Max; 14 physical CPU cores; 36GiB unified memory",
        "python": platform.python_version(), "packages": {x: importlib.metadata.version(x)
            for x in ["torch", "transformers", "peft", "tokenizers", "safetensors", "accelerate"]},
        "trainable_parameters": trainable, "total_parameters": sum(p.numel() for p in model.parameters()),
        "seed_variability": "One preregistered training seed; variability not measured",
        "task": "Generative evidence-provenance label prediction on authored synthetic vignettes",
        "model_input": "System rubric and text only; no id, family, split, expected label or evaluator access",
        "status": "training", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write("manifest.json", run_manifest)

    def prompt(text):
        return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
              {"role": "user", "content": text}], tokenize=True,
              add_generation_prompt=True, enable_thinking=False)

    def encode(row):
        prefix = prompt(row["text"])
        suffix = tokenizer.encode(row["label"], add_special_tokens=False) + [tokenizer.eos_token_id]
        return prefix+suffix, [-100]*len(prefix)+suffix

    def batch(examples):
        encoded = [encode(row) for row in examples]
        width = max(len(x[0]) for x in encoded)
        if width > 512:
            raise ValueError("Unexpected context length; refusing silent truncation")
        ids, labels, masks = [], [], []
        for tokens, target in encoded:
            pad = width-len(tokens)
            ids.append(tokens+[tokenizer.pad_token_id]*pad)
            labels.append(target+[-100]*pad)
            masks.append([1]*len(tokens)+[0]*pad)
        return {"input_ids": torch.tensor(ids, device=device),
                "attention_mask": torch.tensor(masks, device=device),
                "labels": torch.tensor(labels, device=device)}

    @torch.no_grad()
    def predict(examples):
        model.eval()
        outputs = []
        for row in examples:
            ids = torch.tensor([prompt(row["text"])], device=device)
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                use_cache=True)
            response = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
            outputs.append({"id": row["id"], "family": row["family"], "label": row["label"],
                            "prediction": response, "generated_tokens": generated.shape[1]-ids.shape[1]})
        return outputs

    @torch.no_grad()
    def dev_loss(examples):
        model.eval()
        values = [float(model(**batch(examples[i:i+8])).loss) for i in range(0, len(examples), 8)]
        return sum(values)/len(values)

    training, development = rows("train"), rows("dev")
    before_dev = predict(development)
    write("development_before.json", {"metrics": metrics(before_dev), "predictions": before_dev})
    print("BASELINE_DEV", json.dumps(metrics(before_dev)), flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0005)
    history, best = [], float("inf")
    for epoch in range(1, 4):
        shuffled = training.copy()
        random.Random(SEED+epoch).shuffle(shuffled)
        model.train()
        losses = []
        epoch_start = time.monotonic()
        for index in range(0, len(shuffled), 8):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch(shuffled[index:index+8])).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
            if (index//8+1) % 8 == 0:
                print("TRAIN", epoch, index//8+1, "loss", losses[-1], flush=True)
        loss_dev = dev_loss(development)
        record = {"epoch": epoch, "training_loss": sum(losses)/len(losses),
                  "development_label_token_nll": loss_dev, "elapsed_seconds": time.monotonic()-epoch_start}
        history.append(record)
        if loss_dev < best:
            best = loss_dev
            model.save_pretrained(OUT/"selected_adapter", safe_serialization=True)
            run_manifest["selected_epoch"] = epoch
        write("training_history.json", history)
        print("EPOCH", json.dumps(record), flush=True)

    # Reload only the checkpoint selected by development NLL, then freeze it.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    adapter = OUT/"selected_adapter"/"adapter_model.safetensors"
    set_peft_model_state_dict(model, load_file(adapter))
    run_manifest["adapter_sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    run_manifest["status"] = "selected_before_test"
    write("manifest.json", run_manifest)
    after_dev = predict(development)
    write("development_after.json", {"metrics": metrics(after_dev), "predictions": after_dev})

    # This is the sole final-test exposure. No training or selection follows.
    write("test_exposure.json", {"stage": "final_evaluation", "adapter_sha256": run_manifest["adapter_sha256"],
                               "policy": "No further tuning or test reuse as unseen data"})
    final = rows("test")
    with model.disable_adapter():
        before = predict(final)
    after = predict(final)
    write("test_predictions_before.json", before)
    write("test_predictions_after.json", after)
    a, b = metrics(before), metrics(after)
    improved = sum(x["prediction"] != x["label"] and y["prediction"] == y["label"] for x,y in zip(before,after))
    regressed = sum(x["prediction"] == x["label"] and y["prediction"] != y["label"] for x,y in zip(before,after))
    shuffled = [r["label"] for r in final]
    random.Random(4401).shuffle(shuffled)
    report = {"before": a, "after": b, "accuracy_difference": b["accuracy"]-a["accuracy"],
              "paired_improved": improved, "paired_regressed": regressed,
              "majority_baseline_accuracy": 0.25,
              "shuffled_truth_negative_control_accuracy": sum(r["prediction"] == label for r,label in zip(after,shuffled))/len(after),
              "elapsed_seconds": time.monotonic()-start,
              "conclusion_scope": "One-seed generative LoRA result on four held-out synthetic families; no independent real data, no physics or agent-transfer claim",
              "split_dependence": "48 vignettes but only4 held-out families; variants are dependent",
              "post_training_completed": True}
    write("metrics.json", report)
    run_manifest["status"] = "completed_final_test_exposed"
    write("manifest.json", run_manifest)
    print("FINAL", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
