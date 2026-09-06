"""Hands-on PyTorch module for Factor's local evidence classifier.

Run the workspace launcher: ./PyTorch.command
This optional module requires the packages in .venv-local. No downloads or APIs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

import torch
from torch import nn


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_device(requested="auto"):
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available.get(requested, False):
        raise ValueError(f"Device unavailable: {requested}")
    return requested


class ProvenanceModule(nn.Module):
    """Real torch.nn.Module: local Qwen + a saved or fresh trainable LoRA adapter.

    forward() returns the causal LM loss/logits. batch() masks prompt and padding
    tokens so training learns the requested label rather than copying the input.
    predict() generates a label; it does not establish that the claim is true.
    """
    def __init__(self, workspace, *, device="auto", fresh_adapter=False, adapter=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model

        self.workspace = Path(workspace).resolve()
        assets = self.workspace / "experiments/local_model"
        manifest = json.loads((assets / "frozen_dataset/manifest.json").read_text())
        self.system, self.labels = manifest["system"], manifest["labels"]
        selected_device = select_device(device)
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        base = assets / "base_model"
        if not (base / "model.safetensors").is_file():
            raise FileNotFoundError("Local base weights missing. See docs/LOCAL_MODELS.md.")
        self.base_sha256 = sha256(base / "model.safetensors")
        expected = json.loads((assets / "base_model_files.json").read_text())
        if self.base_sha256 != expected["model.safetensors"]["sha256"]:
            raise ValueError("Base weight hash differs from the recorded model")
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, use_safetensors=True,
            dtype=torch.float32 if selected_device == "cpu" else torch.bfloat16,
            attn_implementation="sdpa")
        if fresh_adapter:
            if adapter is not None:
                raise ValueError("Choose a fresh adapter or an existing adapter, not both")
            self.lm = get_peft_model(model, LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0,
                target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        else:
            default = assets / "posttrain_run/selected_adapter"
            adapter = Path(adapter).resolve() if adapter else default
            digest = sha256(adapter / "adapter_model.safetensors")
            if adapter == default:
                recorded = json.loads((assets / "posttrain_run/manifest.json").read_text())
                if digest != recorded["adapter_sha256"]:
                    raise ValueError("Saved adapter hash differs from the recorded experiment")
            self.lm = PeftModel.from_pretrained(model, adapter, local_files_only=True,
                                               is_trainable=False)
        self.to(selected_device)
        # Greedy decoding does not use sampling settings from the base config.
        self.lm.generation_config.temperature = None
        self.lm.generation_config.top_p = None
        self.lm.generation_config.top_k = None
        self.eval()

    @property
    def device_name(self):
        return str(self._device_anchor.device)

    def forward(self, **batch):
        return self.lm(**batch)

    def prompt_tokens(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a nonempty claim and its supporting record")
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system}, {"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(tokens) > 500:
            raise ValueError("Input is too long for this lab (500 prompt tokens); shorten it")
        return tokens

    def batch(self, rows):
        if not rows:
            raise ValueError("A training batch cannot be empty")
        encoded = []
        for row in rows:
            if row["label"] not in self.labels:
                raise ValueError("Unknown training label")
            prefix = self.prompt_tokens(row["text"])
            suffix = self.tokenizer.encode(row["label"], add_special_tokens=False)
            suffix += [self.tokenizer.eos_token_id]
            if len(prefix) + len(suffix) > 512:
                raise ValueError("Training sequence exceeds 512 tokens")
            encoded.append((prefix + suffix, [-100] * len(prefix) + suffix))
        width = max(len(tokens) for tokens, _ in encoded)
        return {
            "input_ids": torch.tensor([tokens + [self.tokenizer.pad_token_id] * (width-len(tokens))
                                        for tokens, _ in encoded], device=self.device_name),
            "labels": torch.tensor([labels + [-100] * (width-len(labels))
                                     for _, labels in encoded], device=self.device_name),
            "attention_mask": torch.tensor([[1] * len(tokens) + [0] * (width-len(tokens))
                                             for tokens, _ in encoded], device=self.device_name),
        }

    @torch.inference_mode()
    def predict(self, text):
        self.eval()
        ids = torch.tensor([self.prompt_tokens(text)], device=self.device_name)
        start = time.monotonic()
        output = self.lm.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  max_new_tokens=12, do_sample=False, use_cache=True,
                                  pad_token_id=self.tokenizer.pad_token_id)
        answer = self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return {"label": answer if answer in self.labels else None, "generated_text": answer,
                "seconds": round(time.monotonic()-start, 3),
                "scope": "Experimental provenance classification; not physical validation"}


def load_partition(workspace, name):
    # This learning lab deliberately never opens the old final test partition.
    if name not in {"train", "dev"}:
        raise ValueError("Only training and development partitions are available in this lab")
    folder = Path(workspace) / "experiments/local_model/frozen_dataset"
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / f"{name}.jsonl"
    if sha256(path) != manifest["partitions"][name]["sha256"]:
        raise ValueError(f"Recorded {name} partition changed")
    return [json.loads(line) for line in path.read_text().splitlines()]


@torch.inference_mode()
def development_loss(model, rows):
    model.eval()
    weighted_loss, label_tokens = 0.0, 0
    for offset in range(0, len(rows), 4):
        batch = model.batch(rows[offset:offset+4])
        count = int((batch["labels"][:, 1:] != -100).sum())
        weighted_loss += float(model(**batch, use_cache=False).loss) * count
        label_tokens += count
    return weighted_loss / label_tokens


def train_experiment(workspace, *, steps=8, device="auto", seed=42, out=None):
    if not 1 <= steps <= 100:
        raise ValueError("Choose 1–100 steps for this small learning lab")
    workspace = Path(workspace).resolve()
    destination = Path(out).resolve() if out else workspace / "runs/pytorch-lab" / uuid.uuid4().hex[:12]
    # Reserve a new directory before loading weights; never overwrite a previous run.
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    train, dev = load_partition(workspace, "train"), load_partition(workspace, "dev")
    random.Random(seed).shuffle(train)
    model = ProvenanceModule(workspace, device=device, fresh_adapter=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    before = development_loss(model, dev)
    print(f"Development loss before: {before:.4f}", flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=0.0005)
    history = []
    start = time.monotonic()
    for step in range(steps):
        model.train()
        offset = (step * 4) % len(train)
        batch = model.batch(train[offset:offset+4])
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss; stopping without saving a usable checkpoint")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
        print(f"Step {step+1}/{steps}: training loss {history[-1]:.4f}", flush=True)
    after = development_loss(model, dev)
    model.lm.save_pretrained(destination / "adapter", safe_serialization=True)
    report = {"scope": "Interactive development experiment; no final-test or physics-validity claim",
              "steps": steps, "seed": seed, "device": model.device_name, "torch": torch.__version__,
              "trainable_parameters": sum(p.numel() for p in parameters),
              "development_examples": len(dev), "development_loss_before": before,
              "development_loss_after": after, "training_losses": history,
              "training_and_final_dev_seconds": time.monotonic()-start,
              "base_sha256": model.base_sha256, "source_sha256": sha256(__file__),
              "partition_sha256": {name: sha256(workspace / f"experiments/local_model/frozen_dataset/{name}.jsonl")
                                   for name in ("train", "dev")},
              "adapter_sha256": sha256(destination / "adapter/adapter_model.safetensors")}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Development loss after: {after:.4f}\nSaved: {destination}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["interactive", "info", "predict", "train"], default="interactive")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--text", default="TARGET: ETI caused the fingers. Images show fingers, but no observation distinguishes ETI from competing mechanisms.")
    parser.add_argument("--adapter", type=Path, help="Use a locally saved adapter for prediction")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "train" and args.adapter is not None:
        parser.error("train starts a fresh adapter; --adapter is for prediction")
    if args.command == "info":
        print(json.dumps({"torch": torch.__version__, "device": select_device(args.device),
                          "model": "Qwen3-0.6B + LoRA", "workspace": str(args.workspace)}, indent=2))
    elif args.command == "train":
        train_experiment(args.workspace, steps=args.steps, device=args.device, out=args.out)
    else:
        print("Loading the local Qwen model and adapter…", flush=True)
        model = ProvenanceModule(args.workspace, device=args.device, adapter=args.adapter)
        print(f"Ready on {model.device_name}. Labels: {', '.join(model.labels)}", flush=True)
        if args.command == "predict":
            print(json.dumps(model.predict(args.text), indent=2))
            return
        print("Enter a record and TARGET claim. Type /example, /info, or /quit.", flush=True)
        while True:
            try:
                text = input("\nFactor PyTorch > ").strip()
                if text == "/quit":
                    break
                if text == "/info":
                    print(f"PyTorch {torch.__version__}; {model.device_name}; local inference only")
                    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
                    for name, weight in list(model.named_parameters())[:3]:
                        print(f"  {name}: {tuple(weight.shape)}, {weight.dtype}")
                    continue
                if text == "/example":
                    text = args.text
                    print(text)
                if text:
                    print(json.dumps(model.predict(text), indent=2))
            except (EOFError, KeyboardInterrupt):
                print("\nClosed.")
                break
            except ValueError as error:
                print(error)


if __name__ == "__main__":
    main()
