"""Load the experimental generative provenance adapter as a local function.

Training itself is the frozen experiment in experiments/local_model/posttrain.py.
Optional Torch/Transformers/PEFT imports occur only when this model is created.
This is a task-specific label generator, not a general scientific reasoner.
"""
import hashlib
import json
from pathlib import Path
import time


class LocalProvenanceModel:
    def __init__(self, base_model_path, adapter_path, dataset_manifest_path, *,
                 device="cpu", expected_adapter_sha256=None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, device
        if device not in {"cpu", "mps"}:
            raise ValueError("This local adapter supports cpu or mps")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        adapter = Path(adapter_path)/"adapter_model.safetensors"
        self.adapter_sha256 = hashlib.sha256(adapter.read_bytes()).hexdigest()
        if expected_adapter_sha256 and self.adapter_sha256 != expected_adapter_sha256:
            raise ValueError("Adapter checkpoint hash does not match")
        manifest = json.loads(Path(dataset_manifest_path).read_text())
        self.system, self.labels = manifest["system"], manifest["labels"]
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True,
            use_safetensors=True, torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
            attn_implementation="sdpa").to(device)
        self.model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True,
                                               is_trainable=False).eval()

    def classify(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A nonempty evidence vignette is required")
        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": self.system}, {"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(prompt) > 512:
            raise ValueError("Input exceeds the evaluated context regime; refusing truncation")
        tokens = self.torch.tensor([prompt], device=self.device)
        start = time.monotonic()
        with self.torch.inference_mode():
            result = self.model.generate(input_ids=tokens, attention_mask=self.torch.ones_like(tokens),
                max_new_tokens=12, do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id)
        answer = self.tokenizer.decode(result[0, len(prompt):], skip_special_tokens=True).strip()
        return {"label": answer if answer in self.labels else None, "text": answer,
                "status": "experimental_label" if answer in self.labels else "invalid_label_abstain",
                "elapsed_seconds": time.monotonic()-start, "adapter_sha256": self.adapter_sha256,
                "scope": "Authored evidence-provenance task; no physics-validation claim"}
