"""Small contract checks; no model download, generation, or training."""
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from factor.torch_lab import ProvenanceModule, load_partition, train_experiment
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch environment not installed")
class TorchLabChecks(unittest.TestCase):
    def setUp(self):
        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 2
            def apply_chat_template(self, messages, **kwargs):
                return [7] * len(messages[-1]["content"])
            def encode(self, label, **kwargs):
                return [9]
            def decode(self, tokens):
                return "piece"
        self.model = ProvenanceModule.__new__(ProvenanceModule)
        torch.nn.Module.__init__(self.model)
        self.model.tokenizer = Tokenizer()
        self.model.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.model.system = "fixture"
        self.model.labels = ["recorded"]

    def test_prompt_and_padding_are_excluded_from_loss(self):
        batch = self.model.batch([{"text": "ab", "label": "recorded"},
                                  {"text": "a", "label": "recorded"}])
        self.assertEqual(batch["labels"].tolist(), [[-100,-100,9,2], [-100,9,2,-100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1,1,1,1], [1,1,1,0]])
        self.assertEqual(batch["input_ids"].tolist(), [[7,7,9,2], [7,9,2,2]])

    def test_empty_claim_rejected(self):
        with self.assertRaises(ValueError): self.model.prompt_tokens(" ")

    def test_oversized_claim_is_not_silently_truncated(self):
        with self.assertRaises(ValueError): self.model.prompt_tokens("a" * 501)

    def test_unknown_label_rejected(self):
        with self.assertRaises(ValueError):
            self.model.batch([{"text": "a", "label": "melting"}])

    def test_token_view_exposes_count_without_calling_model(self):
        result = self.model.token_view("abc")
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["first_token_ids"], [7, 7, 7])
        self.assertEqual(result["first_pieces"], ["piece", "piece", "piece"])

    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError): self.model.batch([])

    def test_original_final_test_is_not_exposed_by_lab(self):
        with self.assertRaises(ValueError): load_partition("unused", "test")

    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "keep.txt").write_text("original")
            with self.assertRaises(FileExistsError): train_experiment(path, out=path)
            self.assertEqual((path / "keep.txt").read_text(), "original")
