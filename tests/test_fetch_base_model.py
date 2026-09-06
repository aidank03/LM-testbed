"""Small offline fixtures only: no model downloads or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/local_model/fetch_base_model.py"
SPEC = importlib.util.spec_from_file_location("factor_fetch_fixture", SCRIPT)
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class FetchBaseModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.destination = self.root / "model"
        self.content = b"small deterministic fixture"
        self.expected = {"weights.bin": {"bytes": len(self.content),
                         "sha256": hashlib.sha256(self.content).hexdigest()}}
        self.calls = 0

    def download(self, stage, expected):
        self.calls += 1
        (stage / "weights.bin").write_bytes(self.content)

    def test_published_fixture_is_verified_and_repeat_uses_no_downloader(self):
        self.assertEqual(fetch.fetch_model(self.destination, self.expected, downloader=self.download),
                         "downloaded_and_verified")
        self.assertEqual(fetch.fetch_model(self.destination, self.expected, downloader=self.download),
                         "verified_existing")
        self.assertEqual(self.calls, 1)
        self.assertEqual((self.destination / "weights.bin").read_bytes(), self.content)

    def test_bad_existing_artifact_is_preserved_without_download(self):
        self.destination.mkdir()
        path = self.destination / "weights.bin"
        path.write_bytes(b"x" * len(self.content))  # Same size, wrong hash.
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            fetch.fetch_model(self.destination, self.expected, downloader=self.download)
        self.assertEqual(path.read_bytes(), b"x" * len(self.content))
        self.assertEqual(self.calls, 0)

    def test_incomplete_existing_directory_is_not_repaired(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, "Missing"):
            fetch.fetch_model(self.destination, self.expected, downloader=self.download)
        self.assertEqual(self.calls, 0)

    def test_bad_download_does_not_publish_destination(self):
        def bad(stage, expected):
            (stage / "weights.bin").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            fetch.fetch_model(self.destination, self.expected, downloader=bad)
        self.assertFalse(self.destination.exists())

    def test_verify_only_never_downloads(self):
        with self.assertRaises(FileNotFoundError):
            fetch.fetch_model(self.destination, self.expected, verify_only=True, downloader=self.download)
        self.assertEqual(self.calls, 0)

    def test_racing_creation_cannot_be_replaced(self):
        def racing(stage, expected):
            self.download(stage, expected)
            self.destination.mkdir()
            (self.destination / "user.txt").write_text("preserve")
        with self.assertRaises(FileExistsError):
            fetch.fetch_model(self.destination, self.expected, downloader=racing)
        self.assertEqual((self.destination / "user.txt").read_text(), "preserve")

    def test_recorded_contract_is_pinned_and_tampering_is_rejected(self):
        self.assertIn("model.safetensors", fetch.load_contract())
        for name in ("base_model_manifest.json", "base_model_files.json"):
            (self.root / name).write_bytes((SCRIPT.parent / name).read_bytes())
        manifest = self.root / "base_model_manifest.json"
        model = json.loads(manifest.read_text())
        model["revision"] = "main"
        manifest.write_text(json.dumps(model))
        with self.assertRaisesRegex(ValueError, "pinned revision"):
            fetch.load_contract(self.root)
        manifest.write_bytes((SCRIPT.parent / manifest.name).read_bytes())
        (self.root / "base_model_files.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Expected-file manifest"):
            fetch.load_contract(self.root)


if __name__ == "__main__":
    unittest.main()
