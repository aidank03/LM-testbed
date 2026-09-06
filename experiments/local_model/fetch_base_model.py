"""Explicit, pinned download for this experiment; never called by Factor runtime.

Existing artifacts are verified, never repaired or overwritten. Run --help
without ML dependencies. Downloading requires the saved requirements.lock.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILE_MANIFEST_SHA256 = "35619810cdf20ada8ce0b62df174bccb8b6942198c315cea9374332d36fe708b"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(root=ROOT):
    model = json.loads((root / "base_model_manifest.json").read_text())
    if model.get("model") != MODEL or model.get("revision") != REVISION:
        raise ValueError("Model identity or pinned revision differs from the recorded experiment")
    manifest = root / "base_model_files.json"
    if file_hash(manifest) != FILE_MANIFEST_SHA256:
        raise ValueError("Expected-file manifest differs from the recorded experiment")
    files = json.loads(manifest.read_text())
    for name, info in files.items():
        if (Path(name).name != name or name in {".", ".."}
                or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])
                or type(info["bytes"]) is not int or info["bytes"] < 0):
            raise ValueError("Invalid expected-file specification")
    return files


def verify_files(directory, expected):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Expected an ordinary model directory: {directory}")
    # Hugging Face's local download metadata is not a model input.
    extra = {p.name for p in directory.iterdir()} - set(expected) - {".cache"}
    if extra:
        raise ValueError(f"Unexpected artifact files: {sorted(extra)}")
    for name, info in expected.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or nonordinary artifact file: {name}")
        if path.stat().st_size != info["bytes"] or file_hash(path) != info["sha256"]:
            raise ValueError(f"Artifact size or SHA-256 mismatch: {name}")


def download_pinned(stage, expected):
    # Deliberately lazy: importing this file or asking for help cannot download.
    from huggingface_hub import HfApi, snapshot_download

    resolved = HfApi().model_info(MODEL, revision=REVISION).sha
    if resolved != REVISION:
        raise ValueError("Remote revision did not resolve to the pinned commit")
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=stage,
                      allow_patterns=list(expected), max_workers=2)


def fetch_model(destination, expected, *, verify_only=False, downloader=download_pinned):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        verify_files(destination, expected)
        return "verified_existing"
    if verify_only:
        raise FileNotFoundError(f"No artifact to verify: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".factor-model-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        downloader(stage, expected)
        verify_files(stage, expected)
        # Exclusive creation and links refuse races and never replace user files.
        # Interruption can leave an incomplete directory; a later run refuses it.
        destination.mkdir(exist_ok=False)
        for name in expected:
            os.link(stage / name, destination / name)
        verify_files(destination, expected)
    return "downloaded_and_verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "base_model")
    parser.add_argument("--verify-only", action="store_true", help="Check local files without network access")
    args = parser.parse_args()
    try:
        status = fetch_model(args.output, load_contract(), verify_only=args.verify_only)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Refused: {error}\n")
    print(json.dumps({"status": status, "model": MODEL, "revision": REVISION,
                      "expected_file_manifest_sha256": FILE_MANIFEST_SHA256,
                      "output": str(args.output.absolute())}, indent=2))


if __name__ == "__main__":
    main()
