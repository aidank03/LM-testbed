# Running and sharing the local development release

The active repository is `factor/` in the current project workspace. The Desktop original was not changed. The source/evidence ZIP is a private local working-tree snapshot of 0.3.0, built on recovered Git commit `d3f48b01819738dcd1531f904833a61c3868e789`; it is not a published repository, signed release or production certification.

## Prepared workspace

From the active repository:

```sh
.venv/bin/factor doctor
.venv/bin/factor run --request configs/motion_request.json --out runs/my-first-run
```

Use a fresh output directory. The supplied example should finish with `ambiguous`: apparent displacement alone cannot separate material motion from unbounded optical drift. For the actual local model, load the explicitly named instance as described in [LOCAL_MODELS.md](LOCAL_MODELS.md), then use the README command. Only the prepared machine has its recorded environments and weights; the ZIP does not contain them.

## Fresh machine

Unpack the source/evidence ZIP and work inside its `factor/` directory. Installation may download Python packages; model weights are a separate explicit download. The commands below install into a new environment and do not enable cloud spending or remote jobs.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/factor doctor
.venv/bin/factor run --request configs/motion_request.json --out runs/first
.venv/bin/python -m unittest discover -s tests -v
```

The measured core run used Python 3.12.14 on macOS arm64. `requirements-core.lock` lists its numerical package versions; on a compatible machine use it before installing Factor to reproduce those dependency versions. The package allows Python >=3.10, but other operating systems and Python versions have not been exercised here. The full 120-test source suite requires the `experiments/` scripts and fixture data supplied in the source/evidence ZIP. A wheel is an application installation, not the full research evidence package.

An already prepared numerical environment can install the tested wheel without downloading dependencies:

```sh
python -m pip install --no-index --no-deps dist/liner_stability-0.3.0-py3-none-any.whl
```

This command assumes all required numerical dependencies are already installed. The exact tested wheel hash and installation procedure are in the [verification record](../reports/verification-final/REPORT.md). Its Python source matches the source tree; the source-tree README received later result and release links.

## What is in the ZIP

The bundle includes source, configs, tests, design documents, evaluator contracts, frozen public cases and separate evaluator truth, predictions, observable model actions, failure reports, source snapshots, the small trained LoRA adapter and the tested wheel. `RELEASE_MANIFEST.json` records each included file's bytes and SHA-256. Its own bytes are excluded from that file list; the accompanying ZIP checksum covers the whole archive.

All released evaluation cases are exposed development material. Original recorded machine paths are preserved as provenance; run IDs and hashes identify the corresponding bundled files. Hashes detect accidental changes, not a malicious writer replacing the whole bundle.

The bundle excludes environments, large base weights, caches, Git internals and top-level runtime scratch jobs. It **includes** `reports/**/runs/`, the audited model execution evidence. Closed-loop numerical output is separately retained under `reports/closed-loop-job-evidence/`.

For the trained adapter, use the explicit pinned-weight helper and fresh reproduction directories described in [LOCAL_MODELS.md](LOCAL_MODELS.md). The historical training script and original results remain unchanged. The original test has already been inspected; another run on it is reproduction, not a new generalization test.

The Qwen3-0.6B upstream license is retained in [third-party/Qwen3-0.6B-LICENSE.txt](third-party/Qwen3-0.6B-LICENSE.txt); source model identity and revision are in the training manifest. No large Qwen or Nemotron serving weights are redistributed. The local Factor project still has no selected public distribution license; see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Remaining integration work

Cloud execution needs a locally configured API key, explicit model and verified rates, plus an authorized total cap. Slurm execution needs a supplied cluster profile and allocation. Shipped profiles leave these disabled. Follow [FRONTIER_APIS.md](FRONTIER_APIS.md) and [HPC.md](HPC.md) when access exists.

Plain-language takeaway: the bundle contains runnable code and the evidence behind the reported local results. A different computer needs its own dependencies and model weights; a real cluster or paid API also needs explicit access and limits.
