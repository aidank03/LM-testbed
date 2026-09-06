# Local models: first executed experiment

Factor now has actual local model inference and an actual PyTorch LoRA checkpoint. The measurements below concern two different tasks. They do not establish physics accuracy or transfer between those tasks.

## Running a local model

The installed Qwen3.5-9B Q4_K_M model is served by LM Studio as `factor-qwen35-9b`, with a 4,096-token context. Its Factor-specific load expires after 3,600 idle seconds. No existing model was unloaded; no user model template or persistent configuration was changed. This machine has an Apple M4 Max, 14 physical CPU cores and 36 GiB unified memory.

When this installed model has expired, load it again explicitly with the verified LM Studio model key:

```sh
lms load qwen/qwen3.5-9b --identifier factor-qwen35-9b --context-length 4096 --gpu max --ttl 3600 -y
```

LM Studio's local API server must also be running. The model key is an installed-library identifier, not a filesystem path. The separate simulation-loop demonstration used `factor-qwen35-loop` with a 16,384-token context because its larger tool registry and job results require more room.

```python
from factor.local_models import LocalChatClient

client = LocalChatClient(
    base_url="http://127.0.0.1:1234/v1",
    model="factor-qwen35-9b",
    transport="lmstudio-native",
)
result = client.complete(
    [{"role": "user", "content": "Return a JSON object with the key status."}],
    max_tokens=128,
    timeout=20,
)
print(result["text"])
```

The client accepts only HTTP loopback destinations, disables redirects and environment proxies, limits response bodies to 2 MB, and uses explicit timeouts. It never falls back to a cloud endpoint or invokes a returned tool. Reasoning fields are redacted from recorded traces. Scientific and action-schema validation belong to the calling runtime.

The native LM Studio endpoint supports an explicit `reasoning: off` setting and `store: false`. A requested JSON schema is conveyed in the prompt in native mode; it is **not server-enforced** and must be validated locally. The native response has no explicit finish reason. The client records that its completion status is inferred from a nonempty final message and an output count below the token limit. Exhausted or unknown budgets are rejected conservatively. Native mode does not expose a seed parameter; this run used temperature zero. [LM Studio native chat documentation](https://lmstudio.ai/docs/developer/rest/chat)

Three serving conditions were retained:

| Condition | Result |
|---|---|
| OpenAI-compatible JSON object mode | HTTP 400; no predictions |
| OpenAI-compatible explicit JSON schema | 0/12 usable final answers; output was routed to the reasoning channel despite the requested non-thinking setting |
| Native API with reasoning off | 12/12 usable answers; 5/12 correct on the original public tasks, in 12.6 seconds |

The native run got all five numerical tasks wrong, chose `classical_ETI` from a single unsupported morphology description, and selected the wrong illustrative experiment. These errors are preserved. The successful serving change solved an output-routing failure; it did not make the model scientifically reliable. The observed compatibility behavior resembles issues reported to LM Studio, but our own saved traces are the evidence for this installation. [Official issue tracker, #1990](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1990), [#1971](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1971)

Results: `experiments/local_model/qwen35_public_probe_native/`. Earlier failures remain in the neighboring `qwen35_public_probe*` directories. The twelve original tasks were already public and have been exposed repeatedly; none is an unseen test.

## Actual smaller-model post-training

We downloaded the Apache-2.0 Qwen3-0.6B checkpoint at revision `c1899de289a04d12100db370d81485cdf75e47ca` and trained causal-language-model LoRA adapters on its attention `q_proj` and `v_proj` matrices. The model still generates tokens. This is **generative LLM adapter training**, not training a separate classifier head. [Official Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B), [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/developer_guides/lora), [PyTorch fine-tuning guidance](https://pytorch.org/blog/finetune-llms/)

The task is deliberately narrow: classify a specified claim as `recorded`, `reconstructed`, `model_inferred` or `unresolved`, from an authored evidence vignette. Some vignettes include an unjustifiably confident caption. Labels follow the stipulated provenance, not an assessment of real experiments. The task does not ask the model to recover motion or identify an instability mechanism.

Before any model evaluation we froze 192 training cases from six measurement-and-wording families, 24 development cases from two other families, and 48 final cases from four additional families. Numeric variants within a family never cross partitions. All labels occur in every family. Only the rubric and vignette text enter the model; IDs, families, split names and expected labels do not. The evaluator and training process are logically separated, not isolated by an operating-system sandbox. The author constructed all partitions and labels; there was no independent annotation or external hidden-data custodian.

The frozen plan specified rank 8, alpha 16, no adapter dropout, three epochs, batch size 8, AdamW learning rate 0.0005, seed 20260905, response-token-only loss, and greedy non-thinking decoding. There were 1,146,880 trainable parameters. Training and evaluation used PyTorch 2.8.0, Transformers 4.57.6, PEFT 0.18.1, MPS and bfloat16. The full process took 81.9 seconds. Three training epochs took approximately 61.2 seconds together. Checkpoint selection used only the lowest development label-token loss, selecting epoch three.

After checkpoint selection we evaluated the unchanged base model and selected adapter on the final partition once. No tuning followed. This test is now exposed and cannot be reused as an unseen test.

| Metric | Before | After |
|---|---:|---:|
| Final accuracy | 15/48 (31.25%) | 33/48 (68.75%) |
| Recorded | 10/12 | 12/12 |
| Reconstructed | 5/12 | 6/12 |
| Model inferred | 0/12 | 3/12 |
| Unresolved | 0/12 | 12/12 |
| Invalid generated labels | 0/48 | 0/48 |

Twenty answers improved and two regressed, a net improvement of 37.5 percentage points. Both regressions called a reconstructed reflectance curve a recorded signal. Fifteen errors remain, including nine of twelve model-inferred claims. Development accuracy rose from 10/24 to 24/24; the worse final result exposes limited generalization rather than being hidden by the development score.

The four family groups improved as follows:

| Family | Before | After |
|---|---:|---:|
| Neutrons | 3/12 | 9/12 |
| Reflectometry | 3/12 | 6/12 |
| Strain | 3/12 | 6/12 |
| Velocimetry | 6/12 | 12/12 |

There are four family groups, not 48 independent experimental demonstrations. One training seed was evaluated; seed variability is unmeasured. A majority-label baseline scores 25%. Shuffling evaluator labels yields 29.17% accuracy for the trained predictions, an evaluator sanity check—not a shuffled-label-training ablation. We have not compared against the strongest rule-based provenance parser. These results establish improvement over this same pretrained model on these authored cases, not general scientific competence.

The run includes independently recomputed family scores, all remaining errors, and both regressions in `posttrain_run/independent_audit.json`. The full dataset hashes, planned criteria, package versions, training history, checkpoint selection and final-test exposure are recorded alongside predictions.

## Calling the trained artifact

```python
from factor.training import LocalProvenanceModel

model = LocalProvenanceModel(
    "experiments/local_model/base_model",
    "experiments/local_model/posttrain_run/selected_adapter",
    "experiments/local_model/frozen_dataset/manifest.json",
    device="mps",  # "cpu" is also supported
    expected_adapter_sha256="728568751cba991b88660bfecdc1058949e3d0b3efa1c9203abef16fd3e66fbd",
)
result = model.classify("A detector stored counts. TARGET: the stored counts.")
```

Weights load locally without automatic downloads. The wrapper refuses contexts beyond its evaluated length and abstains on invalid generated labels. This does not guarantee appropriate abstention on unfamiliar content. A new, separate reload smoke test returned `unresolved` for an unsupported MRT attribution; it verified checkpoint loading and function-call integration, not additional final-test performance.

## Reproduction and artifact identity

From the Factor repository root:

```sh
python3 -m venv .venv-local
.venv-local/bin/python -m pip install -r experiments/local_model/requirements.lock
PYTHONPATH=src .venv-local/bin/python -m unittest discover -s tests -p 'test_local_models.py' -v
.venv-local/bin/python -m unittest discover -s tests -p 'test_fetch_base_model.py' -v
.venv-local/bin/python -m unittest discover -s tests -p 'test_posttrain_metadata.py' -v
PYTHONPATH=src .venv-local/bin/python experiments/local_model/serve_probe.py --out NEW_PUBLIC_PROBE_DIRECTORY --transport lmstudio-native
.venv-local/bin/python experiments/local_model/audit_results.py
```

The saved `fetch_base_model.py` command downloads Qwen/Qwen3-0.6B only when explicitly run. It resolves commit `c1899de289a04d12100db370d81485cdf75e47ca`, checks the recorded expected-file manifest, and verifies every expected file's size and SHA-256 before publishing a new model directory. A matching existing directory is verified without downloading; an incomplete or mismatched directory is refused without repairs or overwrites. `--verify-only` uses no network. Runtime model loading never invokes this command. The helper's offline fixture tests verify these behaviors; its new download path has not been exercised with another full model download.

To reproduce the frozen training plan, create a new experiment directory. These commands retain the original scripts and partitions and leave all original results in place. The explicit fetch may download about 1.52 GB; training is local. The reused final partition is already exposed, so this is a reproduction, not another held-out experiment.

```sh
(
set -eu
factor_repro=experiments/local_model_reproduction_001
mkdir "$factor_repro"
cp experiments/local_model/dataset.py experiments/local_model/posttrain.py experiments/local_model/run_metadata.py experiments/local_model/audit_results.py experiments/local_model/fetch_base_model.py experiments/local_model/base_model_manifest.json experiments/local_model/base_model_files.json experiments/local_model/requirements.lock "$factor_repro/"
cp -R experiments/local_model/frozen_dataset "$factor_repro/frozen_dataset"
.venv-local/bin/python -c 'import json, pathlib, sys; (pathlib.Path(sys.argv[1]) / "reproduction_context.json").write_text(json.dumps({"evaluation_status": "EXPOSED_REPRODUCTION_NOT_NEW_HELDOUT", "source": "experiments/local_model", "instruction": "Preserve the original results; do not tune against the reused final partition."}, indent=2))' "$factor_repro"
.venv-local/bin/python "$factor_repro/fetch_base_model.py"
.venv-local/bin/python "$factor_repro/posttrain.py"
.venv-local/bin/python "$factor_repro/audit_results.py"
)
```

Use a new directory name if that reproduction directory already exists. Keep `reproduction_context.json` alongside its outputs; the copied training manifest's original freeze description records historical provenance, not renewed test secrecy. `posttrain.py` refuses an existing `posttrain_run`, checks the frozen split hashes, and uses local model files. Hardware and numerical differences may change the checkpoint hash and results. `dataset.py` creates partitions only when its target directory is absent; reproduction copies the recorded partitions instead. No independent generator, physical simulator, real shot, remote API model or cloud training job was used in the original experiment.

The current training script includes a prospective metadata fix: future runs record the detected operating system, architecture, logical CPU count, and selected Torch device. CPU model, physical core count, and memory are explicitly `unknown`; unavailable system probes also return `unknown`. Each future manifest hashes the metadata helper as well as the training script. The exact original script is preserved in `posttrain_run/source_snapshot/posttrain.py`, verified against the original run's `source_sha256` (`eec7d563b8aab2e0e56ddac03155926ade97feb5fea1cb9946b748668d340383`). The original measured manifest, metrics, predictions, and adapter remain unchanged. This code fix was verified with offline metadata tests; training and final evaluation were not rerun.

The served GGUF file is 5,627,044,256 bytes; LM Studio's displayed model-size estimate also includes associated assets. Its SHA-256 is `cd76ec205963b3b33350093e6904d9de16c4e666fd104e1f632d25c7f15f2a13`, verified against the file in `lmstudio-community/Qwen3.5-9B-GGUF` at revision `1379f25c6b505a3fc737bd7818cb09389cf807c1`. The LM Studio CLI reports commit `6041ae0`. Full details are in `serving_manifest.json`. [Official Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)

The selected adapter SHA-256 is `728568751cba991b88660bfecdc1058949e3d0b3efa1c9203abef16fd3e66fbd`. Large baseline weights, caches and virtual environments should remain outside source control; the small adapter is a local experimental artifact with its provenance manifest.

The next research decision is whether the provenance task merits a new, independently authored evaluation with stronger rule baselines and multiple training seeds. The next engineering step is to connect validated numerical tools to the served model and evaluate that workflow separately. Neither step should tune against the now-exposed final partition.
