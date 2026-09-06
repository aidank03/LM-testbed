# Factor PyTorch lab

This is a runnable, editable PyTorch module using the existing local Qwen3-0.6B weights and Factor's LoRA adapter. It classifies whether a scientific claim is recorded, reconstructed, model-inferred, or unresolved. It is an experimental evidence classifier, not a general chatbot or a validator of physical mechanisms.

## Start here

In the Factor directory, run:

```sh
./PyTorch.command
```

You can also double-click `PyTorch.command` in Finder. It uses the existing `.venv-local` environment and current source, so edits take effect the next time you launch it. It makes no API calls or downloads. The initial model load may take several seconds.

At the prompt, type `/example` to try the supplied claim, `/info` to see the device, or `/quit` to close. You can enter your own one-line record, for example:

```text
TARGET: ETI caused the fingers. Images show fingers, but no measurement distinguishes ETI from competing mechanisms.
```

The appropriate interpretation of this example is unresolved. A model can still make mistakes.

## Inspect, predict and train

From a terminal in the Factor directory:

```sh
./PyTorch.command info
./PyTorch.command predict --text 'TARGET: detector counts. The acquisition file stores the original detector counts.'
./PyTorch.command train --steps 8
```

`train` creates a fresh rank-8 LoRA adapter on the base model. It trains only adapter weights, using four training examples per step and AdamW. It prints training loss and development loss before/after, then saves a new adapter and `report.json` under a unique `runs/pytorch-lab/` directory. Lower development loss is preferable, but a decrease is not guaranteed and does not prove scientific usefulness. Repeated experimentation exposes development data; it is not a fresh final evaluation.

To select your output directory or force the CPU:

```sh
./PyTorch.command train --steps 8 --out runs/pytorch-lab/my-experiment --device cpu
```

Existing output directories are refused. The lab reads only the frozen training and development partitions, never the original final test. It does not modify the original trained adapter, earlier evaluation records or the 0.3 release ZIP/wheel. This module is a new source-checkout addition; use its launcher rather than the earlier installed wheel.

After training, test your new checkpoint by substituting its printed directory:

```sh
./PyTorch.command predict --adapter runs/pytorch-lab/my-experiment/adapter
```

## Read and edit the code

Open [torch_lab.py](../src/factor/torch_lab.py). The main pieces are:

- `ProvenanceModule(nn.Module)`: loads local weights, registers the language model as a PyTorch submodule and places it on the chosen device.
- `forward()`: returns the language model's loss/logits for a token batch.
- `batch()`: prepares labels and excludes prompt/padding tokens from the training loss using `-100`.
- `predict()`: disables gradients and generates a label.
- `train_experiment()`: shows the actual forward → loss → backward → gradient clipping → optimizer step loop, followed by development evaluation and checkpoint saving.

For your own Python script, use `.venv-local/bin/python` with `PYTHONPATH=src` from the Factor directory:

```python
from factor.torch_lab import ProvenanceModule

model = ProvenanceModule(".")
print(model.predict("TARGET: melting. A simulation fit suggests melting; no direct phase diagnostic is available."))
```

To explore gradients, construct `ProvenanceModule(".", fresh_adapter=True)`, call `model.train()`, prepare `batch = model.batch(rows)`, and evaluate `loss = model(**batch, use_cache=False).loss`. The `rows` list needs `text` and one of the four valid `label` values. The normal saved-adapter mode is inference-only; its weights are frozen.

These interfaces follow the [PyTorch 2.8 Module documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html). The Mac GPU is selected through [PyTorch's MPS backend](https://docs.pytorch.org/docs/2.8/notes/mps.html); automatic selection falls back to CPU when unavailable. MPS was actually verified on this Mac. CUDA selection is implemented but has not been tested here.

## Setup checks

Seven focused tests cover masking, invalid inputs, final-test exclusion and preservation of existing runs:

```sh
PYTHONPATH=src .venv-local/bin/python -m unittest discover -s tests -p test_torch_lab.py -v
```

A real two-step MPS experiment produced development loss **0.8902 → 0.9484**, a deterioration. It verifies that training and evaluation execute; it is not an improvement claim. The original saved adapter correctly returned `unresolved` for the example in about 0.61 seconds. [Training record](../reports/pytorch-lab-setup/training-report.json) and [exact training source](../reports/pytorch-lab-setup/training-source.py) preserve this run. Its generated checkpoint remains under `runs/pytorch-lab/setup-check/`. The final module was then checked again with two training steps and a successful reload of its newly saved adapter: [final verification](../reports/pytorch-lab-setup/verification.json). That checkpoint is in `runs/pytorch-lab/final-module-check/`; the development loss was again worse, not better.

If the environment or weights are missing on another computer, follow [LOCAL_MODELS.md](LOCAL_MODELS.md). This launcher does not install packages or download weights automatically. On this prepared Mac, everything required is already installed.

Plain-language takeaway: open the file, change a small part, run it again, and inspect both successful and failed results.
