# Factor and PyTorch — explain it like I am five

Imagine a small robot with a very large book inside its head.

You give the robot a short science note. The robot puts the note into one of four boxes:

| Box | Plain meaning | Example |
| --- | --- | --- |
| `recorded` | The instrument saved it directly | Camera pixels or detector voltage |
| `reconstructed` | Math turned the recording into a useful quantity | Velocity calculated from a PDV signal |
| `model_inferred` | A physical model was needed to estimate it | Temperature inferred by fitting a simulation |
| `unresolved` | The evidence cannot decide yet | “ETI caused these fingers” when several causes still fit |

The robot is a local Qwen3-0.6B language model. It runs on this Mac. A small LoRA adapter teaches it this four-box task without changing all of the robot's original weights.

## The five ideas

1. **A tensor is a box of numbers.** Text becomes token numbers, and PyTorch stores those numbers in tensors.
2. **A model is a number-changing machine.** It receives tensors and predicts which token should come next.
3. **Loss is a “how wrong were you?” number.** Smaller usually means the model fits the development examples better.
4. **Training adjusts weights.** PyTorch calculates which direction would reduce the loss, and an optimizer makes a small adjustment.
5. **Evaluation checks new examples.** We keep evaluation separate because memorizing the lesson is different from understanding a new example.

The small training loop is:

```text
science note → token numbers → model → predicted label
                                  ↓
correct label → loss → gradients → optimizer changes LoRA weights
```

## Lesson 1: ask the trained model a question

The interactive window should already be open. If it is closed, open Terminal, move into the Factor folder, and run:

```sh
cd /Users/operator/.codex/.chatgpt-projects/g-p-6a9c774029e08191958c8c7f5fa7e384/factor
./PyTorch.command
```

Wait until you see:

```text
Ready on mps:0. Labels: recorded, reconstructed, model_inferred, unresolved
Factor PyTorch >
```

Type:

```text
/example
```

The example says that images show fingers but do not distinguish ETI from competing mechanisms. A sensible answer is:

```json
{"label": "unresolved"}
```

This label means “the available evidence cannot answer that question.” It does not mean ETI is impossible.

Now try these yourself, one line at a time:

```text
TARGET: detector voltage. The acquisition system saved detector voltage samples directly.
```

Expected box: `recorded`.

```text
TARGET: radial velocity. Signal processing converted the saved PDV waveform into radial velocity.
```

Expected box: `reconstructed`.

```text
TARGET: material temperature. A simulation was fitted to the measured signal to estimate material temperature.
```

Expected box: `model_inferred`.

```text
TARGET: melting caused the motion turnaround. The motion changed direction, but no phase diagnostic was available.
```

Expected box: `unresolved`.

The model may still answer incorrectly. That is useful: save the example as a failure case rather than quietly changing the expected answer.

Type `/info` to inspect the device and a few weight shapes. Type `/quit` when finished.

## Lesson 2: run one answer without the interactive prompt

This is useful when another program wants to call the model:

```sh
./PyTorch.command predict --text 'TARGET: detector counts. The data file stores the original detector counts.'
```

The result contains:

- `label`: one of the four valid boxes, or `null` if the output was invalid.
- `generated_text`: exactly what the model generated.
- `seconds`: local generation time.
- `scope`: a reminder that this is a narrow experiment, not physics validation.

No cloud API is used.

## Lesson 3: watch PyTorch learn

Run a tiny training experiment:

```sh
./PyTorch.command train --steps 8
```

The program will:

1. Load the original local Qwen weights.
2. Add fresh, trainable LoRA weights.
3. Measure development loss before training.
4. Repeat `forward → loss → backward → optimizer step` eight times.
5. Measure development loss again.
6. Save the new adapter and a report in a new `runs/pytorch-lab/` folder.

You might see something like:

```text
Development loss before: 0.8902
Step 1/8: training loss ...
...
Development loss after: ...
Saved: .../runs/pytorch-lab/abc123
```

Lower development loss is encouraging. Higher loss is a real negative result. Eight steps are for learning the machinery, not for claiming that the model improved.

Every run gets a new folder. The program refuses to overwrite an old experiment.

## Lesson 4: use the adapter you just trained

Copy the saved folder printed by the training command. If it printed `runs/pytorch-lab/abc123`, run:

```sh
./PyTorch.command predict \
  --adapter runs/pytorch-lab/abc123/adapter \
  --text 'TARGET: detector counts. The data file stores the original detector counts.'
```

Compare its answer with the original adapter:

```sh
./PyTorch.command predict \
  --text 'TARGET: detector counts. The data file stores the original detector counts.'
```

One example does not establish improvement. A real comparison needs a frozen collection of examples that training never saw.

## Lesson 5: read the important code

Open [torch_lab.py](../src/factor/torch_lab.py). Find these five places:

### `class ProvenanceModule(nn.Module)`

This makes our object a real PyTorch module. PyTorch can discover its model weights, move them to the Mac GPU, switch between training and evaluation, and calculate gradients.

### `forward()`

```python
def forward(self, **batch):
    return self.lm(**batch)
```

This sends a prepared batch through Qwen. During training, its returned result includes the loss.

### `batch()`

This converts text and labels into tensors. Values of `-100` tell PyTorch not to score prompt or padding tokens. The model is graded on the answer label.

### `predict()`

`@torch.inference_mode()` turns off gradient tracking. That saves memory because prediction does not need to update weights.

### `train_experiment()`

These are the central learning lines:

```python
loss = model(**batch, use_cache=False).loss
loss.backward()
torch.nn.utils.clip_grad_norm_(parameters, 1.0)
optimizer.step()
```

In plain language: measure the mistake, work out which way the adjustable weights should move, limit an excessively large move, then take one step.

## Three safe experiments

### Change the number of steps

```sh
./PyTorch.command train --steps 1
./PyTorch.command train --steps 16
```

Compare the two saved `report.json` files. Look at `development_loss_before`, `development_loss_after`, and `training_losses`.

### Change the random seed

The launcher currently fixes seed 42. In `torch_lab.py`, change the default value in `train_experiment()` and run again. Different results show method variability.

### Add a difficult development example

Do this only after copying the dataset to a new experimental file. The existing frozen data has recorded hashes, so Factor will reject silent edits. That refusal protects the evidence behind earlier results.

## What this module proves

It proves that this Mac can load a real open-weight model through PyTorch, run local inference, compute loss, perform backpropagation, update LoRA weights, save an adapter, and reload it.

It does not prove that the model understands liner physics, that a lower loss improves scientific decisions, that the four labels cover every scientific situation, or that the benchmark represents real experimental data.

The first short verification training run actually made development loss worse: **0.8902 → 0.9484**. That is okay. The machinery worked, and the negative result tells us two training steps were not helpful.

## If something goes wrong

- **“Permission denied”**: run `chmod +x PyTorch.command` once.
- **“Local base weights missing”**: follow [LOCAL_MODELS.md](LOCAL_MODELS.md).
- **“Device unavailable”**: add `--device cpu`.
- **“Output directory exists”**: choose a new folder name or let Factor create one automatically.
- **The label is wrong**: record the full input and output as a failure case. Do not call a plausible-sounding answer correct without evidence.

Plain-language takeaway: PyTorch gives us the learning machinery; Factor adds the scientific rules, records, and tests that help us decide whether the learning was actually useful.

Continue with [Cool PyTorch experiments](COOL_PYTORCH_EXPERIMENTS.md) when you are comfortable with these basics.
