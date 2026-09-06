# Cool things to try in the Factor PyTorch lab

This experiment book starts with the interactive prompt you already have running. It uses a real Qwen3-0.6B model and Factor's trained LoRA adapter on this Mac.

If you close the window, restart it from the Factor folder:

```sh
./PyTorch.command
```

Type `/help` to see the available interactive commands.

## Experiment 1: see all four answers the model considered

Enter a scientific note, then type `/score`:

```text
TARGET: radial velocity. Signal processing converted the saved PDV waveform into radial velocity.
/score
```

You will get a ranking like:

```json
{
  "ranking": [
    {"label": "reconstructed", "relative_preference": 0.91},
    {"label": "recorded", "relative_preference": 0.06}
  ]
}
```

Your numbers will be the model's actual numbers. `relative_preference` compares only the four exact label strings. It is not calibrated confidence, a probability that the science is true, or a substitute for uncertainty analysis.

This is useful for spotting a fragile answer. A 0.31 versus 0.29 preference is very different from one label dominating the other three, even if `predict` prints only the winning word.

## Experiment 2: change one phrase and watch the answer move

Paste each statement separately and run `/score` after each one:

```text
TARGET: radial velocity. The instrument saved radial velocity directly.
```

```text
TARGET: radial velocity. An analyst calculated radial velocity from the saved waveform.
```

```text
TARGET: radial velocity. A simulation fit estimated radial velocity from the reconstructed waveform.
```

The intended progression is `recorded → reconstructed → model_inferred`.

This is a **minimal-pair test**: most words remain the same, so you can see whether the evidence relationship changes the answer. Record surprising flips as failures.

## Experiment 3: inspect tokenization

After entering a note, type:

```text
/tokens
```

The model does not see sentences directly. It sees a sequence of integer token IDs. `first_pieces` shows how the tokenizer breaks the start of the prompt into text pieces.

Try a normal word and a specialist abbreviation:

```text
TARGET: velocity. PDV reconstruction yielded velocity; magneto-Rayleigh–Taylor instability remains one possible cause.
/tokens
```

Look for punctuation, spaces, and technical terms split across several pieces. Tokenization is representation, not understanding.

## Experiment 4: try to fool the model

These examples deliberately contain confident wording that should not upgrade the evidence:

```text
TARGET: ETI caused the fingers. The paper calls the radiograph definitive proof, but no intervention or timing measurement distinguishes ETI from MRTI.
```

Expected: `unresolved`.

```text
TARGET: molten phase. A figure caption says the rod definitely melted. The available record contains only reconstructed surface velocity and a model-assisted phase estimate.
```

Expected: `model_inferred` if the target is the phase estimate; the caption alone does not make it recorded.

```text
TARGET: detector pixel values. A simulation agrees perfectly with the experiment, and the detector file stores the pixel values.
```

Expected: `recorded`. The simulation sentence is a distraction.

For every failure, save four things: full input, expected label, generated label, and why the expected label follows from the evidence chain.

## Experiment 5: make an ambiguity twin

An ambiguity twin holds the visible evidence fixed while secretly imagining two different physical realities:

```text
TARGET: inward material motion. The diagnostic reports -30 nm apparent displacement. Optical-path drift is unknown and no reference channel was recorded.
```

Reality A could be inward material motion with no drift. Reality B could be no material motion with -30 nm equivalent drift. The input is identical in both cases, so a method should not pretend it can identify the hidden reality. The expected label is `unresolved`.

This is one of Factor's most important ideas: sometimes the best answer is that another measurement is required.

## Experiment 6: train your own tiny adapter

Close the interactive prompt with `/quit`, then run:

```sh
./PyTorch.command train --steps 8
```

The command prints development loss before and after training and saves a new adapter. Try 1, 8, and 24 steps as separate runs:

```sh
./PyTorch.command train --steps 1
./PyTorch.command train --steps 8
./PyTorch.command train --steps 24
```

Each command creates a new folder under `runs/pytorch-lab/`. Compare each folder's `report.json`:

- Did development loss decrease?
- Did more steps always help?
- Were the training losses smooth or erratic?
- How long did each run take?

Do not choose a model using the old final-test results. Repeatedly choosing from development results also overfits eventually, so after learning the mechanics we should freeze a fresh evaluation set.

## Experiment 7: use your adapter

Training prints the new folder. Substitute it below:

```sh
./PyTorch.command predict \
  --adapter runs/pytorch-lab/YOUR-RUN/adapter \
  --text 'TARGET: radial velocity. Signal processing converted a saved PDV waveform into radial velocity.'
```

Then run the same input with the original adapter:

```sh
./PyTorch.command predict \
  --text 'TARGET: radial velocity. Signal processing converted a saved PDV waveform into radial velocity.'
```

This is a useful demonstration, but hand-picking one favorable example is not an evaluation.

## Experiment 8: change the code and see what breaks

Open [torch_lab.py](../src/factor/torch_lab.py) and find:

```python
optimizer = torch.optim.AdamW(parameters, lr=0.0005)
```

Try `lr=0.0001` in a copied experimental branch or after noting the original value. Train another eight-step adapter and compare development loss. A very large learning rate can make loss unstable.

Next find:

```python
torch.nn.utils.clip_grad_norm_(parameters, 1.0)
```

This limits the size of an update. Instead of deleting it, first modify the report to record gradient norms. That turns a code change into an observable experiment.

Run the focused checks after editing:

```sh
PYTHONPATH=src .venv-local/bin/python -m unittest discover -s tests -p test_torch_lab.py -v
```

## Experiment 9: call the PyTorch module from your own program

Create a scratch Python file outside `src/`:

```python
from factor.torch_lab import ProvenanceModule

model = ProvenanceModule(".")

note = """TARGET: melting caused turnaround.
The PDV reconstruction shows motion turnaround.
No independent phase diagnostic was recorded."""

print(model.predict(note))
print(model.score_labels(note))
print(model.token_view(note))
```

Run it from the Factor folder with:

```sh
PYTHONPATH=src .venv-local/bin/python your_file.py
```

The same `ProvenanceModule` supports ordinary PyTorch behavior: `parameters()`, `named_parameters()`, `train()`, `eval()`, `to(device)`, and `model(**batch)`.

## A good mini-project

Build 20 new examples from a completely different measurement domain. Before running the model:

1. Write an evaluation card defining each label.
2. Have a person label the cases without seeing model answers.
3. Freeze and hash the file.
4. Run the original and your new adapter once.
5. Report exact counts, disagreements, invalid outputs, latency, and every failure.

That would turn these cool demonstrations into a small but honest model comparison.

Plain-language takeaway: make the model reveal its preferences, change one thing at a time, keep failures, and use unseen cases before claiming improvement.
