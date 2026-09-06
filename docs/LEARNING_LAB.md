# Factor Learning Lab

The main Learning Lab is now the VS Code notebook `notebooks/01_pytorch_jax_factor_lab.ipynb`. It is a visual place to learn what PyTorch is doing, edit the actual Python, rerun one cell, and see plots inline. It is deliberately smaller than the full Factor agent system. Start here, understand the loop, then reuse the same evaluation habits for CNNs, PINNs, language models, and physics simulations.

## Open it

Open the notebook in VS Code, choose **Python (Factor AI)** at the upper right, and press **Run All**. To activate the same environment in a new terminal:

```sh
conda activate factor-ai
```

The Python, Jupyter, and Pylance extensions are the editor pieces. The Conda environment holds Python and the scientific packages. PyTorch performs tensor operations and training; Matplotlib and Seaborn provide plots because PyTorch does not include a plotting system.

The earlier local browser lesson remains available with `./Learn.command`, but the notebook is the recommended learning surface.

## Lesson 1: the smallest useful neural network

The model has only two adjustable numbers: a slope and an intercept. It must discover a hidden straight-line rule from noisy examples.

- **Tensor:** a box of numbers. The inputs and answers are tensors.
- **Weight:** an adjustable knob inside the model. The slope and intercept are weights.
- **Loss:** the model's wrongness score. Smaller is better.
- **Gradient:** an arrow showing which way and how strongly to move a weight.
- **Optimizer:** the rule that makes that move.
- **Epoch:** one complete practice step over the training examples in this lesson.

The green and orange loss lines should fall. The learned slope should approach `2.5`, and the intercept should approach `-0.4`. Change the noise, learning rate, and number of steps. A learning rate that is too small learns slowly; one that is too large can jump around or miss the answer.

Training data is the homework. Development data is a practice exam used to observe the training choice. Test data is the final exam. The test set is evaluated only after training and never changes the weights.

## Lesson 2: a small scientific learning loop

The toy diagnostic reports:

```text
apparent displacement = material motion + optical drift + noise
```

Stage 1 trains on clean measurements, then fails when hidden drift appears. Stage 2 adds more varied training data but still receives only the sum. It can improve average error, but it cannot know the two hidden parts for an individual case. Stage 3 adds a noisy reference channel that measures drift. This changes the available information, so the page labels it as a different task.

This is what a safe recursive improvement loop looks like:

1. Freeze the question, final test, metric, and acceptance gate.
2. Train a candidate.
3. Measure a specific failure on unseen cases.
4. Propose one bounded change.
5. Retrain and compare on the same test contract.
6. Keep the result and its evidence, even when it fails.
7. Require a person to review consequential changes.

The candidate does not get to rewrite its own exam or acceptance rule. Otherwise it can appear to improve by making the test easier.

## See the evidence

Each click creates a directory like:

```text
runs/learning-lab/learn_1234.../
  request.json       settings chosen before training
  events.jsonl       append-only training progress
  result.json        final metrics and plots
  run.json           status and a SHA-256 result fingerprint
```

These records make a graph on the screen auditable. They also preserve failed experiments.

## JAX

JAX is installed and the notebook contains an executed gradient parity check. On this Apple Silicon Mac, PyTorch uses the Apple GPU through MPS while the official JAX build runs on CPU. The notebook reports the actual devices. The parity check compares the same loss and gradient; it is an implementation check, not a speed claim or scientific validation.

## Suggested path from here

1. Change the line lesson until you can explain every graph and number.
2. Add an intentionally bad optimizer setting and predict the failure before running it.
3. Build a small CNN lesson that finds simple synthetic structures in images.
4. Challenge it with blur, saturation, timing shifts, and a different image generator.
5. Build a PINN lesson around a transparent differential equation with an analytical answer.
6. Deliberately give the PINN a wrong physics constraint and measure the damage.
7. Evaluate the local language model on a frozen scientific task, diagnose one failure family, create a small training set, post-train, and repeat the locked evaluation.
8. Add JAX parity only when it answers a concrete learning or performance question.

The motion lesson teaches an inference principle using synthetic numbers. It does not validate PDV reconstruction, identify melting, or establish ETI, MRTI, or driver-target coupling. Real scientific validation needs diagnostic forward models, calibration, provenance, independent experiments, and real measurements.
