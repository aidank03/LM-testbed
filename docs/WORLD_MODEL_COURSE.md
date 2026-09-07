# World-model course guide

## Learning sequence

Read and run one notebook at a time, starting at 05. A useful rhythm is 45–90 minutes per lesson: read the explanation, predict a result, run the cell, inspect a failure and attempt an exercise. Later lessons may warrant more time. The saved execution time is much shorter than the learning time.

The first three lessons build the environment. Lessons 08–09 solve inverse problems and quantify uncertainty. Lessons 10–11 learn dynamics and observation representations. Lessons 12–13 introduce a real causal language model and its tool interface. Lesson 14 assembles a synthetic experiment loop.

No earlier notebook must remain running. Shared definitions are imported from `lm_testbed/`, while the main model architectures and training loops are visible in the notebooks. Repeated LM training in 12–14 is intentional: each lesson works independently. None of these lessons loads or changes the previous Qwen adapter or the consumed PDV evaluations.

## Existing Mac environment

The installed **Python (Factor AI)** Jupyter kernel contains the dependencies. Open `notebooks/05_experiment_contract_and_config.ipynb`, choose that kernel, then **Restart Kernel → Run All**. The repository root must contain both `notebooks/` and `lm_testbed/`.

Each run writes to a new `runs/world-model/<lesson>-<unique-id>/` directory. Reports contain configuration, Python/library versions, CPU device and source-cell/module hashes. Model lessons also save weights. Re-running does not overwrite an earlier run.

## Fresh environment

From the repository root, with Python 3.12 available:

```sh
python3.12 -m venv .venv-world-model
source .venv-world-model/bin/activate
python -m pip install -r requirements-world-model.txt
python -m ipykernel install --user --name factor-ai --display-name 'Python (Factor AI)'
python -m jupyterlab
```

If an existing `factor-ai` kernel points to a different environment, retain it and use a distinct kernel name, then select that kernel in the notebooks. The execution script defaults to `factor-ai` and accepts `--kernel` to select another registered kernel.

`lm_testbed/` is a notebook-local Python package, found by the setup cell. It does not alter Factor's existing build configuration or distribution name. You do not need to install the Factor package to run this course.

The tested versions are recorded in `requirements-world-model.txt`. Installing dependencies on another machine requires network access; notebook execution itself does not. CPU is the tested default. Device transfer for CUDA/MPS is an exercise, not a verified capability of these lesson implementations.

## Reproduce execution and checks

```sh
python scripts/execute_notebooks.py
PYTHONPATH=. python -m unittest discover -s tests -p test_world_model.py -v
```

The runner starts a clean Jupyter kernel for each lesson, saves outputs and writes `reports/notebook_execution.json`. To execute one lesson:

```sh
python scripts/execute_notebooks.py --lessons 12
```

`build_notebooks.py` is an authoring source, not a normal learner command: running it regenerates all ten notebooks and clears their outputs. Make a Git commit or copy your edited notebooks before deliberately regenerating them.

## Read the results carefully

- Training changes weights; tuning chooses the checkpoint. Calibration fits interval multipliers. Evaluation measures held-out synthetic behavior.
- All shipped generators, seeds and evaluation cases are public. These are exposed development evaluations, not locked tests. Any independent final evaluation must be newly versioned and withheld during development.
- Correlated frames from one shot are not independent training/evaluation examples.
- The probabilistic inverse model uses an approximate diagonal Gaussian. Its calibrated intervals have a marginal exchangeability argument; they do not guarantee coverage under wrong physics.
- The tiny LM has a closed vocabulary and only 24 semantic cases. Its current four-case evaluation mostly tests missing-current cases. It is a mechanics lesson, not a broad agent benchmark. The authored rule is a stronger baseline and remains available.
- A syntactically valid tool call can be scientifically wrong. A numerical tool result can be conditional on incorrect assumptions. Report these failures separately.
- A failed predictive check says the combined model and measurement assumptions need examination. It does not uniquely identify the cause.

## Configuration ownership

| Layer | Implemented fields | Important later additions |
|---|---|---|
| Machine | Charge proxy, mass, resistance, capacitance, inductance, coupling, damping, thermal loss, growth, interval and samples | Time-varying drive, multiple circuit sections, geometry, material laws, spatial dynamics |
| Acquisition | Gain, shared clock shift, temporal blur, noise, missing channels, PDV wavelength proxy, image PSF/exposure, radiation response | Per-channel clocks/gains, transfer functions, saturation, correlated noise, calibration records |
| Processing | PDV window, integration, image edge and Fourier-mode extraction | Reconstruction ensembles and calibrated output covariance |
| Inference | Parameter subsets, priors, likelihood, timing covariance, gain prior, neural density model | Joint uncertainty over more nuisance parameters and competing model families |
| LM/runtime | Vocabulary, architecture, training budget, allowed calls, call budget, traces | Real tokenizers, pretrained/local providers, multi-turn tool learning, broader held-out tasks |
| Design | Diagnostic candidates, noise assumptions, fictional cost, entropy objective | Feasibility constraints and decision-specific utility; real experts would define these |

## Sources and relationship to earlier work

The diagnostic families are inspired by Sandia's public [Z diagnostic overview](https://www.sandia.gov/app/uploads/sites/129/2022/06/Z_Diagnostics.pdf), particularly the diagnostic-family diagram and B-dot monitor descriptions. That source motivates the instrument categories; it does not validate our coefficients, operators or combined acquisition scenario.

Sandia's [optical-fiber velocimetry publication record](https://www.sandia.gov/research/publications/details/effects-and-mitigation-of-pulsed-power-radiation-on-optical-fiber-velocimet-2022-05-01/) describes radiation effects on optical diagnostics. More realistic PDV nuisance modeling is a natural extension of the existing Factor PDV lessons.

The [PyTorch TransformerEncoderLayer documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) supplies the attention-layer interface used by the course. These are small reference architectures, not an optimized large-model stack.

Earlier Factor notebooks informed the teaching pattern: conventional comparisons, transparent synthetic truth, development/evaluation separation and retained negative results. The new series preserves those notebooks unchanged.
