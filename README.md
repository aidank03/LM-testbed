# Factor

**A callable scientific environment with measured outcomes.** Factor connects a question to a bounded agent, numerical tools, a simulation job, synthetic diagnostics, conditional inference and a recorded next decision. It starts with one researcher; shared-team hosting comes later.

Version 0.3 is an executed development system, not a validated physics product. It extends the recovered `liner-stability` 0.2 repository and preserves its numerical workflow and CLI. The distribution remains `liner-stability`; new code imports `factor` and installs the `factor` command. No external repository or service has been published.

## Start with the visual Learning Lab

Open [`notebooks/01_pytorch_jax_factor_lab.ipynb`](notebooks/01_pytorch_jax_factor_lab.ipynb) in VS Code and select the **Python (Factor AI)** kernel. It contains editable PyTorch and JAX cells, inline explanations, loss plots, a held-out evaluation, and a deliberately simple Factor example in which optical drift imitates motion. The reproducible environment is recorded in [`environment.yml`](environment.yml).

Continue with [`notebooks/02_pdv_stft_neural_benchmark.ipynb`](notebooks/02_pdv_stft_neural_benchmark.ipynb) to generate synthetic PDV waveforms, compare 24 STFT configurations, train a small temporal CNN, and evaluate both methods on matched and deliberately shifted shots. Its saved run preserves the negative result: the conventional spectral ridge beat the first CNN, which is the starting point for the next controlled experiment.

Then use [`notebooks/03_physics_informed_motion_pinn.ipynb`](notebooks/03_physics_informed_motion_pinn.ipynb) to compare an ordinary trajectory network with a physics-informed neural network using the same architecture and measurements. A negative control deliberately supplies the PINN with a wrong motion equation so the notebook demonstrates both the value and the risk of physics constraints.

The development draft [`notebooks/04_robust_pdv_tracking_draft.ipynb`](notebooks/04_robust_pdv_tracking_draft.ipynb) adds harder diagnostic regimes, a continuity-aware conventional baseline, a residual CNN, repeated seeds, uncertainty, ambiguity prediction, and a locked-test switch that remains off by default.

The older browser lesson remains available by double-clicking `Learn.command`. [Learning Lab guide](docs/LEARNING_LAB.md) · [Learning Lab code](src/factor/learn/)

The page runs on this computer at `127.0.0.1`; it does not call a cloud model. Every experiment saves an evidence folder under `runs/learning-lab/`. JAX is an explicitly optional next step and no JAX result is claimed until its parity lesson is installed and run.

## Try the PyTorch module

Run `./PyTorch.command` for the interactive local model, or `./PyTorch.command train --steps 8` for a small training experiment. On a fresh machine, install the optional learning dependency with `python -m pip install '.[learning]'`. [Beginner tutorial](docs/PYTORCH_FOR_BEGINNERS.md) · [Cool experiments](docs/COOL_PYTORCH_EXPERIMENTS.md) · [Open the code](src/factor/torch_lab.py) · [Technical guide](docs/PYTORCH_LAB.md). Uses the prepared `.venv-local` environment and current checkout.

## Run one complete local example

In this prepared workspace, run:

```sh
.venv/bin/factor doctor
.venv/bin/factor run --request configs/motion_request.json --out runs/first
```

After installation, this conventional workflow needs no model, account or network. On a fresh machine, create an environment and install the dependencies first; see [release instructions](docs/RELEASE.md). Its result is **ambiguous**: optical drift can reproduce the apparent motion, and the input supplies no defensible drift bound. Each run saves the request, numerical result, evidence references, actions, usage and final decision. Existing results are preserved.

To use the locally served model explicitly:

```sh
.venv/bin/factor run --request configs/motion_request.json --out runs/local \
  --provider local --model factor-qwen35-9b --policy configs/local_policy.json
```

The model must be loaded in LM Studio at the configured loopback address. Factor does not silently load another model or send requests to a cloud service. [Local inference and actual post-training](docs/LOCAL_MODELS.md) records weights, revisions, recipes and results.

## Call it from another program

```python
import factor
from factor.providers import ConventionalAgent
from factor.runtime import ScientificRuntime

runtime = ScientificRuntime(ConventionalAgent(), root="runs/sdk")
result = factor.run(request_dict, runtime=runtime)
assert result["execution"] == "completed"
decision = result["final"]["decision"]
```

`request_dict` follows `configs/motion_request.json`. Scientific ambiguity is a completed result; failed calls, exceeded budgets and cancellation are execution failures. Inspect both. Evidence-ID validation establishes existence, not scientific support for generated prose.

## Evaluate before expanding

```sh
.venv/bin/factor suite --out runs/new-suite --n 24 --split development
.venv/bin/factor evaluate --suite runs/new-suite --out runs/conventional-comparison
.venv/bin/python -m unittest discover -s tests -v
```

The suite freezes a contract and separates public inputs from evaluator truth. Candidates receive public observations only. Scoring occurs after runs, records exposure and separates matched and misspecified regimes. This is application-level separation: a local operator can still read the files. Released cases and generators are public development material, not secret tests.

This fixed-window task differs from the original minimum-compression-depth task. The example uses 5 nm resolution and the benchmark uses 10 nm; both are provisional. Neither task establishes melting or an instability mechanism. New suites use the [explicit v2 contract](docs/AGENT_EVALUATION_V2_CARD.md); the [v1 contract](docs/AGENT_EVALUATION_CARD.md) and its [conventional results](reports/motion-development/BASELINE_REPORT.md) remain available.

## What has actually run

| Component | Evidence and boundary |
| --- | --- |
| Original implementation | 33 tests and 240-case reproduction completed before extension. Unmodeled drift failure preserved: 0/40 compression-depth intervals covered truth. |
| Scientific runtime | 120 implementation tests pass; a real local-model loop completed seven actions and a 12-case job in 56.337 seconds. |
| Local agent evaluation | 144 assigned attempts across two models, direct/tool modes and two repeats. Nemotron: 4/18 → 14/18 correct with tools; Qwen: 10/18 → 11/18. Conventional bounded baseline: 18/18 conditional decisions; hidden-mismatch failures remain. |
| Local open-weight inference | Actual Qwen3.5-9B Q4_K_M serving. Output-routing failures and usable-but-wrong scientific answers retained. |
| Small-model post-training | Actual Qwen3-0.6B generative LoRA: authored provenance classification improved 15/48 → 33/48; two regressions and fifteen errors remain. Four held-out families, one seed; no physics-transfer claim. |
| Numerical jobs | Real local simulations, collection, cancellation, timeout and restart checks. |
| Frontier APIs | Bounded OpenAI Responses adapter and fixtures. Live comparison needs credentials and an authorized budget. |
| Slurm | Submission, accounting, cancellation and collection implemented and fixture-tested. No live cluster/allocation supplied. |
| Release decisions | Deterministic local candidate gates; no autonomous production deployment. |

See [the build report](reports/BUILD_REPORT.md), [the model comparison](reports/AGENT_V2_COMPARISON.md) and [release instructions](docs/RELEASE.md) for results, reproduction and remaining gaps. Passing software checks is not validation on independent real data.

## Design and operation

- [Program design](docs/PROGRAM_DESIGN.md)
- [Local and Slurm jobs](docs/HPC.md)
- [Frontier API configuration](docs/FRONTIER_APIS.md)
- [Factory gates](docs/FACTORY.md)
- [Video notes and timestamped sources](docs/VIDEO_WORKFLOW_NOTES.md)
- Original liner-stability research notes and legacy commands are kept locally and are not part of the public repository.

The automatic captions for [the requested interview](https://www.youtube.com/watch?v=xgkjtF89-44) informed the design: define outcomes and call paths, build a complete small slice, then evaluate the feedback and review process. Notes contain timestamped paraphrases and source limitations.

Next external inputs: a permitted API project and total spending cap, a scheduler/account with resource ceilings, and eventually one complete diagnostic/calibration record. Factor never operates laboratory equipment or changes facility settings.


## LM-testbed world-model course

Continue the PyTorch notebooks with ten new lessons, **05–14**, building a synthetic experiment environment and the language-model tools around it. The progression covers machine dynamics, diagnostic response, joint inference, uncertainty, learned world models, attention, a tiny causal LM and a complete synthetic inference loop.

[Start the course](START_WORLD_MODEL.md) · [Open notebook 05](notebooks/05_experiment_contract_and_config.ipynb) · [Course guide](docs/WORLD_MODEL_COURSE.md)
