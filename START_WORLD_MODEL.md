# LM-testbed: the synthetic experiment course

**Start with [notebook 05](notebooks/05_experiment_contract_and_config.ipynb).** Open it in VS Code or Jupyter and select **Python (Factor AI)**. The saved notebooks include executed outputs and plots. Run from the top to make a new experiment.

Factor is the scientific application layer. LM-testbed is the place to learn the machine learning, language models, software interfaces and evaluation that support it. These ten lessons continue the existing PyTorch/PDV notebooks 01–04.

| Lesson | Build | Learn |
|---|---|---|
| [05 — Experiment contract](notebooks/05_experiment_contract_and_config.ipynb) | Machine → diagnostics → inference interface | Configurations, validation, tensors, provenance |
| [06 — Differentiable machine](notebooks/06_differentiable_machine_and_liner.ipynb) | Coupled circuit, liner motion and generic perturbation | ODEs, autograd, numerical checks, energy accounting |
| [07 — Synthetic diagnostics](notebooks/07_synthetic_diagnostic_suite.ipynb) | B-dot-like, PDV, image, radiation and spectrum records | Response models, calibration, timing, reconstruction |
| [08 — Joint inference](notebooks/08_joint_inference_and_identifiability.ipynb) | Infer machine parameters and a diagnostic gain | Likelihoods, identifiability, correlated timing errors |
| [09 — Probabilistic inverse model](notebooks/09_amortized_probabilistic_inference.ipynb) | Learn parameter estimates and uncertainty | Conditional density estimation, calibration, mismatch |
| [10 — Learned world dynamics](notebooks/10_learned_dynamics_world_model.ipynb) | Predict a state trajectory recursively | Residual models, rollout error, physical baselines |
| [11 — Multimodal attention](notebooks/11_multimodal_attention_and_missing_data.ipynb) | Combine diagnostic tokens with missing channels | Embeddings, attention, masking, ablation |
| [12 — Tiny language model](notebooks/12_tiny_language_model_for_tool_calls.ipynb) | Train a causal Transformer to generate tool calls | Tokenization, next-token loss, structured generation |
| [13 — LM + numerical tools](notebooks/13_language_model_tools_and_evaluation.ipynb) | Run and evaluate a bounded scientific agent | Schemas, tool use, evidence traces, local model adapter |
| [14 — Complete experiment loop](notebooks/14_closed_loop_experiment_world.ipynb) | Select a synthetic diagnostic and update inference | Information gain, sequential inference, predictive checks |

Each lesson includes explanatory text, editable code, a comparison or failure control, saved run records and exercises. The defaults run on a laptop CPU without downloads or API keys. Notebook 13 includes a separately disabled adapter for an already-running local language model.

This is a **dimensionless Z-inspired teaching world**, restricted to a pre-stagnation interval. It is an extensible first implementation, not a validated simulation of the Z machine. The distinction between the physics model, instrument response, reconstruction and inference is deliberate.

[Course guide and setup](docs/WORLD_MODEL_COURSE.md) · [Architecture and extension points](docs/WORLD_MODEL_ARCHITECTURE.md) · [Executed results](reports/WORLD_MODEL_VERIFICATION.md)

## More ways to learn visually

The course now includes **54 static figures and two embedded animations**, with 27 new visual labs. Look for “Visual lab” headings and their Predict / Notice / Ask prompts. Notebook 06 plays a synthetic shot; notebook 12 plays token generation.

[Visual learning guide](docs/VISUAL_LEARNING_GUIDE.md)
