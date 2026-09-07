# Architecture and extension plan

## Responsibilities

**LM-testbed** owns the learning environments, models, data contracts and evaluation. **Factor** is the scientific application that can call those components, attach evidence and present a conditional decision. The user-facing application should not need to know whether a prediction came from RK4, a learned surrogate or a calibrated external solver; the tool result must declare its method and limitations.

```text
MachineConfig + uncertain parameters
                  ↓
          simulate(theta, config)
                  ↓
     hidden state [shot, time, state]
                  ↓
       observe(state, acquisition)
                  ↓
           raw diagnostic record
                  ↓
        reconstruction + covariance
                  ↓
         posterior / inverse model
                  ↓
      predictive checks and decisions
                  ↓
         Factor application + LM
```

The raw response path is implemented in lesson 07. For speed and clarity, lessons 08–14 use a separate authored Gaussian summary operator rather than propagating lesson 07 reconstruction uncertainty. That is an explicit unfinished connection in the long-term framework, not a calibrated raw-data pipeline.

## Files

- `lm_testbed/world.py`: configuration validation, differentiable RK4 simulator, raw diagnostic operators and reconstructed-summary benchmark.
- `lm_testbed/learning.py`: training helper, discrete posterior, weighted intervals and unique run records.
- `lm_testbed/language.py`: vocabulary, semantic data partitions, causal Transformer, generation and constrained call scoring.
- `lm_testbed/runtime.py`: observation-only bounded dispatcher, provenance hashes and conditional inference.
- `notebooks/05–14`: curriculum, visible model/training code, plots, comparisons and exercises.
- `tests/test_world_model.py`: physics-limit, reproducibility, identifiability, causal-mask and tool-boundary checks.

## Interfaces worth preserving

1. Forward models accept configuration and parameter tensors and return declared state names, time and trajectories. Add a solver/model version before comparing implementations.
2. Observation operators accept trajectories plus acquisition settings and return recorded quantities with time axes and calibration metadata.
3. Reconstructions should return values, uncertainty, valid masks and provenance. The present summary likelihood is a teaching stand-in for that future richer interface.
4. Inference tools should accept only permitted observations and assumptions. Results contain estimates or abstention, diagnostics, evidence references and conditional assumptions.
5. LM policies propose schema-valid calls. The runtime executes allowed numerical functions and enforces required information. No generated program is evaluated as code.
6. Evaluators receive hidden truth separately and score estimates, uncertainty, abstention, tool choice, cost and failure modes. Notebook-level separation is not adversarial isolation.

## What the present world does not contain

There is no facility-calibrated drive waveform, transmission-line network, resistive diffusion, equation of state, phase transition, radiation transport, plasma evolution, axial structure or stagnation/fusion model. The perturbation amplitude is passive and does not feed back onto circuit or motion. Its growth law is a generic authored demonstration and cannot discriminate ETI from MRTI.

The simulator includes a consistent mean-motion/circuit coupling and an energy-balance check within its stated simplifications. Configurable parameters are dimensionless teaching values. Dimensional scaling requires deriving reference units and rechecking every equation and measurement operator, not just relabeling axes in MA or ns.

## Next development increments

| Increment | Concrete deliverable | Evaluation before expansion |
|---|---|---|
| Connect raw diagnostics to inference | Reconstructed values plus covariance/validity masks from waveform and image operators | Recovery and coverage across calibration, timing, dropout and blur conditions |
| Add time-dependent machine inputs | Circuit input waveform and additional circuit sections | Numerical convergence, balances, known limiting cases |
| Add spatial state | A documented 1D/2D material/field model with explicit assumptions | Independent solver comparisons and parameter-domain checks |
| Improve posterior expressiveness | Correlated or mixture posterior over machine and nuisance parameters | Simulation-based calibration, per-regime coverage, multimodality tests |
| Learn an observation/state-space world model | Encoders for raw diagnostics and a latent dynamical model | Prediction from partial observations, missing-channel and long-horizon errors |
| Expand LM learning | Pretrained-model adapter, tool-use traces, broader tasks | Frozen semantic holdouts, rule baseline, invalid actions, costs and failures |
| Connect Factor | Adapter mapping Factor requests to this environment's observation/tool contracts | Application-level trace/replay and unchanged numerical results |
| Introduce real measurements | Immutable raw/calibration records and independent reference questions | Explicit domain-gap assessment before any physics claim |

The capstone selects synthetic diagnostics under an authored information objective. It does not select real Z settings or operate equipment.
