# Architecture

The core command is `liner-stability run`. A validated JSON configuration selects the synthetic experiment. The workflow produces observations, invokes the diagnostic analysis, scores it against separate synthetic truth, and writes an evaluation card with provenance. Each run has a new output directory and an explicit running/completed/failed state.

| Module | Responsibility |
| --- | --- |
| `simulation.py` | Prescribed physical state and synthetic measurements; separate truth return |
| `diagnostics.py` | Measurement reduction, blur correction, uncertainty and abstention |
| `evaluation.py` | Strict prediction contract and per-scenario scorecards |
| `reporting.py` | Reproducible case generation, saved examples and figures |
| `workflow.py` | Config validation, run lifecycle, evaluation card and package hashes |
| `physics.py` | Explicit-unit analytic reference calculations with assumptions |
| `design.py` | Candidate ranking from user-supplied predictions and covariance |
| `evidence.py` | Explicit-file indexing, lexical ranking, source hashes and citation-ID checks |
| `benchmarks.py` | Public development cases and separate deterministic grader |
| `llm.py` | Structured model requests, output validation, completion/refusal handling and benchmark execution |
| `io.py` | Atomic result writes and JSON helpers |
| `cli.py` | User commands |

The evidence packet can include a deterministic analytic result. Optional model synthesis reads the packet and produces a provisional conclusion, assumptions, unresolved alternatives and a next measurement. Source entailment and physical adequacy are not delegated to the output schema.

The candidate benchmark runner accepts a narrow public prompt contract. It has no evaluator-key argument. The grader is a separate operation. This separation avoids accidentally adding the answer key to an API request, but public source distribution still means the shipped cases are development-only. A real hidden suite needs external data separation and independent review.

The repository does not execute model-written code, automatically fetch papers, connect to a physical facility, or fit an MHD solver. Future solver integrations should implement a measurement-generating interface and pass verification tests before being used as synthetic truth.
