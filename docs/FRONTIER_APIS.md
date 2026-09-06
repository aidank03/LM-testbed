# Frontier API execution

The new `OpenAIModel` uses Responses native function calls with registered schemas and parallel calls disabled. It records model/response IDs, payload hashes, usage and cost estimates. It does not grade itself or persist private reasoning. The old 0.2 adapter remains for compatibility; use the Factor runtime for budgeted agent experiments.

No API credentials or live provider access were found during this build. Fixtures establish implementation behavior, not provider capability. No frontier score, human baseline or paid run is claimed.

Configure `OPENAI_API_KEY` locally; do not paste it into prompts or source. Copy the two cloud example profiles to private files. Set an authorized total budget in both, enable policy data egress and verify applicable prices. Shipped examples permit no spending.

```sh
factor evaluate --suite NEW_FROZEN_SUITE --out NEW_RESULTS \
  --provider openai --model EXPLICIT_API_MODEL_ID \
  --cloud-profile PRIVATE_CLOUD_PROFILE.json --policy PRIVATE_POLICY.json
```

One adapter shares its reservation budget across cases/repetitions in this command. A new command creates a new ledger; this is not an account-wide billing cap. Unknown usage retains its reservation. Uncertain requests are not automatically retried.

Use equal observations and tools in model comparisons. Freeze cases, methods and criteria before final scoring; retain failures. Direct inference, retrieval, numerical tools and job loops are distinct workflows. Retrieval needs a separately validated corpus before an improvement claim. Latency and dollars omit unmeasured expert effort. A human comparison needs actual recorded answers and timing.

Example prices are zero placeholders, not free-service claims. Replace them with verified positive rates before enabling paid execution. No specific model ID is silently selected. The catalog does not prove this account has access.

Primary references checked 2026-09-05: [Function calling](https://developers.openai.com/api/docs/guides/function-calling), [model catalog](https://developers.openai.com/api/docs/models), [pricing](https://developers.openai.com/api/docs/pricing). Cross-provider portability is not yet tested.
