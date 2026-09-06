# Contributing

Install the package in a virtual environment, run `python -m unittest discover -s tests -v`, and run a small configured experiment before changing interfaces. Add a regression test when a change addresses a concrete numerical, data-contract or integration failure.

Keep experiments reproducible: record input hashes, physical units, timing/calibration, material tables, solver versions and exclusions. Do not mix frames or alternative reductions from a shot across development and test splits. Mark model-assisted labels explicitly.

Generated outputs, API keys and raw/private data are ignored by Git. Add only intentionally selected reference results to `examples/`. Never commit real data without its owner's authorization. Select a distribution license deliberately before public publication; none is declared by this private local project.

Document any proposed change in the scientific assumptions. A passing software test does not establish physical validity, and a provider fixture is not an LLM evaluation.
