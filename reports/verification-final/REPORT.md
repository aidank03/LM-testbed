# Factor 0.3.0 final implementation and installation verification

**Final addendum:** after adding four offline checks for prospective training-hardware reporting, the complete suite passed **120 tests in 15.565 seconds** in the ready `.venv`, with ResourceWarnings treated as errors. [Combined addendum log](unittest-addendum.txt). The four checks concern the experiment script and its new metadata helper; installed package source did not change. The 116-test verification and installation evidence below remain the original record.

**All requested release checks passed on 2026-09-05.** The combined suite ran **116 tests in 21.515 seconds**, with `ResourceWarning` treated as an error. This includes the original 33 tests and 83 added tests. Source, test, and `pyproject.toml` hashes were identical before and after verification. These are implementation and installation checks; they do not establish scientific validity or an AI advantage.

| Check | Result and evidence |
| --- | --- |
| Combined tests | 116 passed; [unittest.txt](unittest.txt) |
| Versions | Package metadata, `factor.__version__`, and `liner_stability.__version__` all 0.3.0 |
| Offline wheel | Built from a temporary copy of the current repository, without dependency downloads or shared package changes |
| Installed entry points | Both `factor doctor` and `liner-stability doctor` passed; `liner-stability --version` returned 0.3.0 |
| Installed conventional request | Completed, returned justified ambiguity for unbounded optical drift |
| Default benchmark version | Installed `factor suite --n 2` produced `factor-cases/2`, 18 development cases |
| Installed conventional evaluation | Completed all 18 cases and produced `factor-method-comparison/2` |
| Corrupted public contract | Rejected before provider construction: 0 provider calls, 0 model calls, no evaluation directory |
| Installation isolation | Both package imports came from the new `.venv` installation, outside the repository source path |

The test breakdown is: original diagnostic/grading 13; original workflow/assistant 20; evaluation v1 8; evaluation v2 4; factory 8; runtime 16; model fetch 7; HPC 19; local model interface 8; PDV window 7; runtime/HPC boundaries 6. Within the HPC tests, three execute real local subprocesses for success, cancellation, and wall timeout; sixteen exercise fixtures and controls. The fetch tests use fixture bytes, and model/remote-scheduler tests use fixtures. **This verification made no live model calls, model downloads, remote HPC submissions, or training runs.** The separate closed-loop report records the actual local-model/local-job demonstration.

The v2 installation smoke test is a small, exposed development sample, not the final model campaign. The bounded conventional method followed the public decision contract on 18/18 cases and recognized all five required ambiguous cases. On matched assumptions it covered 10/10 reported numerical intervals, with MAE 2.567 nm. It nevertheless made four wrong decisive statements across the four hidden-mismatch cases and covered none of their intervals. All provisional matched/contract gates passed while this physical-truth failure remained visible. The deliberately naive baseline failed the expected accuracy, coverage, decision, and abstention gates. [Full smoke comparison](v2-conventional-comparison.json).

The deliverable is `dist/liner_stability-0.3.0-py3-none-any.whl`, 75,429 bytes, SHA-256:

```text
63afb852fa825639bde4c90eef1ab818bd78eccdb9cf15005bf3cb55518a226f
```

The ready core environment is `factor/.venv`, created with the bundled Python 3.12.14 and `--system-site-packages`, then installed from that wheel using `--no-index --no-deps`. Because the bundled runtime lacks SciPy and matplotlib, twelve numerical dependency distributions were copied offline into this new environment from inspected installed distributions: NumPy, SciPy, matplotlib, contourpy, cycler, fonttools, kiwisolver, packaging, Pillow, pyparsing, python-dateutil, and six. The resulting numerical versions are NumPy 2.3.5, SciPy 1.17.0, and matplotlib 3.11.1. The historical `.factor-verification-venv` still imports liner-stability 0.2.0; it was not modified. The separate `.venv-local` was not touched.

This is a machine-local convenience environment: it inherits other bundled-runtime packages, and copying or moving the virtual environment is not a portable installation procedure. Exact environment configuration, inherited package snapshot, dependency file hashes and origins are recorded in [environment.json](environment.json), [core-requirements-freeze.txt](core-requirements-freeze.txt), [core-dependency-copy-manifest.json](core-dependency-copy-manifest.json), and [installed-package-origins.txt](installed-package-origins.txt). Numerical dependency command-line scripts outside `site-packages` were not copied. Verification ran the installed Factor commands from a temporary working directory with the source `PYTHONPATH` removed; temporary smoke-run directories were discarded after retaining the comparison and suite.

[validate.py](validate.py) records the exact procedure and command arguments. It deliberately refuses to overwrite the existing `.venv`, wheel, or final evidence. Earlier reports under `reports/verification` remain available. Hashes in `environment.json` cover every Python file in `src/` and `tests/`, plus `pyproject.toml`.

The code is installed and its checked paths work. Correct uncertainty still depends on valid diagnostic assumptions; a working program cannot discover invisible drift from information it was never given.
