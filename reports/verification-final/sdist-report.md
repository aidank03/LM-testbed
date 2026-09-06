# Source distribution completeness

The source archive now includes the experiment scripts and small inputs needed by the full test suite, both dependency lock files, frozen partitions, the original training-source snapshot and manifest, and the Qwen third-party text license. `MANIFEST.in` explicitly excludes model weights, local environments, and model caches.

Built offline from a temporary source copy, then extracted into another temporary directory: **120 tests passed in 11.282 seconds**, with `ResourceWarning` treated as an error. All 19 required additional fixture/input files and the packaged source/test files matched the repository hashes. The archive contains 110 regular files, no weights or environments; small exclusion probes were also absent. No models were downloaded, trained, or called, and no remote HPC job was submitted.

The archive is `dist/liner_stability-0.3.0.tar.gz` (645,516 bytes), SHA-256:

```text
31115d6ab61f80478878b3adbecb630120c0a9f188b82cca8bf2fc9a10dc600b
```

The previously tested wheel was neither rebuilt nor replaced; its SHA-256 remains `63afb852fa825639bde4c90eef1ab818bd78eccdb9cf15005bf3cb55518a226f`. No package source or test files changed. The source archive supplies code and test inputs; full measured run traces and the trained adapter belong to the separate source/evidence bundle.

Evidence: [archive contents and hashes](sdist-verification.json), [packaged test output](sdist-unittest.txt), [build log](sdist-build.txt), and [reproduction procedure](sdist-check.py).
