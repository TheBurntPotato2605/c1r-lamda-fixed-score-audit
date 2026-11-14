# Changelog

## 2.1.0

Added
* `pipeline/operating_points/`: supplementary, post-hoc evaluation metrics (non-interpolated AP; maximum empirical TPR at
  FPR <= 1% and <= 0.1%): verified metric core, driver, independent verifier, synthetic tests (run by `pipeline/run_checks.py`).
* `data/operating_points/`: archived supplementary-metric results (480 unit-arm rows, exact-fraction results, summary).
* `evidence/`: summaries of the real-data reproduction of v2.0.0, of the same-seed reference check and of the
  supplementary-metric verification; validation of the v2.1.0 code against the reproduced outputs.
* `pipeline/make_manifest.py` (deterministic manifest generation and check); `CHANGELOG.md`.

Changed
* `README.md`, `pipeline/README.md`: purpose, methodology, reproduction results and their tested-environment scope,
  installation and execution; statements that real-data fitting had not been performed were removed.
* `PROVENANCE.md`, `EXCLUSIONS.md`, `LICENSE_STATUS.md`, `CITATION.cff`.
* `MANIFEST.sha256` regenerated (the previous manifest no longer matched `README.md` and `LICENSE_STATUS.md`).

Unchanged: frozen scientific sources, `pipeline/c1r_e2e.py`, `src/`, published G3, G5 and S0-S4 aggregates and pins.

## 2.0.0 (commit 02b4566bb6e8d5db0eec516ccce7e496fcc49d9c)

End-to-end pipeline on the frozen study sources, synthetic verification, real-data stages through cohorts.
