# Changelog

## 2.1.1 — reproducibility maintenance

* Document the reference Windows installation and staged plotting setup; allow explicit aggregate-only verification.
* Verify manifest hashes from downloaded source archives without requiring a Git index.
* Run the unchanged independent rational-arithmetic check without Python site initialization.
* Separate aggregate comparison status from strict pinned reference-artifact equality; validate actual score-file bytes and recorded seals.
* Generate the final-manuscript cohort table with upward-rounded bounds beside the unchanged historical table.
* Add lightweight regression tests; report skipped checks explicitly.

No changes to frozen scientific code, archived results, reproduction evidence or historical artifacts.

## 2.1.0

Added
* `pipeline/operating_points/`: supplementary, post-hoc evaluation metrics (non-interpolated AP; maximum empirical TPR at
  FPR <= 1% and <= 0.1%): verified metric core, driver, independent verifier, synthetic tests (run by `pipeline/run_checks.py`).
* `data/operating_points/`: archived supplementary-metric results (480 unit-arm rows, exact-fraction results, summary).
* `evidence/`: summaries of the real-data reproduction of v2.0.0, of the same-seed reference check and of the
  supplementary-metric verification; validation of the v2.1.0 code against the reproduced outputs.
* `pipeline/make_manifest.py` (deterministic manifest generation and check); `CHANGELOG.md`.
* `LICENSE`: MIT License for the authors' original source code; `NOTICE.md` (license scope and third-party
  attribution) replaces `LICENSE_STATUS.md`.

Changed
* `README.md`, `pipeline/README.md`: purpose, methodology, reproduction results and their tested-environment scope,
  installation and execution; statements that real-data fitting had not been performed were removed.
* `PROVENANCE.md`, `EXCLUSIONS.md`, `CITATION.cff` (version 2.1.0, license MIT).
* `MANIFEST.sha256` regenerated (the previous manifest no longer matched two edited documentation files).

Unchanged: frozen scientific sources, `pipeline/c1r_e2e.py`, `src/`, published G3, G5 and S0-S4 aggregates and pins.

## 2.0.0 (commit 02b4566bb6e8d5db0eec516ccce7e496fcc49d9c)

End-to-end pipeline on the frozen study sources, synthetic verification, real-data stages through cohorts.
