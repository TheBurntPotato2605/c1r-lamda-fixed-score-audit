# End-to-end pipeline

Public execution path from the pinned LAMDA release to the primary and supplementary evaluation results, tables and
computational result figures of the manuscript. The scientific code is the **frozen C1-R implementation** (not a re-implementation):
see [PARITY.md](PARITY.md). The aggregate-only workflow of the repository root (`src/reproduce_all.py`) remains
available and independent of this pipeline.

## Verification status (read this first)

| # | Status | State at this release |
|---|---|---|
| 1 | Pipeline implementation complete | **Yes**: all stages implemented through the frozen code |
| 2 | Clean-clone installation and synthetic verification passed | **Yes**: synthetic end-to-end 13/13, source parity PASS, aggregate workflow PASS (logs: release assets) |
| 3 | Full real-dataset execution completed | **Yes** (release 2.0.0, all stages, 20 fits): 20/20 score files and seals bit-identical to the original sealed results, four aggregate files byte-identical, three published tables identical, S0-S4 independent check PASS (8,530 checks). Summary: `evidence/study_records/real_data_reproduction_summary.json` |
| 4 | Scope | Exact reproduction **in the tested environment** (reference laptop, Python 3.13.3, offline hash-pinned installation of the lock, locally verified data). Other hardware, GPUs, drivers and operating systems have not been tested |

Earlier real-data logs of release 2.0.0 (download, verify, cohorts on a fresh public download) remain in
`verification/real_dataset_stages_through_cohorts/`.

## Requirements

* **Software:** Python 3.13.3 and the exact packages in `environment/requirements-pipeline.txt` (hash-pinned original Windows/CUDA fitting lock:
  `environment/requirements-g4.lock`; see the root README for staged installation and the unverified online-install caveat). The frozen G3 stage checks Python 3.13.3 and pyarrow 25.0.1 and stops otherwise.
* **Hardware of the reference runs:** Intel i5-12450H (12 logical CPUs), 16 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB),
  driver 566.07, CUDA 12.6 wheels. XGBoost (`device=cuda`) and the MLP need a CUDA GPU; SVM and LightGBM are CPU-only.
* **Resources (per slot, original caps):** SVM 4 GiB RAM / 2 h; LightGBM 8 GiB / 3 h; XGBoost 5 GiB RAM + 3.8 GiB GPU / 3 h;
  MLP 5 GiB RAM + 2 GiB GPU / 1.5 h; about 4 GiB disk for all runs; 232 MB for the dataset. Observed wall time on the
  reference machine: SVM ~170 s, MLP ~240 s, XGBoost ~610 s, LightGBM ~390 s per run, about **1.95 h for all 20 runs**
  (plus ~2 min for `cohorts`, ~5 min for `download`).
* **Windows drive guard:** the frozen G3 core refuses data/work paths on drive `D:` (inherited guard); use another drive.

## Commands

```
python pipeline/c1r_e2e.py download     --work-dir WORK            # 24 pinned files, size + SHA-256 verified (--dry-run: metadata only)
python pipeline/c1r_e2e.py verify       --work-dir WORK            # frozen G2 verification (footers, schema, rows)
python pipeline/c1r_e2e.py cohorts      --work-dir WORK            # frozen G3 core: cohorts, keys/seen, conflicts; checked against pins
python pipeline/c1r_e2e.py train        --work-dir WORK            # 4 models x 5 runs (or --model svm --run 3)
python pipeline/c1r_e2e.py evaluate     --work-dir WORK            # exact AUCs, estimands, bounds, oracle (4 aggregate JSON)
python pipeline/c1r_e2e.py sensitivity  --work-dir WORK            # S0-S4 (S0 must equal the published primary values)
python pipeline/c1r_e2e.py tables       --work-dir WORK            # historical Tables 4-6, final cohort table and bounds
python pipeline/c1r_e2e.py figures      --work-dir WORK            # Figures 2-3 (matplotlib)
python pipeline/c1r_e2e.py compare      --work-dir WORK --require-exact  # strict reference-artifact comparison
python pipeline/c1r_e2e.py all          --work-dir WORK --require-exact  # only if all dependencies are already installed
```

For PowerShell, set `$P = ".\.venv\Scripts\python.exe"` and replace `python` with `& $P`. Install plotting
requirements after the fitting stage as shown in the root README. To reuse locally verified data, add
`--source-dir <directory-containing-the-24-pinned-files>` to the `download` command; size and SHA-256 checks remain active.
Stop at any failed stage. Use a dedicated disposable work directory, not the archive of the original runs.

Seeds: run *r* uses model seed *r* (r = 1..5), exactly as in the study. Model settings: `configs/models.json`.
Existing `SUCCEEDED` slots are kept unless `--force`. The original sealed results are never modified (`compare` only reads
`data/g5_aggregates` and `pipeline/reference/original_seals.json`).

## Expected outputs and numerical criteria

| Artifact | Criterion |
|---|---|
| Downloaded files | size and SHA-256 equal `reference/dataset_files.json` (hard stop) |
| Membership digest, 13 cohort sizes, keys-table SHA-256 | exactly equal to the frozen pins (hard stop) |
| `G3_AGGREGATE_TABLES.csv` | byte-identical to `data/g3_aggregates/` |
| Score seals (`runs/*/scores.sha256`) | compared with `reference/original_seals.json`; reported per slot. Bit-identical seals are the strongest outcome, but **bit-identity on other hardware is not guaranteed and has not been tested**: CPU models (SVM, LightGBM) use pinned library builds with deterministic settings, GPU models (XGBoost, MLP) can differ in the last bits across GPUs/drivers. Actual score-file hashes and recorded seals are compared separately; a mismatch against the original can still allow aggregate agreement, but prevents strict reference-artifact equality |
| Aggregate JSON (`eval/`) | `compare` reports aggregate `EXACT` (byte-identical to `data/g5_aggregates`), else `WITHIN_TOLERANCE` (structure equal and every AUC-scale value within `--tolerance`, default 5e-5, the half-unit of the fourth decimal in which Tables 5-6 report AUC values) with a listing of the largest deviations, else `DIFFERS` |
| S0 | `wp1_sensitivity.py` Gate B requires exact equality of S0 with the stored per-run values and the published Table 5 cells at 4/2 decimals; a hardware-dependent shift that changes a published digit stops the stage and is reported, never hidden |
| Historical Tables 4-6 | `compare` lists whether each regenerated historical table file equals its archived reference in `outputs/tables/` |
| Final cohort table | `table3_cohorts_manuscript.tex` uses the same exact counts and upward-rounded bounds; historical `table4_cohorts.tex` is preserved byte-for-byte |

`compare.json` retains `status` for the four aggregate files, with explicit `status_scope`. It independently hashes the
actual score files and checks agreement of the recorded seals with successful job records (the evaluation stage
recomputes the semantic seal from score rows). A local file/record inconsistency fails comparison even if the aggregates
match. `reference_artifacts_exact` requires the canonical reference, all 20 pinned score files and recorded seals, all four
aggregate files and all three historical tables. Missing tables and custom comparison references cannot satisfy this
predicate. `--require-exact` turns it into a hard success criterion. This comparison is not itself a new training run or
proof of bit-identity across platforms. All published historical evidence is unchanged.

## Supplementary evaluation metrics (added in 2.1.0)

```
python -X int_max_str_digits=0 pipeline/operating_points/operating_points.py --work-dir WORK [--compare-dir data/operating_points]
python -X int_max_str_digits=0 pipeline/operating_points/verify_operating_points.py --work-dir WORK
python -I -B pipeline/operating_points/tests/test_operating_points.py
```

Run after `train` and `evaluate`. AP (non-interpolated) and the maximum empirical TPR at FPR <= 1% and <= 0.1% per score
table, unit and arm, with tie groups taken whole; the operating points are selected in sample. `metric_core.py` is
byte-identical to the verified core (SHA-256 `083a1c6b...bf6`); the driver keeps the verified evaluation, aggregation and
output functions verbatim. Exploratory: these metrics do not replace the ROC-AUC analysis, and no ROC-AUC bound applies to
them. `-X int_max_str_digits=0` is required because exact AP fractions exceed Python's default 4,300-digit limit.

## Tests

```
python pipeline/run_checks.py                          # all default checks, pinned environment; --with-figures adds plotting
python pipeline/run_checks.py --no-network --skip-synthetic  # local standard-library checks
python pipeline/verify_source_parity.py                # frozen sources vs original (hashes, ASTs, job call sequence)
python pipeline/tests/test_synthetic_e2e.py            # synthetic end-to-end (pinned environment, no network, no GPU)
python -I src/reproduce_all.py --skip-figures            # aggregate-only workflow; omit flag after installing plots
python -I -S -B pipeline/tests/test_reproduction_checks.py  # lightweight integrity tests
python pipeline/c1r_e2e.py download --dry-run --work-dir WORK   # remote metadata equals the pins (no download)
```

The independent S0-S4 verifier runs with `-I -S` (isolated Python without site initialization); its original arithmetic
implementation and its check for unused `fractions` are unchanged. This avoids unrelated startup imports from the local
Python environment. When invoking it directly, use `python -I -S src/wp1_independent_check.py --results-dir <fresh-results>`.
Do not point it at the historical archived results directory because it writes a new verification report there.

The synthetic test builds a LAMDA-shaped dataset (no LAMDA data), runs verify, cohorts (the frozen G3 core must reproduce an
**independently built** keys table byte for byte), all 20 train slots, evaluate and compare, and checks: CPU determinism
(identical seals after retraining), 480 unit-arm AUCs against an independent exact implementation, `compare` EXACT against
itself and DIFFERS against the real reference (negative control), and that no computation stage opens a reference result.

## Not published

Per-row keys/seen table (it is regenerated and hash-checked), sealed score tables, checkpoints and the private authorization
system. The authorization gate of the original runner is replaced by `public/g4_access.py`, a no-op execution interface (PARITY.md).
