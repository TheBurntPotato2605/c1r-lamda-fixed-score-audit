# c1r-lamda-fixed-score-audit

Aggregate data, analysis code and verification scripts for the study

> **Exact Feature-Vector Overlap and Measured Temporal Degradation: A Fixed-Score Audit of the LAMDA Android Malware Benchmark**
> (C1-R; manuscript in preparation, not yet submitted or published)

The study asks how much the IID-to-FAR ROC-AUC degradation of four baseline detectors on the LAMDA benchmark changes
when evaluation is restricted to rows whose exact feature vector never occurs in the training cohort. Both
populations are masks over the same sealed scores. All results are descriptive and conditional on the released
feature representation.

This repository publishes **aggregates and code only**. It is a verification package, **not** a full end-to-end
reproduction package (see section 3).

## 1. What can be reproduced directly from the published aggregates

Run from the repository root (Python 3.13; `matplotlib==3.10.3` only for figures):

```
python -I src/reproduce_all.py
```

| Step | Reproduces | Inputs | Check |
|---|---|---|---|
| `src/wp1_sensitivity.py` | S0 (primary) and exploratory post-hoc S1-S4: exact-rational re-weighting of the stored per-unit AUCs, 12 distinct score tables, count bounds | `data/g5_aggregates`, `data/provenance/score_table_sha256.json` | S0 must equal the stored per-run and summary values exactly; CSV is byte-identical to `data/wp1_aggregates` |
| `src/wp1_independent_check.py` | 8,530 checks by a second, structurally different rational-arithmetic implementation | the above plus the re-run output | status `PASS` |
| `src/make_tables.py` | Tables 4, 5 and 6 of the manuscript and the count bounds of Proposition 1 / Eq. (5) | `data/g5_aggregates`, WP1 results | `outputs/tables/*` (bounds are **rounded upward** wherever an inequality is printed) |
| `src/make_figures.py` | Figures 2 and 3 (full-arm AUC per unit; Full-vs-Unseen contrast and count bound) | `data/g5_aggregates` | plotted values equal `outputs/figures/plotted_values.json` |
| `MANIFEST.sha256` | integrity of every published file | - | verified first by `reproduce_all.py` |

Key numbers (exact values in `outputs/tables/bounds.json`): the FAR count bound of the primary analysis is
0.0208093... and is therefore printed as <= 0.0209; the exploratory bounds are 0.0238, 0.0275 and 0.0296 (rounded up).

## 2. What requires separately obtaining the pinned LAMDA dataset

Not included here and not reproducible without the original rows:

* the exact-vector membership of every evaluation row in the training cohort ("seen"/"unseen"), the class counts per
  arm, the conflict-group statistics and the oracle AUC bound (`G5_model_independent_counts.json`, `G5_E3.json`
  are published as *results* of that computation, not as something this repository can recompute);
* the cohort construction from the 24 annual partitions (cohort definitions: `configs/cohorts.json`).

Obtain the LAMDA Baseline configuration at the pinned revision `ad9614bdd5556767f97ced2fce797c2f06408ebf`
(<https://huggingface.co/datasets/IQSeC-Lab/LAMDA/tree/ad9614bdd5556767f97ced2fce797c2f06408ebf>); the authors'
public code is at commit `7728bfafcd5539a286b5f8c47b6f1e3b2d1f4249`
(<https://github.com/IQSeC-Lab/LAMDA/tree/7728bfafcd5539a286b5f8c47b6f1e3b2d1f4249>). The generic dataset DOI
`10.57967/hf/5563` identifies the dataset, not this revision. This repository ships no code to rebuild the
membership keys from the raw data; the frozen internal implementation is described, not published (section 3).

## 3. What cannot be independently reproduced from this public package

* **Model fitting and scoring.** The 20 sealed score tables (4 models x 5 runs), the model checkpoints, the per-row
  labels and keys, and the internal fitting/scoring/sealing pipeline are **not** published. Only the SHA-256 digests of
  the 20 score files are given (`data/provenance/score_table_sha256.json`), which establish that the 20 nominal runs
  correspond to 12 distinct score tables (SVM and XGBoost: 1 each; LightGBM and MLP: 5 each). Frozen model settings
  are in `configs/models.json`; the original environment pins are in `environment/`.
* **The per-unit AUCs.** They are published as stored aggregates computed from the sealed scores. Their recomputation
  from scores is not possible here; this package verifies everything downstream of them.
* **Governance records.** Internal approval records, authorization files and chat-relayed decisions are not published.
  The manuscript's governance table cites them by hash prefix. The scientific-code equivalence of the V6 and V6.1
  authorization-gate versions rests on source-level comparison, not on independent reruns, and no targeted-review
  sign-off for the evaluation driver is on record (see the manuscript).

Do not read this repository as an end-to-end reproduction of the study.

## Repository layout

```
configs/      cohort definitions and frozen model settings
data/g5_aggregates/     published primary aggregates (hash-pinned; SHA256SUMS.txt is the original, unmodified file)
data/g3_aggregates/     cohort/partition count tables
data/wp1_aggregates/    exact S0-S4 results and the independent-check record
data/provenance/        score-file digests (hash strings only)
environment/  package pins of the original runs
src/          analysis, verification, table and figure scripts (path handling adapted, see PROVENANCE.md)
outputs/      regenerated tables, bounds and figures
PROVENANCE.md, EXCLUSIONS.md, LICENSE_STATUS.md, MANIFEST.sha256
```

## Attribution and license

LAMDA is by the IQSeC Lab; see `LICENSE_STATUS.md` for links and the status of third-party and own material.
**No license has been selected by the authors yet**, so none is granted.
