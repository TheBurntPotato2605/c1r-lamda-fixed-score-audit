# Provenance

## Origin of the published aggregates

| Published file | Origin | SHA-256 |
|---|---|---|
| `data/g5_aggregates/G5_per_run_metrics.json` | unmodified copy of the archived study output | `b4616e70c70a992580147aa7372fde1367c7afef45a0c0074980884954f8004b` |
| `data/g5_aggregates/G5_model_summary.json` | unmodified copy of the archived study output | `58d4708a5280109a5558e82f1cec74ae0d9b490d7f799bc7eb42e0682965d8f0` |
| `data/g5_aggregates/G5_model_independent_counts.json` | unmodified copy of the archived study output | `97b1c4bec9fd655fd06d65db91e99dae6d49c04cfac9be75a80262575b616080` |
| `data/g5_aggregates/G5_E3.json` | unmodified copy of the archived study output | `4d66548b4aecc0bf21bf00f1bfb3ddff0cfe625ca5a29730716ce8ce8a1bb947` |
| `data/g5_aggregates/SHA256SUMS.txt` | unmodified copy of the archived study output | `b3223645f2890fc24687de54d17bbd9c686ff44604c99aec52f719532588068c` |
| `data/wp1_aggregates/WP1_S0_S4_RESULTS.json` | unmodified copy of the archived study output | `a2ba67331f78eb53c1c5b951b35f559d94251c66d479df2b19775defb8f03790` |
| `data/wp1_aggregates/WP1_S0_S4_RESULTS.csv` | unmodified copy of the archived study output | `0d40dd23cc1448efc3590531b20eaa441e271f5264bdf3902eef1ba89489f2fc` |
| `data/wp1_aggregates/WP1_INDEPENDENT_CHECK.json` | unmodified copy of the archived study output | `c3055cf3d5c490e9f8e3ae54a2a7e39881ccb6d02dae12527299fbc93511a8e1` |
| `data/g3_aggregates/G3_AGGREGATE_TABLES.csv` | unmodified copy of the archived study output | `956700c99f6c7d027d38ace9f4c9335cff18ff5be215d00f2ccfa96e304edcb8` |
| `data/provenance/score_table_sha256.json` | derived: the 20 score-file digests extracted from the internal G5 authorization record (not published); hash strings only | `18bdf24b859261dfd5e2b40591f946010aa2bc1f37f03bbf99f39bdc40ab081d` |
| `configs/models.json`, `configs/cohorts.json` | derived: frozen model settings (from the run manifests via an internal annotation file) and cohort definitions with counts from the published aggregates | see `MANIFEST.sha256` |
| `environment/original_run_env_freeze.txt` | package freeze of the environment used for the 20 fits (unmodified) | `b1583767fed7ea81b1ec899b99e952aef2bcc60dd8c46e5cf768f72557bfcc5d` |

`data/g5_aggregates/SHA256SUMS.txt` is the original file and also lists three annotation files (`RG-N`, `RG-S`, `RG-T`) that are intentionally **not** published; the scripts only check the files they read.

The primary G5 aggregates were produced on 10 November 2026 from the sealed scores of the 20 runs; the exploratory S1-S4
results were computed on 13 November 2026 from those aggregates (post-hoc, not preregistered). Values are exact rationals
("exact": "p/q") with float display copies.

## Scripts: original and adapted

The scientific computations are unchanged. Only path handling was adapted so that the scripts read the repository
copies instead of internal absolute paths, and display labels naming input files were updated accordingly
(`wp1_sensitivity.py`). `make_figures.py` additionally (i) reads four instead of seven pinned files (the three
annotation files are not used by the figures), (ii) omits the PDF creation timestamp, and (iii) in Figure 3 (finding F10)
places the legend below panel b on a canvas 0.25 in taller, keeping panel sizes, axes and every plotted value.

| Script | Original SHA-256 (internal) | Published SHA-256 |
|---|---|---|
| `src/wp1_sensitivity.py` | `dcdf5c58247fbdd12dadc9c86162cc812089a50fd94430525cc8a5c3be6f1251` | `09cf502981b9fc495ea745b6edda683dfa9a4140458c46bc82153e8c7d8a377b` |
| `src/wp1_independent_check.py` | `606515256c0c88d332d79e84f3f729632ce094e87cee5f68d0f13fd7b9149097` | `bd0b2c45a7da13c2190be81eb5c2d6a9e555f22a7fb286e2926bec6eda34fb30` |
| `src/make_figures.py` | `06665b4c20f71a537b58bd57e12e3bfebf420b82e9d68d6569081211679ce8da` | `d96086a2e43af5784651ba67547b4d9083ef33e8ff41b41e1a3e9871d4b4ba05` |
| `src/make_tables.py`, `src/reproduce_all.py` | new in this repository | see `MANIFEST.sha256` |

## Verification performed before publication

* `wp1_sensitivity.py` on the published aggregates: all gates PASS; CSV byte-identical and JSON identical (five provenance labels masked) to the archived results.
* `wp1_independent_check.py`: PASS, 8,530 checks.
* `make_tables.py`: every generated row is found verbatim in the corrected manuscript source.
* `make_figures.py`: plotted values identical to those of the original script; Figures 2 and 3 are rendered with matplotlib 3.10.3.

## End-to-end pipeline (added in release 2.0.0)

* `pipeline/frozen/src/`: the frozen C1-R scientific sources. Original SHA-256, published SHA-256 and AST-parity hashes per file, the SHA-256 recorded in the manifests of all 20 original runs (V6 for run 1, V6.1 for runs 2-5) and the original job's scientific call sequence are in `pipeline/frozen/frozen_source_manifest.json`; `pipeline/PARITY.md` explains the four files whose private path literals were neutralised and the public wrappers.
* `pipeline/reference/`: `dataset_files.json` (24 files: URL, size, SHA-256, rows) and `g1_inventory.json` (reduced copy of the original G1 remote-metadata inventory: size, footer length and footer SHA-256, row count, server x-linked-etag = file SHA-256) derive from the G1 inventory of 4 November 2026; `original_seals.json` holds the seal and file SHA-256 of the 20 original score tables (from the internal G5 authorization record, hash strings only).
* `pipeline/verification/`: logs of the stages executed on real data (download, verify, cohorts), the synthetic end-to-end log.
* `data/g3_aggregates/G3_AGGREGATE_TABLES.csv` was reproduced byte for byte by the public pipeline from a fresh download (together with the conflicts, coverage and estimability tables).

## Release 2.1.0: supplementary metrics and reproduction evidence

| File | Origin | SHA-256 |
|---|---|---|
| `data/operating_points/unit_arm_metrics.csv` | unmodified bytes of the archived supplementary-metric output (480 rows: model, run, unit, arm; aggregate metrics only) | `87bff35c8527f5635e2f1d969a54873efaea7e6fd151fb10b9b5557cadc72f31` |
| `data/operating_points/results.json` | unmodified bytes of the archived output (exact fractions; contains the archived analysis-record label) | `5e3e9833d3fcb3831025b5efd11924b48be14ca865925f3c81dcf3961631c138` |
| `data/operating_points/manuscript_summary.csv` | unmodified bytes of the independent verifier's summary (source of the manuscript table) | `5f6e483a85983216e143bcbb1cc2db7db49f2a5b2da80344a7cfc067f004dcf0` |
| `pipeline/operating_points/metric_core.py` | byte-identical copy of the verified metric core (its docstring keeps the archived record label) | `083a1c6b27b2f9ca2df9db922af569ef5ee7275765d43eb2e9a7c00b51e87bf6` |
| `pipeline/operating_points/operating_points.py` | driver: the verified evaluation, aggregation and output functions are AST-identical to the study driver apart from output file names and the exception name; the private authorization layer is replaced by the public work-directory context | see `MANIFEST.sha256` |
| `pipeline/operating_points/verify_operating_points.py` | adapted from the study's independent verifier (same numerical checks; private record checks removed); never imports the core or the driver | see `MANIFEST.sha256` |
| `pipeline/operating_points/tests/test_operating_points.py` | the study's synthetic tests of the core and driver (13 tests), import paths adapted | see `MANIFEST.sha256` |
| `evidence/study_records/*` | summaries derived from sealed records of the study; `real_data_reproduction_compare_report.json` is the unmodified `compare` report of the v2.0.0 reproduction | see `MANIFEST.sha256` |
| `evidence/v2.1.0_validation/*` | produced for v2.1.0: the v2.1.0 code run read-only on the reproduction's score tables (local paths replaced by a placeholder) | see `MANIFEST.sha256` |

The validation run regenerated `unit_arm_metrics.csv` byte for byte; the independent verifier regenerated
`manuscript_summary.csv` byte for byte.
