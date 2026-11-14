# Scientific source parity with the frozen C1-R implementation

The fitted models and every number of the study were produced by the frozen C1-R pipeline (G2 input verification, G3
core, G4 adapters/data layer/seal, G5 metric module and evaluation driver). This package ships those sources, not a
re-implementation. `python pipeline/verify_source_parity.py` checks the statements below.

## Frozen files (`pipeline/frozen/src/`, manifest `frozen/frozen_source_manifest.json`)

| File | Role | Relation to the original |
|---|---|---|
| `g3core.py`, `g4_adapters.py`, `g4_data.py`, `g4_seal.py`, `metrics_g5.py` | exact-vector keys; the four model adapters; verified data layer and cohorts; score seal; exact-rational metrics | **byte-identical** to the originals |
| `g2_analysis.py`, `g3_main.py`, `g4_common.py`, `g5_eval.py` | G2 verification; G3 driver; constants; G5 evaluation functions | identical abstract syntax tree; only string literals that held private machine paths (argument defaults, `REAL_PATHS`, `WORK`) are replaced by `<PRIVATE-PATH-NOT-PUBLISHED>`. Recorded: original SHA-256, published SHA-256, AST hash of the original with path literals normalised |

The original files' SHA-256 equal those in the manifests of **all 20 original runs**: run 1 was executed under the frozen
V6 sources, runs 2-5 under V6.1. V6 and V6.1 differ only in `g4_runner.py` and `g4_access.py` (authorization gate); none of the files
above differ between them (`frozen_source_manifest.json`, `_provenance`).

Because the frozen modules pin each other by SHA-256 (`g4_data` loads G2 and G3 only if their bytes equal the pins), the public
driver passes the hashes of the *published* G2/G3 files as the pins (`DataSpec(g2_sha=..., g3_sha=...)`, `G3M.G2_SOURCE_SHA`).

## Publication-specific wrappers (the only differences, none touches a scientific computation)

| Wrapper | Replaces | Difference |
|---|---|---|
| `public/g4_access.py` | original authorization gate (private approval records, global lock, ledger) | `assert_dataset_access` is a no-op. The gate decided whether a run may start; it never altered data, features, cohorts, models, seeds, fitting, scoring or sealing |
| `public/job_public.py` | original `g4_job.py` | removed: gate call, fault-injection hooks, runtime-pin comparison, phase/ledger files, resource monitor. The ordered scientific call sequence (seeding, adapter, `Dataset`, `load_train`, `fit`, `resolved`, `check_contract`, `save`, reload, two scoring passes, `seal_digest`, `write_table`, pass identity) is asserted equal to the original's by `verify_source_parity.py` (21 calls) |
| `c1r_e2e.py`, `g3` stage | original G3 command line | `ANCHORS` (hash anchors of private approval documents) emptied at run time; G2-run check replaced by comparison with the pinned digest and sizes afterwards; the inherited `D:`-drive/`android-malware-detection` guards are disabled in `g4_common` (`K.forbidden_path`) because they protected an unrelated private project |
| `evaluate` stage | `g5_eval.py` `--execute` | reuses its functions (`build_metadata`, `evaluate_runs`, `model_independent`, `e3_all`, `ser`) and writes the same four JSON files in the same format; omitted: authorization record, ledger/manifest evidence checks, annotation files RG-N/S/T |

Not published: the private authorization system, ledgers, approval records and the sealed per-row keys/score tables.
