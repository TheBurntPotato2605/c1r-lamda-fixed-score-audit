# Excluded material

An explicit file allowlist was used; nothing else was copied. The following were deliberately **not** published.

| Excluded | Reason |
|---|---|
| Raw LAMDA data (feature vectors, labels, APK hashes, partitions) | Third-party dataset; obtain it from the pinned revision (see README) |
| Per-application / per-row sealed scores, score files, model checkpoints, keys tables | Sealed study artifacts; per-application outputs are not for release. Only SHA-256 digests of the 20 score files are published |
| Internal authorization files (`G4_AUTHORIZATION*.json`, `G5_AUTHORIZATION.json`), approval and deviation records, review packages, chat-relayed decisions | Internal approval records; the manuscript cites them by hash prefix |
| Annotation files `RG-N.json`, `RG-S.json`, `RG-T.json` | Contain internal authorization/governance labels; not needed for the published tables, bounds or figures |
| `g3_summary.json` and other G3 run files | Contain row-level provenance entries beyond aggregate counts; only `G3_AGGREGATE_TABLES.csv` is published |
| Internal pipeline code (`g4_*.py`, `metrics_g5.py`, `g5_eval.py`) | Operates on sealed data and imports internal gate modules; not executable from the public package. Pinned hashes are cited in the manuscript |
| Independent reviewer's verification scripts and reports | Authorship/redistribution not established for this release |
| LAMDA code and Springer LNCS class files | Third-party; license not established for redistribution (LAMDA code) / not part of this package |
| Manuscript sources (LaTeX, Word) | Not part of this release |
| Any local machine path, user name or account detail | Privacy; scripts use repository-relative paths |
