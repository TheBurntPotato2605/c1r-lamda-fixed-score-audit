"""C1-R G5 evaluation driver (minimal). Evaluates the 20 existing sealed G4 training results with the FROZEN V6
metric module (src/metrics_g5.py) under Manifest v0.2.4 section 7 / 8.3 and Protocol V2.2.1. No training, no inference,
no confidence interval, no p-value, no ranking, no threshold metric.

Modes (exactly one is required):
  --verify-only         identities, hashes, authorization, environment and the 20 evidence slots. Hashes bytes only:
                        NEVER decodes a score table, NEVER reads evaluation labels.
  --execute --out-dir D real evaluation. Refused unless the authorization record has authorized == true.
Everything is fail-closed: any missing, extra or mismatching item stops before any output is written. Output goes to a
new directory (never inside runs/); training artifacts are opened read-only and never modified.
"""
import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import os
import re
import sys
from fractions import Fraction

WORK = r"<PRIVATE-PATH-NOT-PUBLISHED>"
MODELS = ("svm", "mlp", "xgb", "lgbm")
RUNS = (1, 2, 3, 4, 5)
SLOTS = tuple(f"{m}_r{r}" for m in MODELS for r in RUNS)
AUTH_V1 = "5695d7b2f4d2e5f7edac8092eb1cc954f36e8aa0df3c06eebf98b24fd469652b"
AUTH_V2 = "7a51e2308917a17b547d3b4166cb7ae90e0ec84a5baf2628d1d8fa6cbeb0fbed"
GATE_FILES = {   # run 1 = frozen V6 ; runs 2-5 = V6.1 (G4_G5_V6_1_DEVIATION_DECISION_2026-11-10.md)
    1: {"g4_runner.py": "08a1af343ed29c04e875a3184c31520ed8c1e5e1f49228439dc4d0e73f584840",
        "g4_access.py": "7be6ccb910e7b7981bfc40c86fd6e6430e5b09ec32d60bdef6f2c92e6db0ba3f"},
    2: {"g4_runner.py": "228bdae7c00c19cf3e923782c671a0064b7a4ec83a61183d5c39fe0d70cfbe25",
        "g4_access.py": "76304eb3241aea7cc7cfe46edca1c3172cbde52be9db21006e5232d41dca4e97"}}
PINNED_SOURCES = ("metrics_g5.py", "g4_seal.py", "g4_common.py", "g4_data.py", "g4_access.py")
AUTH_KEYS = {"gate", "authorized", "driver_sha256", "source_sha256", "env_lock_sha256", "deviation_record_sha256",
             "evidence", "v1_authorization_sha256", "v2_authorization_sha256", "approved_by", "notes"}
EVID_KEYS = {"manifest_sha256", "scores_file_sha256", "seal_sha256"}
HEX = re.compile(r"^[0-9a-f]{64}$")
BANNED = re.compile(r"reproduced|faithful", re.I)           # Manifest 8.3: words prohibited in RG-N reporting
RGN_REF = {   # arXiv v1 Table 1 / ICLR 2026 Table 2, LAMDA ROC-AUC (%), mean, SD (Manifest 8.3; Supervisor double-entry check)
    "lgbm": {"IID": ("99.55", "0.03"), "NEAR": ("74.05", "23.76"), "FAR": ("78.04", "20.83")},
    "mlp": {"IID": ("99.48", "0.04"), "NEAR": ("82.71", "11.19"), "FAR": ("84.04", "11.23")},
    "svm": {"IID": ("98.89", "0.28"), "NEAR": ("75.18", "17.98"), "FAR": ("79.07", "15.06")},
    "xgb": {"IID": ("99.15", "0.16"), "NEAR": ("77.75", "16.85"), "FAR": ("76.85", "16.49")}}
RGN_ANNOTATIONS = [
    "cohort P-A differs from the paper's PC-A partitions",
    "the paper's active code path uses VarianceThreshold 0.0001 while Table 1 is labelled Baseline (0.001)",
    "XGBoost objective: explicit in this study versus default in the paper's code",
    "explicit seeds s_r = r (r = 1..5)",
    "unknown-month rows retained",
    "historical execution provenance (HP) of the paper's numbers is UNKNOWN",
    "paper SD is shown for context and is NOT a tolerance",
    "descriptive comparison only; no inference, interval, p-value or ranking",
    "run 1 executed under frozen V6 / authorization v1; runs 2-5 under V6.1 / authorization v2 (see deviation record)",
    "runs 2-5 were executed under a G4 label; the manifest assigns r=2..5 to G5 (see deviation record)"]


class G5Stop(Exception):
    """Fail-closed stop: evidence, authorization, environment or contract problem."""


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def need(cond, msg):
    if not cond:
        raise G5Stop(msg)


# ------------------------------------------------------------------------------------------ authorization
def load_auth(path):
    def pairs(pp):
        d = {}
        for k, v in pp:
            need(k not in d, f"duplicate key {k!r} in authorization")
            d[k] = v
        return d

    def no_const(c):
        raise G5Stop(f"non-finite literal {c} in authorization")
    need(os.path.isfile(path), f"G5 authorization record missing: {path}")
    raw = open(path, "rb").read()
    need(not raw.startswith(b"\xef\xbb\xbf"), "authorization must be UTF-8 without BOM")
    try:
        a = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=no_const)
    except ValueError as e:
        raise G5Stop(f"authorization is not valid strict JSON: {e}")
    need(isinstance(a, dict) and set(a) == AUTH_KEYS, f"authorization keys must be exactly {sorted(AUTH_KEYS)}")
    need(a["gate"] == "G5" and type(a["authorized"]) is bool, "gate must be 'G5' and authorized a boolean")
    for k in ("driver_sha256", "env_lock_sha256", "deviation_record_sha256", "v1_authorization_sha256", "v2_authorization_sha256"):
        need(type(a[k]) is str and HEX.match(a[k]), f"{k} must be a lowercase SHA-256")
    need(a["v1_authorization_sha256"] == AUTH_V1 and a["v2_authorization_sha256"] == AUTH_V2, "G4 authorization hashes differ from the recorded v1/v2")
    s = a["source_sha256"]
    need(isinstance(s, dict) and set(s) == set(PINNED_SOURCES) and all(type(v) is str and HEX.match(v) for v in s.values()),
         f"source_sha256 must pin exactly {PINNED_SOURCES}")
    e = a["evidence"]
    need(isinstance(e, dict) and set(e) == set(SLOTS), "evidence must pin exactly the 20 slots")
    for k, v in e.items():
        need(isinstance(v, dict) and set(v) == EVID_KEYS and all(type(x) is str and HEX.match(x) for x in v.values()), f"evidence[{k}] malformed")
    for k in ("approved_by", "notes"):
        need(type(a[k]) is str, f"{k} must be a string")
    return a


def check_driver_sources_env(a, work, driver_path):
    need(sha256_file(driver_path) == a["driver_sha256"], "driver SHA-256 differs from the authorized driver")
    src = os.path.join(work, "src")
    for n, h in a["source_sha256"].items():
        need(os.path.isfile(os.path.join(src, n)) and sha256_file(os.path.join(src, n)) == h, f"frozen source {n} differs from the authorization")
    lock = os.path.join(work, "env", "requirements-g4.lock")
    need(sha256_file(lock) == a["env_lock_sha256"], "dependency lock differs from the authorization")
    rec = os.path.join(work, "G4_G5_V6_1_DEVIATION_DECISION_2026-11-10.md")
    need(os.path.isfile(rec) and sha256_file(rec) == a["deviation_record_sha256"], "deviation decision record missing or changed")
    for n, h in (("G4_AUTHORIZATION_v1_run1_2026-11-06.json", AUTH_V1), ("G4_AUTHORIZATION_v2_runs2-5_2026-11-09.json", AUTH_V2)):
        need(os.path.isfile(os.path.join(work, n)) and sha256_file(os.path.join(work, n)) == h, f"preserved {n} missing or changed")
    need(sys.version_info[:3] == (3, 13, 3), f"Python {sys.version.split()[0]} != pinned 3.13.3")
    pins = {}
    for line in open(lock, encoding="utf-8"):
        if "==" in line and not line.startswith("#"):
            n, v = line.split("==", 1)
            pins[n.strip().lower()] = v.split()[0].strip(" \\")
    for pkg in ("numpy", "pyarrow", "scikit-learn"):
        need(md.version(pkg) == pins.get(pkg), f"{pkg} {md.version(pkg)} != locked {pins.get(pkg)}")


def import_frozen(work):
    src = os.path.join(work, "src")
    sys.path.insert(0, src)
    import g4_common as K
    import g4_seal as S
    import metrics_g5 as MG
    return K, S, MG


# ------------------------------------------------------------------------------------------ evidence (hash/metadata only)
def read_jsonl_chain(path):
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]
    prev, ev = "0" * 64, []
    for l in lines:
        e = json.loads(l)
        need(e.get("prev") == prev, f"ledger hash chain broken: {path}")
        prev = hashlib.sha256(l.encode()).hexdigest()
        ev.append(e)
    head = open(path + ".head", encoding="utf-8").read().strip()
    need(head == prev, f"ledger head mismatch: {path}")
    return ev


def check_evidence(a, runs_root, K, S):
    """The 20 slots. Opens ledgers/manifests/job results (JSON) and hashes score tables as raw bytes. Never decodes a table."""
    import numpy as np
    dirs = [d for d in os.listdir(runs_root) if "_G4_" in d]
    need(len(dirs) == 20 and all(d.endswith("_a1") for d in dirs), f"expected exactly 20 attempt-1 directories, found {dirs}")
    out, ident = {}, set()
    for slot in SLOTS:
        m, r = slot.rsplit("_r", 1)
        r = int(r)
        pin = a["evidence"][slot]
        found = [d for d in dirs if d.endswith(f"_G4_{m}_r{r}_a1")]
        need(len(found) == 1, f"{slot}: expected one directory, found {found}")
        d = os.path.join(runs_root, found[0])
        ev = read_jsonl_chain(os.path.join(runs_root, "ledger", f"{slot}.jsonl"))
        need([e["event"] for e in ev] == ["registered", "launched", "finished"], f"{slot}: ledger events {[e['event'] for e in ev]}")
        need(all(e.get("model") == m and e.get("run") == r and e.get("attempt") == 1 for e in ev), f"{slot}: ledger identity")
        fin = ev[-1]
        need(fin["state"] == "SUCCEEDED", f"{slot}: ledger state {fin['state']}")
        need(os.path.normcase(os.path.abspath(ev[0]["out_dir"])) == os.path.normcase(os.path.abspath(d)), f"{slot}: ledger out_dir")
        mp = os.path.join(d, "run_manifest.json")
        need(sha256_file(mp) == fin["manifest_sha256"] == pin["manifest_sha256"], f"{slot}: manifest hash differs from ledger/pin")
        M = json.load(open(mp, encoding="utf-8"))
        need((M["model"], M["run"], M["attempt"], M["seed"], M["state"], M["freeze"], M["mode"]) == (m, r, 1, r, "SUCCEEDED", "v6", "real"), f"{slot}: manifest identity/seed/state")
        need(M["metrics_computed_on_evaluation_labels"] is False and M["resource_validation"]["violations"] == [] and
             M["child_exit_code"] == 0 and "parent_failures" not in M, f"{slot}: manifest flags")
        need(M["authorization"]["sha256"] == (AUTH_V1 if r == 1 else AUTH_V2), f"{slot}: authorization identity")
        gate = GATE_FILES[1 if r == 1 else 2]
        src = M["source_sha256"]
        need(all(src[k] == v for k, v in gate.items()), f"{slot}: gate-file hashes differ from the recorded V6/V6.1 set")
        need(all(src[k] == a["source_sha256"][k] for k in ("metrics_g5.py", "g4_seal.py", "g4_common.py", "g4_data.py")), f"{slot}: metric/seal/data source hashes differ from authorization")
        need(M["lock_sha256"] == a["env_lock_sha256"], f"{slot}: lock hash")
        inp = M["inputs"]
        need(inp["eval_rows"] == K.EVAL_ROWS and inp["membership_digest"] == K.MEMBERSHIP_DIGEST and inp["keys_table_sha256"] == K.KEYS_TABLE_SHA256, f"{slot}: input identity differs from frozen G2/G3 values")
        ident.add(json.dumps({k: inp[k] for k in ("file_sha256", "membership_digest", "keys_table_sha256", "eval_rows")}, sort_keys=True))
        jr = os.path.join(d, "job_result.json")
        need(sha256_file(jr) == open(os.path.join(d, "job_result.sha256")).read().strip(), f"{slot}: job_result sidecar")
        J = json.load(open(jr, encoding="utf-8"))
        sc = J["scores"]
        need(J["state"] == "SUCCEEDED" and J["exit_code"] == 0 and J["last_phase"] == "done", f"{slot}: job_result state")
        need(sc["seal_sha256"] == sc["pass2_sha256"] == pin["seal_sha256"] and sc["file_sha256"] == pin["scores_file_sha256"], f"{slot}: seal/file hash differs from pin")
        need(sc["rows"] == K.EVAL_ROWS and sc["dtype"] == np.dtype(S.SCORE_DTYPES[m]).name, f"{slot}: score rows/dtype differ from the frozen contract")
        sp = os.path.join(d, "scores.parquet")
        need(sha256_file(sp) == pin["scores_file_sha256"], f"{slot}: scores.parquet bytes differ from pin")
        need(open(os.path.join(d, "scores.sha256")).read().strip() == pin["seal_sha256"], f"{slot}: scores.sha256 sidecar")
        listed = {}
        for line in open(os.path.join(d, "SHA256SUMS.txt"), encoding="utf-8"):
            if line.strip():
                h, n = line.split(None, 1)
                listed[n.strip().lstrip("*")] = h
        present = {f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f != "SHA256SUMS.txt"}
        need(set(listed) == present and all(sha256_file(os.path.join(d, n)) == h for n, h in listed.items()), f"{slot}: SHA256SUMS coverage/hash")
        out[slot] = {"dir": d, "scores_path": sp, "seal": pin["seal_sha256"], "model_config": M["model_config"], "deviations": M["deviations"],
                     "authorization": M["authorization"]["sha256"], "state": "SUCCEEDED"}
    need(len(ident) == 1, "evaluation-universe identity differs across the 20 runs")
    return out


# ------------------------------------------------------------------------------------------ metadata (labels: execute mode only)
def build_metadata(spec, K, MG, grant=None):
    """Frozen G4 Dataset constructor (cohorts, unit/label assignment, digest and size checks, keys-table hash) plus the
    sealed keys table (kid, all_zero, seen). Reads ONLY label/year_month columns (and hashes the 24 files)."""
    import numpy as np
    import g4_data as D
    if grant is not None:
        import g4_access
        g4_access._GRANT = dict(grant)                 # explicit, in-process G5 grant, set only after authorization checks
    ds = D.Dataset(spec)                               # raises NotAuthorized for real data without the grant
    n = len(ds.labels)
    kid, allz, seen = np.empty(n, np.int64), np.empty(n, np.int8), np.full(n, -1, np.int8)
    with open(spec.keys_csv, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    need(len(rows) == n, "keys table row count differs from the cohort table")
    for i, row in enumerate(rows):
        need(int(row["gid"]) == i, "keys table gids are not 0..N-1")
        kid[i], allz[i] = int(row["kid"]), int(row["all_zero"])
        if row["seen"] == "TRAIN":
            need(bool(ds.is_train[i]), f"keys table marks gid {i} TRAIN but cohorts do not")
        else:
            need(not ds.is_train[i] and row["seen"] in ("0", "1"), f"keys table seen flag invalid for gid {i}")
            seen[i] = int(row["seen"])
    eg = ds.eval_gids
    meta = MG.EvalMeta(eg, ds.units[eg], ds.labels[eg], seen[eg])
    return ds, meta, kid, allz, seen


def model_independent(ds, meta, kid, allz, seen, MG):
    out = {}
    for u in MG.ALL_UNITS:
        F = meta.units == u
        U = F & (meta.seen == 0)
        rows = {}
        for arm, mask in (("F", F), ("U", U)):
            lab = meta.labels[mask]
            n1, n = int(lab.sum()), int(mask.sum())
            g = meta.gids[mask]
            rows[arm] = {"n": n, "malware": n1, "benign": n - n1, "prevalence": Fraction(n1, n) if n else None,
                         "distinct_keys": int(len(set(kid[g].tolist()))), "all_zero_rows": int(allz[g].sum()),
                         # Manifest 7.3, per unit x arm (F and U): rows with the 'unknown' month sentinel, and keys carrying
                         # both labels inside the arm (frozen E3 definition, MG.e3_scope, applied to the arm's rows)
                         "unknown_month_rows": int(sum(1 for x in g.tolist() if ds.yms[x] == "unknown")),
                         "conflict_keys": int(MG.e3_scope(kid[g], lab)["conflict_keys"])}
        g = meta.gids[F]
        sv = seen[g]
        lab = meta.labels[F]
        rows["F"]["seen_rows"] = {"benign": int(((sv == 1) & (lab == 0)).sum()), "malware": int(((sv == 1) & (lab == 1)).sum())}
        rows["F"]["seen_rate"] = Fraction(int((sv == 1).sum()), len(g)) if len(g) else None
        out[u] = rows
    return out


def e3_all(ds, kid, allz, MG):
    """Manifest 7.5 on the declared scopes: TRAIN, each of the 12 units, TRAIN u unit. Family: never authorized/read."""
    import numpy as np
    lab, units = ds.labels, ds.units
    tr = np.flatnonzero(ds.is_train)
    scopes = {"TRAIN": tr}
    for u in MG.ALL_UNITS:
        iu = np.flatnonzero(units == u)
        scopes[u] = iu
        scopes[f"TRAIN+{u}"] = np.concatenate([tr, iu])
    out = {}
    for name, idx in scopes.items():
        r = MG.e3_scope(kid[idx], lab[idx])
        z = allz[idx] == 1
        r["all_zero_key_pn"] = {"p": int(lab[idx][z].sum()), "n": int(z.sum() - lab[idx][z].sum())}
        out[name] = r
    far = {}
    for tag, pre in (("unit_scope", ""), ("TRAIN_plus_unit_scope", "TRAIN+")):
        v = [out[pre + u]["AUC_max"] for u in MG.FAR_UNITS]
        far[tag] = None if any(x is None for x in v) else MG._mean(v)
    out["_FAR_equal_year_average_of_AUC_max"] = {
        "label": "average of per-scope bounds; not necessarily jointly attainable by a single scoring rule (F-03)", **far}
    return out


# ------------------------------------------------------------------------------------------ evaluation
def evaluate_runs(meta, load_table, states, MG):
    """load_table(model, run) -> (gids, scores, expected_digest) already hash-verified by the caller. Exact arithmetic."""
    res = {m: {} for m in MODELS}
    for m in MODELS:
        for r in RUNS:
            g, s, dg = load_table(m, r)
            res[m][r] = MG.run_metrics(g, s, dg, meta, sklearn_check=True)
    summ = {m: MG.model_summary(res[m], {r: states[f"{m}_r{r}"] for r in RUNS}) for m in MODELS}
    return res, summ


def rgn_table(res, MG):
    out = {}
    for m in MODELS:
        out[m] = {}
        for split, units in (("IID", MG.IID_UNITS), ("NEAR", MG.NEAR_UNITS), ("FAR", MG.FAR_UNITS)):
            per = {}
            for r in RUNS:
                v = [res[m][r]["units"][u]["A_F"] for u in units]
                per[r] = None if any(x is None for x in v) else MG._mean(v) * 100
            mean = MG._mean(list(per.values())) if all(x is not None for x in per.values()) else None
            ref, sd = RGN_REF[m][split]
            out[m][split] = {"per_run_x100": per, "mean_x100": mean, "paper_mean": ref, "paper_sd_not_a_tolerance": sd,
                             "signed_difference_x100": None if mean is None else mean - Fraction(ref)}
    return out


def ser(o):
    import numpy as np
    if isinstance(o, Fraction):
        return {"value": float(o), "exact": f"{o.numerator}/{o.denominator}"}
    if isinstance(o, dict):
        return {str(k): ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [ser(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_outputs(out_dir, a, evid, res, summ, rgn, mi, e3, K):
    parent = os.path.dirname(os.path.abspath(out_dir))
    part = os.path.abspath(out_dir) + ".partial"
    os.makedirs(part)
    files = {}

    def put(name, obj):
        txt = json.dumps(ser(obj), indent=1, sort_keys=True) + "\n"
        need(not BANNED.search(txt), f"prohibited wording in {name}")
        with open(os.path.join(part, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
        files[name] = hashlib.sha256(txt.encode()).hexdigest()
    put("G5_per_run_metrics.json", {m: {str(r): {k: v for k, v in res[m][r].items()} for r in RUNS} for m in MODELS})
    put("G5_model_summary.json", summ)
    put("G5_model_independent_counts.json", mi)
    put("G5_E3.json", e3)
    put("RG-N.json", {"reference": "arXiv v1 Table 1 / ICLR 2026 Table 2, LAMDA ROC-AUC", "annotations": RGN_ANNOTATIONS, "table": rgn})
    put("RG-S.json", {
        "status": "ANNOTATED_FROZEN_SETTINGS",
        "paper_specified_vs_active_code_vs_C1plus": "Manifest v0.2.4 section 5.5 (sha256 ac8fee5f014778533d12359d2052f985277d6a614f181c6c1fdad9e163721898); not re-transcribed by this driver",
        "frozen_model_configs_as_recorded_in_manifests": {m: evid[f"{m}_r1"]["model_config"] for m in MODELS},
        "deviations_recorded_in_manifests": {m: sorted({d for r in RUNS for d in evid[f"{m}_r{r}"]["deviations"]}) for m in MODELS},
        "authorization_per_slot": {s: evid[s]["authorization"] for s in SLOTS},
        "deviation_decision_record_sha256": a["deviation_record_sha256"], "mandatory_annotations": RGN_ANNOTATIONS})
    put("RG-T.json", {"status": "PASS", "checks": [
        "authorization record, driver, frozen sources, lock and python/package pins verified before evaluation",
        "20 slots: ledger chain/head, terminal SUCCEEDED, attempt 1, identities, manifest/job_result/seal/file hashes equal to pins",
        "cohorts: frozen G2/G3 membership digest and set sizes verified by the frozen Dataset constructor",
        "keys table: frozen sha256 verified; gids, TRAIN flags and seen flags consistent with the cohorts",
        "stable prediction joins: each sealed table digest re-verified; gids equal the evaluation universe; Unseen == Full restricted to mask",
        "score dtype and row count equal the frozen contract; sklearn cross-check within 1e-12",
        "any failure above stops evaluation before outputs are written"],
        "evaluation_universe_rows": K.EVAL_ROWS, "sklearn_max_abs_deviation": max(res[m][r]["sklearn_max_abs_deviation"] for m in MODELS for r in RUNS)})
    sums = "".join(f"{h}  {n}\n" for n, h in sorted(files.items()))
    with open(os.path.join(part, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(sums)
    os.rename(part, os.path.abspath(out_dir))
    return files


# ------------------------------------------------------------------------------------------ main
def main(argv=None, work=WORK):
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", default=os.path.join(work, "G5_AUTHORIZATION.json"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify-only", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir")
    a_ = ap.parse_args(argv)
    runs_root = os.path.join(work, "runs")
    try:
        a = load_auth(a_.auth)
        check_driver_sources_env(a, work, os.path.abspath(__file__))
        K, S, MG = import_frozen(work)
        if a_.execute:
            need(a["authorized"] is True, "authorization record is not authorized (authorized != true): evaluation refused")
            need(a_.out_dir, "--out-dir is required with --execute")
            od = os.path.abspath(a_.out_dir)
            need(not os.path.exists(od) and not os.path.exists(od + ".partial"), "output directory already exists")
            need(os.path.isdir(os.path.dirname(od)), "output parent directory missing")
            need(not (od + os.sep).lower().startswith(os.path.abspath(runs_root).lower() + os.sep), "output must not be inside runs/")
        evid = check_evidence(a, runs_root, K, S)
        if a_.verify_only:
            print(json.dumps({"mode": "verify-only", "result": "PASS", "slots": len(evid), "authorized_flag": a["authorized"],
                              "scores_decoded": False, "labels_read": False}, indent=1))
            return 0
        from g4_data import DataSpec
        spec = DataSpec.real()
        ds, meta, kid, allz, seen = build_metadata(spec, K, MG, grant=spec.to_dict())
        need(len(meta.gids) == K.EVAL_ROWS, "evaluation universe size differs from the frozen value")

        def load_table(m, r):
            e = evid[f"{m}_r{r}"]
            gg, ss = S.read_table(e["scores_path"], e["seal"])                 # digest re-verified (P4)
            need(ss.dtype == __import__("numpy").dtype(S.SCORE_DTYPES[m]), f"{m} r{r}: score dtype")
            return gg, ss, e["seal"]
        res, summ = evaluate_runs(meta, load_table, {s: evid[s]["state"] for s in SLOTS}, MG)
        rgn = rgn_table(res, MG)
        files = write_outputs(od, a, evid, res, summ, rgn, model_independent(ds, meta, kid, allz, seen, MG), e3_all(ds, kid, allz, MG), K)
        print(json.dumps({"mode": "execute", "result": "DONE", "out_dir": od, "files": files}, indent=1))
        return 0
    except G5Stop as e:
        print(f"G5 STOP: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
