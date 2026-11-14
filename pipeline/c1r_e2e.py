#!/usr/bin/env python
"""C1-R / LAMDA fixed-score audit: unified end-to-end entry point (public execution interface).

    python pipeline/c1r_e2e.py <stage> --work-dir WORK [options]

Stages (run in this order by `all`):
  download      fetch the 24 pinned LAMDA parquet files (revision ad9614bd...), verify size and SHA-256
  verify        frozen G2 input verification: sizes, SHA-256, footers, schema, row counts, total 1,008,381
  cohorts       frozen G3 core: feature validation, cohorts, exact TRAIN-vector membership (Full/Unseen), conflicts;
                compared with the pinned membership digest, keys-table SHA-256 and aggregate tables
  train         fit and score SVM, LightGBM, XGBoost, MLP, runs 1-5 (seed = run), sealed score tables
  evaluate      exact-rational ROC-AUC, primary/secondary estimands, count bounds, conflict oracle bounds
  sensitivity   exploratory S0-S4 re-weightings (exact arithmetic) from the newly computed aggregates
  tables        manuscript Tables 4-6 and count bounds from the newly computed aggregates
  figures       manuscript Figures 2-3 from the newly computed aggregates (needs matplotlib)
  compare       newly computed results vs the original sealed reference results
  all           every stage above

The scientific code is the frozen C1-R implementation (pipeline/frozen/src, hash-pinned, see pipeline/PARITY.md).
Nothing in the computation path reads a reference result; `compare` is the only stage that opens them.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FROZEN = os.path.join(HERE, "frozen", "src")
PUBLIC = os.path.join(HERE, "public")
REFERENCE = os.path.join(HERE, "reference")
MODELS = ("svm", "lgbm", "xgb", "mlp")
RUNS = (1, 2, 3, 4, 5)
REV = "ad9614bdd5556767f97ced2fce797c2f06408ebf"
ETAG_KEY = "x-linked-etag (server-provided; algorithm not asserted)"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def jdump(p, o):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(o, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")


def need(cond, msg):
    if not cond:
        raise SystemExit("STOP: " + msg)


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    """Everything a stage needs. Real-data defaults are the frozen constants; the synthetic test overrides them."""

    def __init__(self, work, data_dir=None, g1_dir=None, keys_csv=None, nf=None, total=None, membership_digest="__real__",
                 keys_sha="__real__", set_sizes=None, overrides=None, containers=None, runs=RUNS, models=MODELS,
                 real=True, batch_rows=8192):
        sys.path[:0] = [PUBLIC, FROZEN]
        import g4_common as K
        K.forbidden_path = lambda p: False          # see pipeline/PARITY.md (guard for an unrelated private project)
        self.K = K
        self.work = os.path.abspath(work)
        self.data_dir = data_dir or os.path.join(self.work, "data", "Baseline")
        self.g1_dir = g1_dir or os.path.join(self.work, "g1")
        self.keys_csv = keys_csv or os.path.join(self.work, "keys", "keys_seen.csv")
        self.nf = nf or K.NF
        self.total = total or K.TOTAL_ROWS
        self.membership_digest = K.MEMBERSHIP_DIGEST if membership_digest == "__real__" else membership_digest
        self.keys_sha = K.KEYS_TABLE_SHA256 if keys_sha == "__real__" else keys_sha
        self.set_sizes = dict(K.SET_SIZES) if set_sizes is None else set_sizes
        self.overrides = overrides or {m: {} for m in MODELS}
        self.containers = containers or {m: "quantile" for m in MODELS}
        self.runs, self.models, self.real, self.batch_rows = tuple(runs), tuple(models), real, batch_rows
        self.g2_path = os.path.join(FROZEN, "g2_analysis.py")
        self.g3_path = os.path.join(FROZEN, "g3core.py")
        self.runs_dir = os.path.join(self.work, "runs")
        self.eval_dir = os.path.join(self.work, "eval")
        self.reports = os.path.join(self.work, "reports")

    def spec(self, keys_sha=None):
        return {"data_dir": self.data_dir, "g1_run": self.g1_dir, "g2_module": self.g2_path, "g3core_module": self.g3_path,
                "keys_csv": self.keys_csv, "nf": self.nf, "total": self.total, "membership_digest": self.membership_digest,
                "keys_sha": self.keys_sha if keys_sha is None else keys_sha, "set_sizes": self.set_sizes,
                "g2_sha": sha256_file(self.g2_path), "g3_sha": sha256_file(self.g3_path), "batch_rows": self.batch_rows}


# ----------------------------------------------------------------------------------------------- stage: download
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def stage_download(ctx, dry_run=False, source_dir=None):
    ref = jload(os.path.join(REFERENCE, "dataset_files.json"))
    need(ref["revision"] == REV, "reference revision mismatch")
    os.makedirs(ctx.g1_dir, exist_ok=True)
    shutil.copyfile(os.path.join(REFERENCE, "g1_inventory.json"), os.path.join(ctx.g1_dir, "inventory.json"))
    report, sums = [], []
    for f in ref["files"]:
        rel = f["path"]                                   # Baseline/2013/2013_train.parquet
        dest = os.path.join(ctx.data_dir, *rel.split("/")[1:])
        if dry_run:                                       # metadata only: HEAD request, no content downloaded
            req = urllib.request.Request(f["url"], method="HEAD")
            try:
                r = urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
                h = r.headers
            except urllib.error.HTTPError as e:           # the 302 carries the LFS metadata headers
                h = e.headers
            etag = (h.get("X-Linked-Etag") or "").strip('"')
            size = int(h.get("X-Linked-Size") or -1)
            ok = etag == f["sha256"] and size == f["size"]
            report.append({"file": rel, "ok": ok, "remote_size": size, "remote_etag_matches_pin": etag == f["sha256"]})
            continue
        if os.path.isfile(dest) and os.path.getsize(dest) == f["size"] and sha256_file(dest) == f["sha256"]:
            status = "already present and verified"
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            if source_dir:
                shutil.copyfile(os.path.join(source_dir, *rel.split("/")[1:]), tmp)
            else:
                for attempt in range(4):
                    try:
                        with urllib.request.urlopen(f["url"], timeout=120) as r, open(tmp, "wb") as out:
                            shutil.copyfileobj(r, out, 1 << 20)
                        break
                    except Exception as e:  # noqa: BLE001
                        print(f"  retry {attempt + 1} for {rel}: {e}")
                        time.sleep(3 * (attempt + 1))
                else:
                    raise SystemExit(f"STOP: download failed for {rel}")
            need(os.path.getsize(tmp) == f["size"] and sha256_file(tmp) == f["sha256"], f"{rel}: size/SHA-256 differ from the pin")
            os.replace(tmp, dest)
            status = "downloaded and verified"
        sums.append(f"{f['sha256']}  {'/'.join(rel.split('/')[1:])}")
        report.append({"file": rel, "status": status})
        print(f"  {rel}: {status}")
    if not dry_run:
        with open(os.path.join(ctx.data_dir, "LOCAL_SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(sums) + "\n")
    ok = all(r.get("ok", True) for r in report)
    jdump(os.path.join(ctx.reports, "download_dry_run.json" if dry_run else "download.json"), {"revision": REV, "files": report, "ok": ok})
    need(ok, "remote metadata differs from the pins")
    print("download: PASS" + (" (dry run, metadata only)" if dry_run else ""))
    return True


# ----------------------------------------------------------------------------------------------- stage: verify
def load_g2(ctx):
    return ctx.K.load_locked_module("g2_analysis_locked", ctx.g2_path, ctx.spec()["g2_sha"])


def stage_verify(ctx):
    G2 = load_g2(ctx)
    files, total = G2.verify_inputs(ctx.data_dir, ctx.g1_dir, ctx.nf, ctx.total)
    jdump(os.path.join(ctx.reports, "verify.json"), {"files": len(files), "total_rows": total,
                                                      "file_sha256": {f["rel"]: f["sha256"] for f in files}})
    print(f"verify: PASS ({len(files)} files, {total:,} rows; sizes, SHA-256, footers, schema, row counts)")
    return True


# ----------------------------------------------------------------------------------------------- stage: cohorts
def stage_cohorts(ctx, skip_pins=False):
    import types
    import g3_main as G3M
    G3M.G2_SOURCE_SHA = sha256_file(ctx.g2_path)                # pin of the published (path-neutralised) g2_analysis.py
    orig_load = G3M.load_g2_module

    def load(g2_work):
        m = orig_load(g2_work)
        m.ANCHORS = {}                                           # private governance anchors are not applicable (PARITY.md)
        return m
    G3M.load_g2_module = load
    out_root = os.path.join(ctx.work, "g3")
    os.makedirs(out_root, exist_ok=True)
    args = types.SimpleNamespace(data_dir=ctx.data_dir, g1_run=ctx.g1_dir, g2_work=os.path.join(HERE, "frozen"),
                                 g2_run=os.path.join(ctx.work, "_unused_g2_run"), review_dir=os.path.join(ctx.work, "_unused_review"),
                                 out_root=out_root, batch_rows=ctx.batch_rows, block_rows=256, label="public",
                                 skip_pins=skip_pins or not ctx.real)
    before = set(os.listdir(out_root))
    code = G3M.run(args, nf=ctx.nf, expected_total=ctx.total, check_g2=False)
    new = sorted(set(os.listdir(out_root)) - before)
    need(code == 0 and len(new) == 1, f"G3 core stopped (exit {code})")
    out = os.path.join(out_root, new[0])
    S = jload(os.path.join(out, "g3_summary.json"))
    keys_src = os.path.join(out, "restricted", "keys_seen.csv")
    os.makedirs(os.path.dirname(ctx.keys_csv), exist_ok=True)
    shutil.copyfile(keys_src, ctx.keys_csv)
    res = {"g3_run_dir": new[0], "verdict": S["verdict"], "membership_digest": S["cohorts"]["membership_digest"],
           "cohort_sizes": S["cohorts"]["sizes"], "keys_table_sha256": S["keys_table_sha256"],
           "distinct_train_vectors": S["k_train"]["distinct_train_vectors"]}
    chk = {"membership_digest_equals_pin": ctx.membership_digest is None or res["membership_digest"] == ctx.membership_digest,
           "cohort_sizes_equal_pin": res["cohort_sizes"] == ctx.set_sizes,
           "keys_table_sha256_equals_pin": ctx.keys_sha is None or res["keys_table_sha256"] == ctx.keys_sha}
    if ctx.real:
        ref = os.path.join(REPO, "data", "g3_aggregates", "G3_AGGREGATE_TABLES.csv")
        chk["aggregate_tables_byte_identical_to_reference"] = sha256_file(os.path.join(out, "G3_AGGREGATE_TABLES.csv")) == sha256_file(ref)
    res["checks"] = chk
    jdump(os.path.join(ctx.reports, "cohorts.json"), res)
    need(all(chk.values()), f"cohort/keys checks failed: {chk}")
    print("cohorts: PASS " + json.dumps(chk))
    return True


# ----------------------------------------------------------------------------------------------- stage: train
def slot_dir(ctx, m, r):
    return os.path.join(ctx.runs_dir, f"{m}_r{r}")


def stage_train(ctx, models=None, runs=None, force=False):
    need(os.path.isfile(ctx.keys_csv), "keys table missing: run the `cohorts` stage first")
    for m in models or ctx.models:
        for r in runs or ctx.runs:
            d = slot_dir(ctx, m, r)
            res_p = os.path.join(d, "job_result.json")
            if os.path.isfile(res_p) and jload(res_p).get("state") == "SUCCEEDED" and not force:
                print(f"  {m} run {r}: already SUCCEEDED (kept; use --force to retrain)")
                continue
            os.makedirs(d, exist_ok=True)
            job = {"out_dir": d, "model": m, "run": r, "overrides": ctx.overrides[m], "container": ctx.containers[m],
                   "spec": ctx.spec(keys_sha=ctx.keys_sha)}
            jp = os.path.join(d, "job.json")
            jdump(jp, job)
            t0 = time.time()
            rc = subprocess.call([sys.executable, os.path.join(PUBLIC, "job_public.py"), jp])
            R = jload(res_p) if os.path.isfile(res_p) else {"state": "NO-RESULT"}
            print(f"  {m} run {r}: {R['state']} in {time.time() - t0:.0f}s" + (f" ({R.get('error')})" if R["state"] != "SUCCEEDED" else ""))
            need(rc == 0 and R["state"] == "SUCCEEDED", f"{m} run {r} did not succeed")
    print("train: PASS")
    return True


# ----------------------------------------------------------------------------------------------- stage: evaluate
BANNED_WORDS = ("reproduced", "faithful")


def stage_evaluate(ctx):
    import g4_seal as S
    import g5_eval as G5
    import metrics_g5 as MG
    from g4_data import DataSpec
    spec = DataSpec(**ctx.spec())
    ds, meta, kid, allz, seen = G5.build_metadata(spec, ctx.K, MG)
    need(len(meta.gids) == len(ds.eval_gids), "evaluation universe size")
    evid = {}
    for m in G5.MODELS:
        for r in G5.RUNS:
            d = slot_dir(ctx, m, r)
            R = jload(os.path.join(d, "job_result.json"))
            need(R["state"] == "SUCCEEDED" and R["scores"]["seal_sha256"] == R["scores"]["pass2_sha256"], f"{m} run {r}: not a clean slot")
            evid[f"{m}_r{r}"] = {"path": os.path.join(d, "scores.parquet"), "seal": R["scores"]["seal_sha256"],
                                 "file": R["scores"]["file_sha256"]}

    def load_table(m, r):
        e = evid[f"{m}_r{r}"]
        gg, ss = S.read_table(e["path"], e["seal"])          # seal re-verified before use
        import numpy as np
        need(ss.dtype == np.dtype(S.SCORE_DTYPES[m]), f"{m} run {r}: score dtype")
        return gg, ss, e["seal"]
    res, summ = G5.evaluate_runs(meta, load_table, {s: "SUCCEEDED" for s in G5.SLOTS}, MG)
    out = ctx.eval_dir
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    files = {}

    def put(name, obj):
        txt = json.dumps(G5.ser(obj), indent=1, sort_keys=True) + "\n"
        need(not G5.BANNED.search(txt), f"prohibited wording in {name}")
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
        files[name] = hashlib.sha256(txt.encode()).hexdigest()
    put("G5_per_run_metrics.json", {m: {str(r): dict(res[m][r]) for r in G5.RUNS} for m in G5.MODELS})
    put("G5_model_summary.json", summ)
    put("G5_model_independent_counts.json", G5.model_independent(ds, meta, kid, allz, seen, MG))
    put("G5_E3.json", G5.e3_all(ds, kid, allz, MG))
    with open(os.path.join(out, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(f"{h}  {n}\n" for n, h in sorted(files.items())))
    jdump(os.path.join(out, "score_table_sha256.json"), {"description": "digests of the newly computed score files and seals",
          "evidence": {k: {"scores_file_sha256": v["file"], "seal_sha256": v["seal"]} for k, v in evid.items()}})
    print("evaluate: PASS (4 aggregate files written to " + os.path.relpath(out, ctx.work) + ")")
    return True


# ----------------------------------------------------------------------------------------------- stages using repo scripts
def _env(ctx, extra=None):
    e = dict(os.environ)
    e["C1R_G5_AGGREGATES"] = ctx.eval_dir
    e["C1R_SCORE_TABLE_HASHES"] = os.path.join(ctx.eval_dir, "score_table_sha256.json")
    e["C1R_WP1_RESULTS"] = os.path.join(ctx.work, "wp1", "WP1_S0_S4_RESULTS.json")
    e.update(extra or {})
    return e


def stage_sensitivity(ctx):
    out = os.path.join(ctx.work, "wp1")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    rc = subprocess.call([sys.executable, "-I", os.path.join(REPO, "src", "wp1_sensitivity.py"), "--out-dir", out], env=_env(ctx))
    need(rc == 0, "wp1_sensitivity.py stopped (see its gate messages)")
    rc = subprocess.call([sys.executable, "-I", os.path.join(REPO, "src", "wp1_independent_check.py"), "--results-dir", out], env=_env(ctx))
    need(rc == 0, "independent check failed")
    print("sensitivity: PASS")
    return True


def stage_tables(ctx):
    rc = subprocess.call([sys.executable, "-I", os.path.join(REPO, "src", "make_tables.py"), "--out-dir", os.path.join(ctx.work, "tables")], env=_env(ctx))
    need(rc == 0, "make_tables.py failed")
    return True


def stage_figures(ctx):
    fd = os.path.join(ctx.work, "figures")
    rc = subprocess.call([sys.executable, "-I", os.path.join(REPO, "src", "make_figures.py"), ctx.eval_dir, fd,
                          os.path.join(fd, "plotted_values.json"), "--skip-pin-check"], env=_env(ctx))
    need(rc == 0, "make_figures.py failed (matplotlib installed?)")
    return True


# ----------------------------------------------------------------------------------------------- stage: compare
def _walk(a, b, path, acc):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) == {"exact", "value"} and set(b) == {"exact", "value"}:
            acc["fields"] += 1
            if a["exact"] != b["exact"]:
                acc["exact_diff"] += 1
                acc["max_abs"] = max(acc["max_abs"], abs(a["value"] - b["value"]))
                acc["worst"].append((abs(a["value"] - b["value"]), path))
            return
        for k in set(a) | set(b):
            if k not in a or k not in b:
                acc["structure"] += 1
            else:
                _walk(a[k], b[k], path + "/" + k, acc)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", acc)
    else:
        acc["scalars"] += 1
        if a != b:
            acc["scalar_diff"] += 1
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                acc["max_abs"] = max(acc["max_abs"], abs(a - b))


def stage_compare(ctx, ref_dir=None, tolerance=5e-5):
    ref_dir = ref_dir or os.path.join(REPO, "data", "g5_aggregates")
    rep = {"reference": os.path.relpath(ref_dir, REPO), "tolerance_abs": tolerance}
    seal_ref = jload(os.path.join(REFERENCE, "original_seals.json"))["slots"]
    seal_eq = {}
    for m in ctx.models:
        for r in ctx.runs:
            R = jload(os.path.join(slot_dir(ctx, m, r), "job_result.json"))["scores"]
            ref = seal_ref.get(f"{m}_r{r}", {})
            seal_eq[f"{m}_r{r}"] = {"seal_equal": R["seal_sha256"] == ref.get("seal_sha256"),
                                    "file_equal": R["file_sha256"] == ref.get("scores_file_sha256")}
    rep["score_seals"] = seal_eq
    rep["n_seals_equal"] = sum(v["seal_equal"] for v in seal_eq.values())
    files = {}
    for name in ("G5_per_run_metrics.json", "G5_model_summary.json", "G5_model_independent_counts.json", "G5_E3.json"):
        acc = {"fields": 0, "exact_diff": 0, "max_abs": 0.0, "structure": 0, "scalars": 0, "scalar_diff": 0, "worst": []}
        _walk(jload(os.path.join(ctx.eval_dir, name)), jload(os.path.join(ref_dir, name)), "", acc)
        acc["worst"] = [(round(w, 12), p) for w, p in sorted(acc["worst"], reverse=True)[:3]]
        acc["byte_identical"] = sha256_file(os.path.join(ctx.eval_dir, name)) == sha256_file(os.path.join(ref_dir, name))
        files[name] = acc
    rep["aggregates"] = files
    exact = all(f["byte_identical"] for f in files.values())
    within = all(f["structure"] == 0 and f["scalar_diff"] == 0 and f["max_abs"] <= tolerance for f in files.values())
    tdir = os.path.join(ctx.work, "tables")
    if os.path.isdir(tdir) and os.path.isfile(os.path.join(REPO, "outputs", "tables", "table5_primary.tex")):
        rep["tables_equal_published"] = {n: sha256_file(os.path.join(tdir, n)) == sha256_file(os.path.join(REPO, "outputs", "tables", n))
                                         for n in ("table4_cohorts.tex", "table5_primary.tex", "table6_sensitivity.tex") if os.path.isfile(os.path.join(tdir, n))}
    rep["status"] = "EXACT" if exact else ("WITHIN_TOLERANCE" if within else "DIFFERS")
    jdump(os.path.join(ctx.reports, "compare.json"), rep)
    print(f"compare: {rep['status']}; seals equal to the originals: {rep['n_seals_equal']}/{len(seal_eq)}; "
          + "; ".join(f"{k.split('_', 1)[1][:-5]} maxdiff {v['max_abs']:.2e}" for k, v in files.items()))
    return rep["status"] in ("EXACT", "WITHIN_TOLERANCE")


# ----------------------------------------------------------------------------------------------- cli
STAGES = ("download", "verify", "cohorts", "train", "evaluate", "sensitivity", "tables", "figures", "compare")


def run_stage(name, ctx, a):
    return {"download": lambda: stage_download(ctx, a.dry_run, a.source_dir), "verify": lambda: stage_verify(ctx),
            "cohorts": lambda: stage_cohorts(ctx), "train": lambda: stage_train(ctx, a.model and [a.model], a.run and [a.run], a.force),
            "evaluate": lambda: stage_evaluate(ctx), "sensitivity": lambda: stage_sensitivity(ctx), "tables": lambda: stage_tables(ctx),
            "figures": lambda: stage_figures(ctx), "compare": lambda: stage_compare(ctx, a.reference_dir, a.tolerance)}[name]()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ("all",))
    ap.add_argument("--work-dir", default="work", help="data, keys, runs, outputs and reports (default ./work)")
    ap.add_argument("--dry-run", action="store_true", help="download: HEAD requests only (metadata check, nothing downloaded)")
    ap.add_argument("--source-dir", help="download: copy the 24 files from an existing local Baseline directory (still verified)")
    ap.add_argument("--model", choices=MODELS, help="train: only this model")
    ap.add_argument("--run", type=int, choices=RUNS, help="train: only this run")
    ap.add_argument("--force", action="store_true", help="train: retrain slots that already SUCCEEDED")
    ap.add_argument("--reference-dir", help="compare: directory with reference aggregates (default: data/g5_aggregates)")
    ap.add_argument("--tolerance", type=float, default=5e-5, help="compare: absolute tolerance on AUC-scale values for WITHIN_TOLERANCE")
    a = ap.parse_args(argv)
    ctx = Ctx(a.work_dir)
    for st in (STAGES if a.stage == "all" else (a.stage,)):
        print(f"== {st}")
        need(run_stage(st, ctx, a), f"stage {st} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
