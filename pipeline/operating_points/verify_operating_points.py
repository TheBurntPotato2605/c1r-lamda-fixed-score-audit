#!/usr/bin/env python
"""Independent verification of the operating-point metrics (never imports metric_core or operating_points).

Recomputes, from the sealed score tables of the work directory, every seal, class count, exact ROC-AUC (alignment with
WORK/eval), AP (scikit-learn; exact Fractions where the driver reports them), TP/FP/budget at FPR <= 1% and <= 0.1%,
the five-run aggregates and the CSV rows, and compares them with the driver outputs: exact equality for integers and
fractions, |difference| <= 1e-12 for floating-point values. The frozen pipeline code is used only to build the
evaluation metadata (labels, units, seen flags). Adapted from the verifier used for the study (same checks; the private
authorization and evidence-record layer was removed).

Usage (pinned environment):
    python -X int_max_str_digits=0 pipeline/operating_points/verify_operating_points.py --work-dir WORK [--results-dir WORK/operating_points] [--out report.json]
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from fractions import Fraction as F

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
ALPHAS = {"fpr_le_1pct": 100, "fpr_le_0p1pct": 1000}
MODELS = ("svm", "lgbm", "xgb", "mlp")
STATE = {"maxdev": 0.0, "float": 0, "exact": 0}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with open(p, encoding="utf-8-sig") as fh:
        return json.load(fh)


def need(c, msg):
    if not c:
        raise AssertionError(msg)


def frac(v):
    return F(v["exact"]) if isinstance(v, dict) else F(v)


def eq(x, y, path=""):
    if isinstance(x, F):
        need(isinstance(y, dict) and frac(y) == x, "fraction " + path)
        STATE["exact"] += 1
        need(abs(y["value"] - float(x)) <= 1e-12, "fraction value " + path)
    elif isinstance(x, float):
        d = abs(x - y)
        STATE["maxdev"] = max(STATE["maxdev"], d)
        STATE["float"] += 1
        need(d <= 1e-12, "float " + path)
    elif isinstance(x, dict):
        need(set(x) <= set(y), "keys " + path)
        for k, v in x.items():
            eq(v, y[k], path + "/" + k)
    elif isinstance(x, list):
        need(len(x) == len(y), "length " + path)
        for i, (a, b) in enumerate(zip(x, y)):
            eq(a, b, path + "/" + str(i))
    else:
        need(x == y, "value " + path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--results-dir")
    ap.add_argument("--out")
    ap.add_argument("--summary-out", help="write the manuscript summary CSV computed by this verifier (Table 6 source)")
    a = ap.parse_args(argv)
    sys.set_int_max_str_digits(0)
    sys.path.insert(0, PIPE)
    import c1r_e2e as E
    ctx = E.Ctx(a.work_dir)
    import g5_eval as G5
    import metrics_g5 as MG
    from g4_data import DataSpec
    O = os.path.abspath(a.results_dir or os.path.join(ctx.work, "operating_points"))
    for line in open(os.path.join(O, "SHA256SUMS.txt"), encoding="utf-8").read().splitlines():
        h, n = line.split(None, 1)
        need(sha(os.path.join(O, n.strip().lstrip("*"))) == h, "hash " + n)
    R = load(os.path.join(O, "results.json"))
    counts = load(os.path.join(ctx.eval_dir, "G5_model_independent_counts.json"))
    auc = load(os.path.join(ctx.eval_dir, "G5_per_run_metrics.json"))
    _, meta, _, _, _ = G5.build_metadata(DataSpec(**ctx.spec()), ctx.K, MG)
    ev, groups = {}, {}
    for m in MODELS:
        for r in range(1, 6):
            d = E.slot_dir(ctx, m, r)
            j = load(os.path.join(d, "job_result.json"))
            ev[f"{m}_r{r}"] = (os.path.join(d, "scores.parquet"), j["scores"]["seal_sha256"])
            groups.setdefault(j["scores"]["file_sha256"], []).append(f"{m}_r{r}")
    need(sum(map(len, groups.values())) == 20, "20 slots")
    need({h: set(v) for h, v in R["score_table_groups"].items()} == {h: set(v) for h, v in groups.items()}, "group mapping")
    units = sorted(set(meta.units))
    vals, auc_checks, exact_ap = {}, 0, 0
    for h, slots in groups.items():
        need(sha(ev[slots[0]][0]) == h, "score file bytes " + slots[0])
        tab = pq.read_table(ev[slots[0]][0], columns=["gid", "score"])
        gids, scores = tab["gid"].to_numpy(), tab["score"].to_numpy()
        need(np.array_equal(gids, meta.gids) and np.isfinite(scores).all(), "gid/finite")
        fam = slots[0].rsplit("_r", 1)[0]
        need(scores.dtype == np.dtype("float64" if fam in ("svm", "lgbm") else "float32"), "dtype")
        packed = np.empty(len(gids), dtype=[("gid", "<i8"), ("score", scores.dtype.newbyteorder("<"))])
        packed["gid"], packed["score"] = gids, scores
        seal = hashlib.sha256(packed.tobytes()).hexdigest()
        need(all(seal == ev[s][1] for s in slots), "independent seal")
        vals[h] = {}
        for u in units:
            vals[h][u] = {}
            for arm in ("F", "U"):
                mask = meta.units == u
                if arm == "U":
                    mask &= meta.seen == 0
                sc, lab = scores[mask], meta.labels[mask]
                n, n1 = len(lab), int(lab.sum())
                n0 = n - n1
                c = counts[u][arm]
                need((n, n1, n0) == (c["n"], c["malware"], c["benign"]), "counts")
                z = {"n": n, "malware": n1, "benign": n0, "prevalence": F(n1, n) if n else None}
                uniq, index = np.unique(sc, return_inverse=True)            # ascending bins, independent of the driver
                pos = np.bincount(index, weights=lab, minlength=len(uniq)).astype(np.int64)
                neg = np.bincount(index, minlength=len(uniq)) - pos
                before = np.cumsum(neg) - neg
                anum = sum(int(p) * (2 * int(b) + int(q)) for p, b, q in zip(pos, before, neg))
                exact_auc = F(anum, 2 * n1 * n0) if n1 and n0 else None
                for s in slots:
                    m, r = s.rsplit("_r", 1)
                    st = auc[m][r]["units"][u]["A_" + arm]
                    need(exact_auc == (frac(st) if st else None), "AUC alignment")
                    auc_checks += 1
                if not n1 or not n0:
                    z.update(AP=None, tie_groups=None, positive_groups=None, **{k: None for k in ALPHAS})
                else:
                    tp = np.r_[0, np.cumsum(pos[::-1])]
                    fp = np.r_[0, np.cumsum(neg[::-1])]
                    z.update(AP=float(average_precision_score(lab, sc)), tie_groups=len(uniq), positive_groups=int(np.count_nonzero(pos)))
                    stored = R["per_score_table"][h][u][arm]
                    if "AP_exact" in stored:
                        ex = sum((F(int(p) * int(t), int(t + f)) for p, t, f in zip(pos[::-1], tp[1:], fp[1:]) if p), F(0)) / n1
                        need(ex == frac(stored["AP_exact"]), "exact AP")
                        z["AP_exact"] = ex
                        exact_ap += 1
                    for k, den in ALPHAS.items():
                        budget = n0 // den
                        j = int(np.flatnonzero(fp <= budget)[-1])
                        t, f = int(tp[j]), int(fp[j])
                        nxt = j + 1
                        has = nxt < len(tp)
                        gain = int(tp[nxt] - tp[j]) if has else None
                        z[k] = dict(alpha=f"1/{den}", fp_budget=budget, TP=t, FP=f, tpr=F(t, n1), achieved_fpr=F(f, n0),
                                    tpr_resolution=F(1, n1), fpr_resolution=F(1, n0), low_resolution=budget < 10,
                                    tie_blocked=bool(has and gain > 0 and f < budget),
                                    next_group_fp=int(fp[nxt] - fp[j]) if has else None, next_group_tp=gain)
                vals[h][u][arm] = z
                need(set(z) == set(R["per_score_table"][h][u][arm]), "unit schema")
                eq(z, R["per_score_table"][h][u][arm], h + "/" + u + "/" + arm)
    slot_hash = {s: h for h, ss in groups.items() for s in ss}

    def v(m, r, u, arm):
        return vals[slot_hash[f"{m}_r{r}"]][u][arm]
    summary_rows = {}
    for m in MODELS:
        rep = R["per_model"][m]
        for u in units:
            rec = rep["units"][u]
            for arm in ("F", "U"):
                for k in ["AP"] + list(ALPHAS):
                    vv = [v(m, r, u, arm)[k] if k == "AP" else v(m, r, u, arm)[k]["tpr"] for r in range(1, 6)]
                    mean = math.fsum(vv) / 5 if k == "AP" else sum(vv, F(0)) / 5
                    eq(dict(mean=mean, min=min(vv), max=max(vv)), rec[arm][k], m + "/" + u + "/" + arm + "/" + k)
            for k in ["AP"] + list(ALPHAS):
                dif = [v(m, r, u, "U")[k] - v(m, r, u, "F")[k] if k == "AP" else v(m, r, u, "U")[k]["tpr"] - v(m, r, u, "F")[k]["tpr"] for r in range(1, 6)]
                eq(dif, rec["U_minus_F"]["AP_per_run" if k == "AP" else k + "_tpr_per_run"], m + "/contrasts")
        for arm in ("F", "U"):
            sec = []
            for r in range(1, 6):
                res = {}
                for k in ["AP"] + list(ALPHAS):
                    means = {}
                    for dom in ("IID", "FAR"):
                        us = [u for u in units if u.startswith(dom + "_")]
                        vv = [v(m, r, u, arm)[k] if k == "AP" else v(m, r, u, arm)[k]["tpr"] for u in us]
                        means[dom] = math.fsum(vv) / len(vv) if k == "AP" else sum(vv, F(0)) / len(vv)
                        res[f"{k}_{dom}_mean"] = means[dom]
                    res[k + "_IID_minus_FAR"] = means["IID"] - means["FAR"]
                sec.append(res)
            means = {k: math.fsum([q[k] for q in sec]) / 5 if k.startswith("AP_") else sum((q[k] for q in sec), F(0)) / 5 for k in sec[0]}
            eq(dict(per_run=sec, five_run_mean=means), rep["secondary_descriptive_equal_weight_summary"][arm], m + "/secondary")
            summary_rows.setdefault(m, {"model": m}).update({arm + "_" + k: float(xx) for k, xx in means.items()})
    with open(os.path.join(O, "unit_arm_metrics.csv"), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    need(len(rows) == 480, "CSV 480 rows")
    keys = set()
    for row in rows:
        key = (row["model"], int(row["run"]), row["unit"], row["arm"])
        need(key not in keys, "CSV duplicate")
        keys.add(key)
        m, r, u, arm = key
        z = v(m, r, u, arm)
        need(row["score_table_sha256"] == slot_hash[f"{m}_r{r}"], "CSV table")
        for k in ("n", "malware", "benign", "tie_groups", "positive_groups"):
            need(int(row[k]) == z[k], "CSV " + k)
        need(abs(float(row["AP"]) - z["AP"]) <= 1e-12 and abs(float(row["prevalence"]) - float(z["prevalence"])) <= 1e-12, "CSV AP/prevalence")
        need((row["AP_exact_available"] == "True") == ("AP_exact" in z), "CSV exact flag")
        for k in ALPHAS:
            for fld in ("fp_budget", "TP", "FP"):
                need(int(row[k + "_" + fld]) == z[k][fld], "CSV counts")
            for fld in ("tpr", "achieved_fpr"):
                need(F(row[k + "_" + fld]) == z[k][fld], "CSV fraction")
            for fld in ("low_resolution", "tie_blocked"):
                need((row[k + "_" + fld] == "True") == z[k][fld], "CSV flags")
    report = {"result": "PASS", "distinct_tables": len(groups), "slots": 20, "csv_rows": len(rows),
              "operating_points": 2 * len(rows), "auc_alignment_checks": auc_checks, "exact_AP_checks": exact_ap,
              "exact_fraction_checks": STATE["exact"], "float_checks": STATE["float"], "max_abs_float_difference": STATE["maxdev"],
              "tolerance": 1e-12}
    if a.summary_out:
        with open(a.summary_out, "x", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows["svm"]))
            w.writeheader()
            w.writerows(summary_rows[m] for m in MODELS)
    print(json.dumps(report, indent=1))
    if a.out:
        with open(a.out, "x", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=1)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print("VERIFY FAIL: %s" % e, file=sys.stderr)
        sys.exit(2)
