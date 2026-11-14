#!/usr/bin/env python
"""Exploratory, post-hoc operating-point metrics on the sealed score tables of the public pipeline.

Computes, for every distinct score table, unit and arm (Full/Unseen): non-interpolated average precision (AP) and the
maximum attainable empirical TPR at FPR <= 1% and <= 0.1% (tie groups taken whole; in-sample operating points, not
prospective thresholds). Definitions: metric_core.py (byte-identical to the verified core, SHA-256 083a1c6b...bf6).
These quantities do not replace the protocol-locked ROC-AUC analysis, and no ROC-AUC bound applies to them.

Usage (after the `train` and `evaluate` stages of c1r_e2e.py, pinned environment):
    python -X int_max_str_digits=0 pipeline/operating_points/operating_points.py --work-dir WORK [--out-dir WORK/operating_points]
        [--compare-dir data/operating_points]

`-X int_max_str_digits=0` is required: exact AP fractions reach about 18,700 decimal digits, beyond Python's default
4,300-digit integer-string conversion limit (without it, writing the outputs fails with ValueError).

Fail-closed integrity before any output: seals re-verified by the frozen read_table; frozen arms() join (Unseen = Full
restricted to the mask); per unit-arm class counts equal WORK/eval/G5_model_independent_counts.json; exact ROC-AUC
recomputed with the frozen auc_exact equals all 480 values of WORK/eval/G5_per_run_metrics.json (alignment control);
scikit-learn cross-checks (|dAP| <= 1e-12, TP/FP exact) and exact-Fraction AP on the approved subset.
The evaluation, aggregation and output functions below are copied verbatim from the verified study driver (output file
names and the exception name neutralised); the private authorization layer was replaced by the public work-directory context.
"""
import argparse
import csv
import filecmp
import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
REPO = os.path.dirname(PIPE)
TOL = 1e-12
MODELS = ("svm", "mlp", "xgb", "lgbm")
RUNS = (1, 2, 3, 4, 5)
CORE_SHA256 = "083a1c6b27b2f9ca2df9db922af569ef5ee7275765d43eb2e9a7c00b51e87bf6"
sys.path.insert(0, HERE)
import metric_core as C  # noqa: E402


class MetricStop(Exception):
    """Fail-closed stop."""


def need(c, m):
    if not c:
        raise MetricStop(m)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _skl_check(s, lab, arm, curves):
    from sklearn.metrics import average_precision_score, roc_curve
    d = abs(float(average_precision_score(lab, s)) - arm["AP"])
    need(d <= TOL, f"AP differs from scikit-learn by {d}")
    fpr, tpr, _ = roc_curve(lab, s, drop_intermediate=False)
    fps, tps = C.np.rint(fpr * arm["benign"]).astype(C.np.int64), C.np.rint(tpr * arm["malware"]).astype(C.np.int64)
    for k, _a in C.ALPHAS:
        r = arm[k]
        j = int(C.np.searchsorted(fps, r["fp_budget"], side="right")) - 1
        need((int(tps[j]), int(fps[j])) == (r["TP"], r["FP"]), f"TPR@{k} TP/FP differ from scikit-learn roc_curve")
    return d


def evaluate_tables(meta, tables, stored_auc, stored_counts, MG, max_groups=8000, large_limit_s=600):
    """tables: list of {"sha", "slots": [(model, run)], "gids", "scores", "seal"}; stored_auc(model, run, unit, key) -> Fraction|None;
    stored_counts[unit][arm] -> {"n","malware","benign"}. Returns (per_table, checks)."""
    import numpy as np
    checks = {"count_checks": 0, "auc_alignment_checks": 0, "sklearn_ap_max_abs_dev": 0.0, "sklearn_tpr_checks": 0,
              "exact_subset": {"threshold_positive_groups": max_groups, "evaluated": 0, "max_abs_dev": 0.0},
              "exact_large_units": {}}
    for u in MG.ALL_UNITS:
        F = meta.units == u
        for arm, mask in (("F", F), ("U", F & (meta.seen == 0))):
            n, n1 = int(mask.sum()), int(meta.labels[mask].sum())
            st = stored_counts[u][arm]
            need((n, n1, n - n1) == (st["n"], st["malware"], st["benign"]), f"{u}/{arm}: class counts differ from stored G5 counts")
            checks["count_checks"] += 1
    per_table, large = {}, {}
    for t in tables:
        res = {}
        fam = t["slots"][0][0]
        for u in MG.ALL_UNITS:
            try:
                (fs, fl), (us, ul) = MG.arms(t["gids"], t["scores"], t["seal"], meta, u)
            except MG.MetricContractError as e:
                raise MetricStop(f"{t['sha'][:12]} {u}: {e}")
            aF, aU = MG.auc_exact(fs, fl), MG.auc_exact(us, ul)
            for (m, r) in t["slots"]:
                need(aF == stored_auc(m, r, u, "A_F") and aU == stored_auc(m, r, u, "A_U"), f"{m} r{r} {u}: exact ROC-AUC differs from stored G5 value (alignment)")
                checks["auc_alignment_checks"] += 2
            sub = meta.seen[meta.units == u] == 0
            um, curves = C.unit_metrics(fs, fl, sub)
            need(np.array_equal(np.sort(us), np.sort(fs[sub])), f"{u}: Unseen arm mismatch")
            for arm, s, lab in (("F", fs, fl), ("U", us, ul)):
                st = stored_counts[u][arm]
                need((um[arm]["n"], um[arm]["malware"], um[arm]["benign"]) == (st["n"], st["malware"], st["benign"]), f"{u}/{arm}: metric-core counts differ")
                if um[arm]["AP"] is None:
                    continue
                o = C.descending_order(s)
                d = _skl_check(s[o], lab[o], um[arm], curves[arm])
                checks["sklearn_ap_max_abs_dev"] = max(checks["sklearn_ap_max_abs_dev"], d)
                checks["sklearn_tpr_checks"] += len(C.ALPHAS)
                tp, fp = curves[arm]
                K = um[arm]["positive_groups"]
                if K <= max_groups:
                    ex = C.ap_exact(tp, fp, um[arm]["malware"])
                    dev = abs(ex - Fraction(um[arm]["AP"]))
                    need(dev <= Fraction(1, 10 ** 12), f"{u}/{arm}: double AP differs from exact Fraction AP by {float(dev)}")
                    um[arm]["AP_exact"] = ex
                    checks["exact_subset"]["evaluated"] += 1
                    checks["exact_subset"]["max_abs_dev"] = max(checks["exact_subset"]["max_abs_dev"], float(dev))
                elif K > large.get(fam, (0,))[0]:
                    large[fam] = (K, t["sha"], u, arm, tp, fp, um[arm]["malware"], um[arm]["AP"])
            res[u] = um
        per_table[t["sha"]] = res
    per_fam = large_limit_s / max(1, len(large))
    for fam, (K, tsha, u, arm, tp, fp, n1, ap) in sorted(large.items()):
        t0 = time.monotonic()
        ex = C.ap_exact(tp, fp, n1, deadline=t0 + per_fam)
        rec = {"table": tsha, "unit": u, "arm": arm, "positive_groups": K, "time_limit_s": per_fam, "seconds": round(time.monotonic() - t0, 1)}
        if ex is None:
            rec["status"] = "TIMEOUT (not a failure; outside the guaranteed subset)"
        else:
            dev = abs(ex - Fraction(ap))
            need(dev <= Fraction(1, 10 ** 12), f"large unit {fam} {u}/{arm}: double AP differs from exact by {float(dev)}")
            rec.update(status="EXACT_MATCH", abs_dev=float(dev))
            per_table[tsha][u][arm]["AP_exact"] = ex
        checks["exact_large_units"][fam] = rec
    return per_table, checks


def _fmean(v):
    return math.fsum(v) / len(v)


def aggregate(per_table, slot_table, MG):
    """Per slot -> per model five-run mean/min/max per unit and arm; U-F contrasts; secondary descriptive IID/FAR means."""
    keys_t = [k for k, _ in C.ALPHAS]

    def val(m, r, u, arm):
        return per_table[slot_table[f"{m}_r{r}"]][u][arm]
    models = {}
    for m in MODELS:
        units = {}
        for u in MG.ALL_UNITS:
            rec = {}
            for arm in ("F", "U"):
                ap = [val(m, r, u, arm)["AP"] for r in RUNS]
                rec[arm] = {"AP": None if None in ap else {"mean": _fmean(ap), "min": min(ap), "max": max(ap)}}
                for k in keys_t:
                    v = [None if val(m, r, u, arm)[k] is None else val(m, r, u, arm)[k]["tpr"] for r in RUNS]
                    rec[arm][k] = None if None in v else {"mean": sum(v, Fraction(0)) / 5, "min": min(v), "max": max(v)}
            ok = rec["F"]["AP"] is not None and rec["U"]["AP"] is not None
            rec["U_minus_F"] = {"AP_per_run": [val(m, r, u, "U")["AP"] - val(m, r, u, "F")["AP"] for r in RUNS] if ok else None}
            for k in keys_t:
                rec["U_minus_F"][f"{k}_tpr_per_run"] = ([val(m, r, u, "U")[k]["tpr"] - val(m, r, u, "F")[k]["tpr"] for r in RUNS] if ok else None)
            units[u] = rec
        sec = {}
        for arm in ("F", "U"):
            per_run = []
            for r in RUNS:
                iid = _fmean([val(m, r, u, arm)["AP"] for u in MG.IID_UNITS])
                far = _fmean([val(m, r, u, arm)["AP"] for u in MG.FAR_UNITS])
                e = {"AP_IID_mean": iid, "AP_FAR_mean": far, "AP_IID_minus_FAR": iid - far}
                for k in keys_t:
                    ti = sum((val(m, r, u, arm)[k]["tpr"] for u in MG.IID_UNITS), Fraction(0)) / 2
                    tf = sum((val(m, r, u, arm)[k]["tpr"] for u in MG.FAR_UNITS), Fraction(0)) / 8
                    e.update({f"{k}_IID_mean": ti, f"{k}_FAR_mean": tf, f"{k}_IID_minus_FAR": ti - tf})
                per_run.append(e)
            sec[arm] = {"per_run": per_run, "five_run_mean": {k: (_fmean([e[k] for e in per_run]) if isinstance(per_run[0][k], float)
                                                                   else sum((e[k] for e in per_run), Fraction(0)) / 5) for k in per_run[0]}}
        models[m] = {"units": units, "secondary_descriptive_equal_weight_summary": {
            "label": "descriptive analog only: NOT Delta_deg; no ROC-AUC bound applies; prevalence differs between units and arms",
            **sec}}
    return models


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
    return o


def csv_rows(per_table, slot_table, MG):
    head = ["model", "run", "score_table_sha256", "unit", "arm", "n", "malware", "benign", "prevalence", "AP", "AP_exact_available",
            "tie_groups", "positive_groups"]
    for k, _ in C.ALPHAS:
        head += [f"{k}_{x}" for x in ("fp_budget", "TP", "FP", "tpr", "achieved_fpr", "low_resolution", "tie_blocked")]
    rows = [head]
    for m in MODELS:
        for r in RUNS:
            t = slot_table[f"{m}_r{r}"]
            for u in MG.ALL_UNITS:
                for arm in ("F", "U"):
                    v = per_table[t][u][arm]
                    row = [m, r, t, u, arm, v["n"], v["malware"], v["benign"], repr(float(v["prevalence"])),
                           "" if v["AP"] is None else repr(v["AP"]), "AP_exact" in v, v["tie_groups"], v["positive_groups"]]
                    for k, _ in C.ALPHAS:
                        x = v[k]
                        row += ([""] * 7 if x is None else [x["fp_budget"], x["TP"], x["FP"], f"{x['tpr'].numerator}/{x['tpr'].denominator}",
                                                           f"{x['achieved_fpr'].numerator}/{x['achieved_fpr'].denominator}", x["low_resolution"], x["tie_blocked"]])
                    rows.append(row)
    return rows


def write_outputs(od, payload, rows):
    part = od + ".partial"
    os.makedirs(part)
    files = {}
    for name, txt in (("results.json", json.dumps(ser(payload["results"]), indent=1, sort_keys=True) + "\n"),
                      ("verification.json", json.dumps(ser(payload["verification"]), indent=1, sort_keys=True) + "\n")):
        with open(os.path.join(part, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
        files[name] = hashlib.sha256(txt.encode()).hexdigest()
    p = os.path.join(part, "unit_arm_metrics.csv")
    with open(p, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    files["unit_arm_metrics.csv"] = sha256_file(p)
    with open(os.path.join(part, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(f"{h}  {n}\n" for n, h in sorted(files.items())))
    os.rename(part, od)
    return files


# ------------------------------------------------------------------------------------------ public main
def manuscript_summary(models):
    """Five-run means of the secondary descriptive IID/FAR summaries (manuscript Table 6 source)."""
    rows = []
    for m in ("svm", "lgbm", "xgb", "mlp"):
        row = {"model": m}
        for arm in ("F", "U"):
            for k, x in models[m]["secondary_descriptive_equal_weight_summary"][arm]["five_run_mean"].items():
                row[f"{arm}_{k}"] = float(x)
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True, help="work directory of pipeline/c1r_e2e.py (after train and evaluate)")
    ap.add_argument("--out-dir", help="output directory (default WORK/operating_points; must not exist)")
    ap.add_argument("--compare-dir", help="published aggregates to compare with (e.g. data/operating_points)")
    x = ap.parse_args(argv)
    try:
        need(sys.get_int_max_str_digits() == 0, "run with: python -X int_max_str_digits=0 (exact AP fractions exceed 4300 digits)")
        need(sha256_file(os.path.join(HERE, "metric_core.py")) == CORE_SHA256, "metric_core.py differs from the verified core")
        sys.path.insert(0, PIPE)
        import c1r_e2e as E
        ctx = E.Ctx(x.work_dir)
        import numpy as np
        import g4_seal as S
        import g5_eval as G5
        import metrics_g5 as MG
        from g4_data import DataSpec
        od = os.path.abspath(x.out_dir or os.path.join(ctx.work, "operating_points"))
        need(not os.path.exists(od) and not os.path.exists(od + ".partial"), "output directory already exists: " + od)
        evid = {}
        for m in G5.MODELS:
            for r in G5.RUNS:
                dd = E.slot_dir(ctx, m, r)
                R = E.jload(os.path.join(dd, "job_result.json"))
                need(R["state"] == "SUCCEEDED" and R["scores"]["seal_sha256"] == R["scores"]["pass2_sha256"], f"{m} run {r}: not a clean slot")
                p = os.path.join(dd, "scores.parquet")
                need(sha256_file(p) == R["scores"]["file_sha256"], f"{m} run {r}: score file hash")
                evid[f"{m}_r{r}"] = {"path": p, "seal": R["scores"]["seal_sha256"], "file": R["scores"]["file_sha256"]}
        groups = {}
        for slot, e in evid.items():
            groups.setdefault(e["file"], []).append(slot)
        for k, slots in groups.items():
            need(len({evid[s]["seal"] for s in slots}) == 1, f"one score file maps to several seals ({slots})")
        groups = {k: sorted(v, key=lambda s: (MODELS.index(s.rsplit("_r", 1)[0]), int(s.rsplit("_r", 1)[1]))) for k, v in groups.items()}
        slot_table = {s: k for k, v in groups.items() for s in v}
        spec = DataSpec(**ctx.spec())
        ds, meta, kid, allz, seen = G5.build_metadata(spec, ctx.K, MG)
        need(len(meta.gids) == len(ds.eval_gids), "evaluation universe size")
        stored = E.jload(os.path.join(ctx.eval_dir, "G5_per_run_metrics.json"))
        counts = E.jload(os.path.join(ctx.eval_dir, "G5_model_independent_counts.json"))

        def stored_auc(m, r, u, key):
            v = stored[m][str(r)]["units"][u][key]
            return None if v is None else Fraction(v["exact"])
        tables = []
        for tsha, slots in groups.items():
            e = evid[slots[0]]
            gg, ss = S.read_table(e["path"], e["seal"])
            m0 = slots[0].rsplit("_r", 1)[0]
            need(ss.dtype == np.dtype(S.SCORE_DTYPES[m0]), f"{slots[0]}: score dtype")
            tables.append({"sha": tsha, "slots": [(s.rsplit("_r", 1)[0], int(s.rsplit("_r", 1)[1])) for s in slots],
                           "gids": gg, "scores": ss, "seal": e["seal"]})
        t0 = time.monotonic()
        per_table, checks = evaluate_tables(meta, tables, stored_auc, counts, MG, 8000, 600)
        checks["metric_seconds"] = round(time.monotonic() - t0, 1)
        models = aggregate(per_table, slot_table, MG)
        results = {"analysis": "exploratory, post-hoc operating-point metrics (do not replace the primary ROC-AUC analysis)",
                   "definitions": C.__doc__, "score_table_groups": groups, "per_score_table": per_table, "per_model": models}
        verification = {"integrity": ["score files hash-verified against job_result.json; same file => same seal",
                                      "seals recomputed by the frozen read_table; frozen arms() joins; Unseen == Full restricted to the mask",
                                      "class counts == WORK/eval counts; exact ROC-AUC == all WORK/eval per-run values"], **checks}
        files = write_outputs(od, {"results": results, "verification": verification}, csv_rows(per_table, slot_table, MG))
        summ = manuscript_summary(models)
        with open(os.path.join(od, "manuscript_summary.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summ[0]), lineterminator="\r\n")
            w.writeheader()
            w.writerows(summ)
        report = {"result": "DONE", "out_dir": od, "files": files, "distinct_score_tables": len(groups),
                  "auc_alignment_checks": checks["auc_alignment_checks"], "sklearn_ap_max_abs_dev": checks["sklearn_ap_max_abs_dev"]}
        if x.compare_dir:
            ref = os.path.join(x.compare_dir, "unit_arm_metrics.csv")
            same = os.path.isfile(ref) and filecmp.cmp(ref, os.path.join(od, "unit_arm_metrics.csv"), shallow=False)
            # the published manuscript_summary.csv was written by the independent verifier (its own floating-point
            # summation); the driver's values must agree within the verification tolerance
            with open(os.path.join(x.compare_dir, "manuscript_summary.csv"), newline="", encoding="utf-8") as fh:
                pub = {r_["model"]: r_ for r_ in csv.DictReader(fh)}
            dev = max(abs(float(pub[r_["model"]][k]) - v) for r_ in summ for k, v in r_.items() if k != "model")
            report["compare"] = {"unit_arm_metrics.csv_byte_identical": same, "manuscript_summary_max_abs_difference": dev}
            need(same and dev <= TOL, "outputs differ from the published aggregates: %s" % report["compare"])
        print(json.dumps(report, indent=1))
        return 0
    except MetricStop as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
