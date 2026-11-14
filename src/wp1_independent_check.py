"""C1-R / ICCIES 2027 V3 -- WP1 gate D: independent arithmetic check of WP1_S0_S4_RESULTS.json / .csv.

Deliberately distinct from wp1_sensitivity.py:
  * does NOT use fractions.Fraction (asserted at exit): a minimal integer rational type Q (num, den, gcd);
  * parses the stored 'p/q' strings itself;
  * uses a different formulation: every quantity is a signed linear functional over units,
      D(S)       = sum_u (a_u - b_u) * A(u,S)
      Delta_IID  = sum_u a_u * Delta_u(stored),  Delta_FAR = sum_u b_u * Delta_u(stored)
      Delta_deg  = sum_u (a_u - b_u) * Delta_u(stored)
    i.e. contrasts come from the STORED per-unit Delta_u, not from differences of arm means;
  * rebuilds weights, S4 weights and count bounds from the counts JSON on its own;
  * rounds for Table 2 with integer half-up arithmetic and requires that no exact half-way tie occurs.
Then compares every exact number in the primary outputs (JSON and CSV) for equality.

Usage: python -I wp1_independent_check.py [--results-dir DIR]
Writes WP1_INDEPENDENT_CHECK.json into the results directory (deterministic content).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G5_DIR = os.environ.get("C1R_G5_AGGREGATES", os.path.join(_REPO, "data", "g5_aggregates"))
AUTH = os.environ.get("C1R_SCORE_TABLE_HASHES", os.path.join(_REPO, "data", "provenance", "score_table_sha256.json"))
FILES = ("G5_per_run_metrics.json", "G5_model_independent_counts.json", "G5_model_summary.json")
MODELS = ("svm", "lgbm", "xgb", "mlp")
LABEL = {"svm": "SVM", "lgbm": "LightGBM", "xgb": "XGBoost", "mlp": "MLP"}
RUNS = ("1", "2", "3", "4", "5")
IID = ("IID_2013", "IID_2014")
FAR = tuple("FAR_%d" % y for y in range(2018, 2026))
METRICS = ("A_IID_F", "A_IID_U", "A_FAR_F", "A_FAR_U", "D_F", "D_U", "Delta_IID", "Delta_FAR", "Delta_deg")
PUBLISHED = {
    "svm":  ("0.9898", "0.9886", "0.8145", "0.8141", "0.1752", "0.1746", "-1.11", "-0.45", "-0.65"),
    "lgbm": ("0.9965", "0.9950", "0.8131", "0.8134", "0.1834", "0.1816", "-1.53", "+0.25", "-1.78"),
    "xgb":  ("0.9963", "0.9947", "0.8466", "0.8463", "0.1497", "0.1484", "-1.60", "-0.29", "-1.31"),
    "mlp":  ("0.9947", "0.9929", "0.8263", "0.8253", "0.1684", "0.1676", "-1.80", "-1.00", "-0.80"),
}


class Q:
    """Exact rational with canonical (num, den>0, gcd=1) representation."""
    __slots__ = ("n", "d")

    def __init__(self, n, d=1):
        if d == 0:
            raise ZeroDivisionError
        if d < 0:
            n, d = -n, -d
        g = math.gcd(n, d)
        self.n, self.d = n // g, d // g

    @staticmethod
    def parse(s):
        neg = s.startswith("-")
        a, b = (s[1:] if neg else s).split("/")
        if not (a.isdigit() and b.isdigit()):
            raise ValueError(s)
        q = Q(-int(a) if neg else int(a), int(b))
        if (q.n, q.d) != ((-int(a) if neg else int(a)), int(b)):
            raise ValueError("not reduced: " + s)
        return q

    def __add__(s, o): return Q(s.n * o.d + o.n * s.d, s.d * o.d)
    def __sub__(s, o): return Q(s.n * o.d - o.n * s.d, s.d * o.d)
    def __mul__(s, o): return Q(s.n * o.n, s.d * o.d)
    def __truediv__(s, o): return Q(s.n * o.d, s.d * o.n)
    def __neg__(s): return Q(-s.n, s.d)
    def __eq__(s, o): return s.n == o.n and s.d == o.d
    def __lt__(s, o): return s.n * o.d < o.n * s.d
    def __gt__(s, o): return s.n * o.d > o.n * s.d
    def __abs__(s): return Q(abs(s.n), s.d)
    def __hash__(s): return hash((s.n, s.d))
    def string(s): return "%d/%d" % (s.n, s.d)
    def flt(s): return s.n / s.d                           # int true division: correctly rounded


ZERO, ONE = Q(0), Q(1)


def qsum(it):
    t = ZERO
    for x in it:
        t = t + x
    return t


def round_half_up(q, nd):
    """Exact decimal rounding, half away from zero; returns (string, was_tie)."""
    scale = 10 ** nd
    num = abs(q.n) * scale
    whole, rem = divmod(num, q.d)
    tie = 2 * rem == q.d
    if 2 * rem >= q.d:
        whole += 1
    s = str(whole).rjust(nd + 1, "0")
    out = s[:-nd] + "." + s[-nd:]
    neg = q.n < 0 and whole != 0
    return ("-" if neg else "") + out, tie


def show(metric, q):
    if metric.startswith(("A_", "D_")):
        return round_half_up(q, 4)
    s, tie = round_half_up(q * Q(1000), 2)
    return (s if s.startswith("-") else "+" + s), tie


def dec_eq(a, b):
    """Compare two decimal strings numerically and in sign."""
    def to_q(x):
        neg = x.startswith("-")
        x = x.lstrip("+-")
        i, f = x.split(".")
        return Q((-1 if neg else 1) * int(i + f), 10 ** len(f))
    return to_q(a) == to_q(b)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.dirname(os.path.abspath(__file__)))
    rd = os.path.abspath(ap.parse_args().results_dir)
    checks, fails = {}, []

    def chk(name, cond, detail=""):
        checks[name] = checks.get(name, 0) + 1
        if not cond:
            fails.append(name + ": " + detail)

    # inputs
    listed = {}
    with open(os.path.join(G5_DIR, "SHA256SUMS.txt"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                h, n = line.split()
                listed[n.lstrip("*")] = h
    for f in FILES:
        chk("input_sha256", sha(os.path.join(G5_DIR, f)) == listed[f], f)
    J = {}
    for f in FILES:
        with open(os.path.join(G5_DIR, f), encoding="utf-8") as fh:
            J[f] = json.load(fh)
    prm, cnt, summ = J[FILES[0]], J[FILES[1]], J[FILES[2]]
    with open(os.path.join(rd, "WP1_S0_S4_RESULTS.json"), encoding="utf-8") as fh:
        R = json.load(fh)

    A = {m: {r: {u: (Q.parse(prm[m][r]["units"][u]["A_F"]["exact"]), Q.parse(prm[m][r]["units"][u]["A_U"]["exact"]),
                     Q.parse(prm[m][r]["units"][u]["Delta_u"]["exact"])) for u in IID + FAR} for r in RUNS}
         for m in MODELS}
    for m in MODELS:
        for r in RUNS:
            for u in IID + FAR:
                chk("stored_Delta_u_eq_AU_minus_AF", A[m][r][u][1] - A[m][r][u][0] == A[m][r][u][2], m + r + u)

    # weights
    n1F = {u: cnt[u]["F"]["malware"] for u in FAR}
    tot = sum(n1F.values())
    eq = lambda units: {u: Q(1, len(units)) for u in units}
    W = {"S0": (eq(IID), eq(FAR)), "S1": (eq(("IID_2014",)), eq(FAR)), "S2": (eq(IID), eq(FAR[:7])),
         "S3": (eq(IID), eq(FAR[:6])), "S4": (eq(IID), {u: Q(n1F[u], tot) for u in FAR})}
    for u in FAR:
        chk("S4_weight", Q.parse(R["S4_weights"][u]["w"]["exact"]) == W["S4"][1][u], u)
    chk("S4_weight_sum", R["S4_weights"]["_sum_n1_F"] == tot)

    # count bounds
    Bu = {}
    for u in R["unit_count_bounds_B_u"]:
        n1, n0 = cnt[u]["F"]["malware"], cnt[u]["F"]["benign"]
        n1u, n0u = cnt[u]["U"]["malware"], cnt[u]["U"]["benign"]
        Bu[u] = Q(n1 * n0 - n1u * n0u, n1 * n0)
        chk("B_u", Q.parse(R["unit_count_bounds_B_u"][u]["exact"]) == Bu[u], u)

    def compute(m, r, a, b):
        units = set(a) | set(b)
        c = {u: a.get(u, ZERO) - b.get(u, ZERO) for u in units}           # signed functional
        out = {"A_IID_F": qsum(a[u] * A[m][r][u][0] for u in a), "A_IID_U": qsum(a[u] * A[m][r][u][1] for u in a),
               "A_FAR_F": qsum(b[u] * A[m][r][u][0] for u in b), "A_FAR_U": qsum(b[u] * A[m][r][u][1] for u in b),
               "D_F": qsum(c[u] * A[m][r][u][0] for u in units), "D_U": qsum(c[u] * A[m][r][u][1] for u in units),
               "Delta_IID": qsum(a[u] * A[m][r][u][2] for u in a), "Delta_FAR": qsum(b[u] * A[m][r][u][2] for u in b),
               "Delta_deg": qsum(c[u] * A[m][r][u][2] for u in units)}
        return out

    # S0 vs stored G5 (independent route)
    for m in MODELS:
        vals = {r: compute(m, r, *W["S0"]) for r in RUNS}
        for r in RUNS:
            for k in METRICS:
                chk("S0_vs_stored_per_run", vals[r][k] == Q.parse(prm[m][r][k]["exact"]), m + r + k)
        for k in METRICS:
            st = summ[m] if k == "Delta_deg" else summ[m]["components"][k]
            pr = [vals[r][k] for r in RUNS]
            chk("S0_vs_stored_summary", qsum(pr) / Q(5) == Q.parse(st["mean"]["exact"]), m + k + "mean")
            chk("S0_vs_stored_summary", min(pr) == Q.parse(st["min"]["exact"]), m + k + "min")
            chk("S0_vs_stored_summary", max(pr) == Q.parse(st["max"]["exact"]), m + k + "max")
            s, tie = show(k, qsum(pr) / Q(5))
            chk("table2_no_half_way_tie", not tie, m + k)
            chk("table2_rounding", dec_eq(s, PUBLISHED[m][METRICS.index(k)]) and
                (k.startswith(("A_", "D_")) or s[0] == PUBLISHED[m][METRICS.index(k)][0]),
                "%s %s %s vs %s" % (m, k, s, PUBLISHED[m][METRICS.index(k)]))

    # every analysis vs WP1 JSON
    with open(AUTH, encoding="utf-8") as fh:
        ev = json.load(fh)["evidence"]
    groups = {}
    for m in MODELS:
        for r in RUNS:
            groups.setdefault(ev["%s_r%s" % (m, r)]["scores_file_sha256"], []).append((m, r))
    chk("distinct_tables", R["distinct_score_tables"]["n_distinct"] == len(groups) == 12)
    mine = {}
    for sid, (a, b) in W.items():
        Ra = R["analyses"][sid]
        chk("label", Ra["label"] == ("EXPLORATORY / POST-HOC" if sid != "S0" else Ra["label"]), sid)
        chk("weights_iid", {u: Q.parse(x) for u, x in Ra["provenance"]["iid_weights_a"].items()} == a, sid)
        chk("weights_far", {u: Q.parse(x) for u, x in Ra["provenance"]["far_weights_b"].items()} == b, sid)
        bI, bF = qsum(a[u] * Bu[u] for u in a), qsum(b[u] * Bu[u] for u in b)
        for k, v in (("Delta_IID", bI), ("Delta_FAR", bF), ("Delta_deg", bI + bF)):
            chk("count_bound", Q.parse(Ra["count_bound"][k]["exact"]) == v, sid + k)
        mine[sid] = {}
        for m in MODELS:
            pm = Ra["per_model"][m]
            vals = {r: compute(m, r, a, b) for r in RUNS}
            mine[sid][m] = vals
            for r in RUNS:
                for k in METRICS:
                    q = vals[r][k]
                    chk("per_run_exact", Q.parse(pm["per_run"][r][k]["exact"]) == q, sid + m + r + k)
                    chk("per_run_float", pm["per_run"][r][k]["value"] == q.flt(), sid + m + r + k)
                chk("bound_holds", not (bI < abs(vals[r]["Delta_IID"])) and not (bF < abs(vals[r]["Delta_FAR"])),
                    sid + m + r)
            for k in METRICS:
                pr = [vals[r][k] for r in RUNS]
                for st, q in (("five_run_mean", qsum(pr) / Q(5)), ("run_min", min(pr)), ("run_max", max(pr))):
                    chk("aggregate_exact", Q.parse(pm[st][k]["exact"]) == q, sid + m + st + k)
                    chk("aggregate_float", pm[st][k]["value"] == q.flt(), sid + m + st + k)
                s, tie = show(k, qsum(pr) / Q(5))
                chk("display_no_tie", not tie, sid + m + k)
                chk("display", pm["display_table2_convention"][k] == s, "%s %s %s %s vs %s" %
                    (sid, m, k, pm["display_table2_convention"][k], s))
            neg = sum(1 for r in RUNS if vals[r]["Delta_deg"] < ZERO)
            pos = sum(1 for r in RUNS if ZERO < vals[r]["Delta_deg"])
            chk("sign_runs", pm["Delta_deg_sign_runs"] == {"negative": neg, "positive": pos, "zero": 5 - neg - pos},
                sid + m)
            if sid != "S0":
                vp = pm["vs_primary"]
                dd = qsum(vals[r]["Delta_deg"] for r in RUNS) / Q(5)
                dd0 = qsum(mine["S0"][m][r]["Delta_deg"] for r in RUNS) / Q(5)
                dF = qsum(vals[r]["D_F"] for r in RUNS) / Q(5)
                dF0 = qsum(mine["S0"][m][r]["D_F"] for r in RUNS) / Q(5)
                for key, q in (("Delta_deg_change", dd - dd0), ("Delta_deg_relative_change", (dd - dd0) / dd0),
                               ("D_F_change", dF - dF0), ("D_F_relative_change", (dF - dF0) / dF0),
                               ("abs_Delta_deg_over_D_F", abs(dd) / dF)):
                    chk("vs_primary", Q.parse(vp[key]["exact"]) == q, sid + m + key)
                chk("GF_flag", vp["GF_plan_rule_ratio_gt_5pct"] == (Q(5, 100) < abs(dd) / dF), sid + m)
                changed = [r for r in RUNS if (vals[r]["Delta_deg"] < ZERO) != (mine["S0"][m][r]["Delta_deg"] < ZERO)
                           or (ZERO < vals[r]["Delta_deg"]) != (ZERO < mine["S0"][m][r]["Delta_deg"])]
                chk("sign_change_runs", vp["runs_with_sign_change"] == changed, sid + m)
        tn = {"negative": 0, "positive": 0, "zero": 0}
        for h, mem in groups.items():
            vset = {mine[sid][m][r]["Delta_deg"] for m, r in mem}
            chk("same_hash_same_value", len(vset) == 1, sid + h)
            v = next(iter(vset))
            tn["negative" if v < ZERO else "positive" if ZERO < v else "zero"] += 1
        chk("table_sign_counts", Ra["Delta_deg_sign_over_distinct_score_tables"] == tn, sid)

    # IID month contrasts block
    for m in MODELS:
        for u in IID:
            blk = R["iid_month_contrasts_stored"]["per_model"][m][u]
            chk("iid_month_block", Q.parse(blk["five_run_mean"]["exact"]) ==
                qsum(A[m][r][u][2] for r in RUNS) / Q(5), m + u)

    # CSV vs independent values
    n_csv = 0
    with open(os.path.join(rd, "WP1_S0_S4_RESULTS.csv"), encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sid, model, st, k = row["analysis"], row["model"], row["statistic"], row["metric"]
            n_csv += 1
            if st == "count_bound":
                a, b = W[sid]
                bI, bF = qsum(a[u] * Bu[u] for u in a), qsum(b[u] * Bu[u] for u in b)
                exp = {"|Delta_IID|<=": bI, "|Delta_FAR|<=": bF, "|Delta_deg|<=": bI + bF}[k]
            else:
                m = {v: kk for kk, v in LABEL.items()}[model]
                vals = mine[sid][m]
                pr = [vals[r][k] for r in RUNS]
                exp = (vals[st[4:]][k] if st.startswith("run_") else
                       {"mean": qsum(pr) / Q(5), "min": min(pr), "max": max(pr)}[st])
            chk("csv_exact", Q.parse(row["exact"]) == exp, "%s %s %s %s" % (sid, model, st, k))
            chk("csv_float", float(row["float"]) == exp.flt(), "%s %s %s %s" % (sid, model, st, k))
            chk("csv_label", row["label"] == R["analyses"][sid]["label"], sid)
    chk("csv_row_count", n_csv == 5 * (4 * 8 * 9 + 3), str(n_csv))

    for f in FILES:
        chk("inputs_unchanged_after", sha(os.path.join(G5_DIR, f)) == listed[f], f)
    chk("no_fractions_module", "fractions" not in sys.modules)

    out = {"check": "WP1 gate D independent arithmetic check", "implementation": "wp1_independent_check.py",
           "implementation_sha256": sha(os.path.abspath(__file__)),
           "results_json_sha256": sha(os.path.join(rd, "WP1_S0_S4_RESULTS.json")),
           "results_csv_sha256": sha(os.path.join(rd, "WP1_S0_S4_RESULTS.csv")),
           "status": "PASS" if not fails else "FAIL", "n_checks": sum(checks.values()),
           "checks_by_type": dict(sorted(checks.items())), "failures": fails}
    with open(os.path.join(rd, "WP1_INDEPENDENT_CHECK.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print("Gate D independent check:", out["status"], "(%d checks)" % out["n_checks"])
    for f in fails[:30]:
        print("  FAIL", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
