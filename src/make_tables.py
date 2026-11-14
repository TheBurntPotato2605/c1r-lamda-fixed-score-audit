"""Regenerate the aggregate tables and count bounds of the manuscript from the published aggregates.

Inputs (published in this repository, SHA-256 checked against data/g5_aggregates/SHA256SUMS.txt):
    data/g5_aggregates/G5_per_run_metrics.json, G5_model_summary.json,
    G5_model_independent_counts.json, G5_E3.json, data/wp1_aggregates/WP1_S0_S4_RESULTS.json
Outputs (outputs/tables/):
    table4_cohorts.tex      model-independent structure of the cohorts (Table 4)
    table5_primary.tex      primary results per model (Table 5)
    table6_sensitivity.tex  exploratory S0-S4 summary (Table 6), bounds rounded UPWARD
    bounds.json             exact count bounds B_u and the weighted bounds of Eq. (5), with upward-rounded values

All arithmetic uses fractions.Fraction. Inequalities are asserted on exact values; a displayed bound is always
rounded upward so that the printed inequality is valid. Reads no raw data, labels or scores.

Usage: python -I src/make_tables.py [--out-dir outputs/tables]
"""
import argparse
import hashlib
import json
import math
import os
from fractions import Fraction as Fr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G5 = os.environ.get("C1R_G5_AGGREGATES", os.path.join(REPO, "data", "g5_aggregates"))
WP1 = os.environ.get("C1R_WP1_RESULTS", os.path.join(REPO, "data", "wp1_aggregates", "WP1_S0_S4_RESULTS.json"))
UNITS = ["IID_2013", "IID_2014", "NEAR_2016", "NEAR_2017"] + ["FAR_%d" % y for y in range(2018, 2026)]
FAR = UNITS[4:]
M = ["svm", "lgbm", "xgb", "mlp"]
LABEL = {"svm": "SVM", "lgbm": "LightGBM", "xgb": "XGBoost", "mlp": "MLP"}
R = ["1", "2", "3", "4", "5"]
ROWNAME = {"IID_2013": "IID Dec 2013", "IID_2014": "IID Aug 2014", "NEAR_2016": "NEAR 2016", "NEAR_2017": "NEAR 2017"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(name):
    listed = {}
    for ln in open(os.path.join(G5, "SHA256SUMS.txt"), encoding="utf-8"):
        if ln.strip():
            h, n = ln.split(None, 1)
            listed[n.strip().lstrip("*")] = h
    p = os.path.join(G5, name)
    if sha(p) != listed[name]:
        raise SystemExit("SHA-256 mismatch: " + name)
    return json.load(open(p, encoding="utf-8"))


def fd(x, nd):
    """Nearest rounding to nd decimals (exact, half-even on exact ties)."""
    return "%.*f" % (nd, float(round(Fr(x), nd)))


def up(x, nd):
    """Round UP to nd decimals, exactly (ceiling), so that |.| <= printed value holds."""
    q = Fr(x) * 10 ** nd
    return "%.*f" % (nd, float(Fr(math.ceil(q), 10 ** nd)))


def c(n):
    return format(n, ",").replace(",", "{,}")


def sg(x, nd=2):
    return ("$-$" if x < 0 else "$+$") + fd(abs(x), nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(REPO, "outputs", "tables"))
    out = os.path.abspath(ap.parse_args().out_dir)
    os.makedirs(out, exist_ok=True)
    P, S = load("G5_per_run_metrics.json"), load("G5_model_summary.json")
    C, E = load("G5_model_independent_counts.json"), load("G5_E3.json")
    W = json.load(open(WP1, encoding="utf-8"))

    # ---------------------------------------------------------------- count bounds B_u (Proposition 1)
    B = {u: 1 - Fr(C[u]["U"]["malware"] * C[u]["U"]["benign"], C[u]["F"]["malware"] * C[u]["F"]["benign"]) for u in UNITS}
    for m in M:
        for r in R:
            for u in UNITS:
                assert abs(Fr(P[m][r]["units"][u]["Delta_u"]["exact"])) <= B[u], (m, r, u)
    bounds = {"B_u": {u: {"exact": "%d/%d" % (B[u].numerator, B[u].denominator), "float": float(B[u]),
                          "nearest_4dp": fd(B[u], 4), "up_4dp": up(B[u], 4)} for u in UNITS}}
    biid = (B["IID_2013"] + B["IID_2014"]) / 2
    bounds["S0_IID_bound"] = {"float": float(biid), "up_3dp": up(biid, 3)}
    sens = {}
    for s in ("S0", "S1", "S2", "S3", "S4"):
        wts = {u: Fr(v) for u, v in W["analyses"][s]["provenance"]["far_weights_b"].items()}
        assert sum(wts.values()) == 1 and all(v >= 0 for v in wts.values())
        b = sum(w * B[u] for u, w in wts.items())
        assert b == Fr(W["analyses"][s]["count_bound"]["Delta_FAR"]["exact"])
        sens[s] = {"exact": "%d/%d" % (b.numerator, b.denominator), "float": float(b),
                   "nearest_4dp": fd(b, 4), "up_4dp": up(b, 4)}
    bounds["FAR_bound_by_analysis"] = sens
    bounds["note"] = "Printed inequalities use the upward-rounded value; nearest-rounded values are shown for comparison only."
    json.dump(bounds, open(os.path.join(out, "bounds.json"), "w", encoding="utf-8", newline="\n"), indent=1)

    # ---------------------------------------------------------------- Table 4
    L = []
    tr = E["TRAIN"]
    L.append("TRAIN & %s & %s & -- & -- & -- & %s & %s & %s \\\\" % (
        c(tr["P"] + tr["N"]), c(tr["P"]), c(tr["rows_in_conflict_keys"]), tr["E_min"], fd(1e6 * (1 - Fr(tr["AUC_max"]["exact"])), 1)))
    for u in UNITS:
        f_, u_ = C[u]["F"], C[u]["U"]
        n, seen = f_["n"], f_["n"] - u_["n"]
        loss = 1e6 * (1 - Fr(E[u]["AUC_max"]["exact"]))
        nm = ROWNAME.get(u, "FAR " + u.split("_")[1])
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            nm, c(n), c(f_["malware"]), fd(100 * Fr(seen, n), 2), c(u_["malware"]), fd(B[u], 4),
            c(E[u]["rows_in_conflict_keys"]), E[u]["E_min"], fd(loss, 2 if loss < Fr(1, 10) else 1)))
    open(os.path.join(out, "table4_cohorts.tex"), "w", encoding="ascii", newline="\n").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- Table 5
    L = []
    for m in M:
        k = S[m]["components"]
        g = lambda key: Fr(k[key]["mean"]["exact"])
        dd = Fr(S[m]["mean"]["exact"])
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            LABEL[m], fd(g("A_IID_F"), 4), fd(g("A_IID_U"), 4), fd(g("A_FAR_F"), 4), fd(g("A_FAR_U"), 4),
            fd(g("D_F"), 4), fd(g("D_U"), 4), sg(1000 * g("Delta_IID")), sg(1000 * g("Delta_FAR")), sg(1000 * dd)))
    open(os.path.join(out, "table5_primary.tex"), "w", encoding="ascii", newline="\n").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- Table 6
    defs = {"S0": "primary", "S1": "IID Aug 2014", "S2": "FAR to 2024", "S3": "FAR to 2023", "S4": "FAR weighted"}
    L = []
    for s in ("S0", "S1", "S2", "S3", "S4"):
        A = W["analyses"][s]
        mean = lambda m, key: Fr(A["per_model"][m]["five_run_mean"][key]["exact"])
        cells = [sg(1000 * mean(m, "Delta_deg")) for m in M] + [fd(mean(m, "D_F"), 4) for m in M]
        neg = A["Delta_deg_sign_over_distinct_score_tables"]["negative"]
        L.append("%s & %s & %s & %d & %s \\\\" % (s, defs[s], " & ".join(cells), neg, sens[s]["up_4dp"]))
    open(os.path.join(out, "table6_sensitivity.tex"), "w", encoding="ascii", newline="\n").write("\n".join(L) + "\n")
    print("tables and bounds written to", out)
    print("FAR bound S0 exact %s (%.17g); printed upward: %s" % (sens["S0"]["exact"], sens["S0"]["float"], sens["S0"]["up_4dp"]))


if __name__ == "__main__":
    main()
