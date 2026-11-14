"""C1-R / ICCIES 2027 V3 -- WP1 sensitivity analyses (S0 gate + EXPLORATORY / POST-HOC S1-S4).

Authorized by the Research Owner on 2026-11-13 (WP1 execution authorization; WP1 only).

What this script does
---------------------
S0  (gate, NOT exploratory): rebuilds the frozen primary G5 per-run quantities from the stored per-unit
    exact-Fraction AUCs and requires EXACT rational equality with the stored per-run metrics and the stored
    model summary, plus correct rounding of the published Table 2. If anything fails, it stops before S1-S4.
S1-S4 (EXPLORATORY / POST-HOC): re-weightings of the same stored per-unit AUCs.
    S1  IID = Aug 2014 only (IID_2014);           FAR = 2018-2025, equal-year weights (primary FAR).
    S2  IID = primary two-month mean;              FAR = 2018-2024, equal-year weights.
    S3  IID = primary two-month mean;              FAR = 2018-2023, equal-year weights.
    S4  IID = primary two-month mean;              FAR = 2018-2025, w_y = n1_F(y) / sum_y n1_F(y)
        (Full-arm malware counts; the SAME fixed w_y are applied to Full and Unseen; NOT a pooled ROC-AUC).

Inputs (read-only; nothing else is opened):
    G5_per_run_metrics.json, G5_model_independent_counts.json, G5_model_summary.json   (data-bearing)
    SHA256SUMS.txt                                     (integrity of the three JSONs)
    C1R_G4_Work/G5_AUTHORIZATION.json                  (provenance only: per-run score-file SHA-256 strings,
                                                        used to count distinct score tables; holds no AUCs)
No score arrays, labels, raw data, model checkpoints or training artifacts are accessed. G4/G5 are not run.
A sys.addaudithook() records every file open / directory listing; the run fails if anything outside the
allowed inputs, the output directory and the Python installation is touched.

Arithmetic: fractions.Fraction throughout. Floats appear only as display copies (float(Fraction), which is
correctly rounded). The structure 4 models x 5 runs is preserved; models are never pooled.

Usage:  python -I wp1_sensitivity.py [--out-dir DIR] [--overwrite]
Output is deterministic (no timestamps), so a re-run into another directory must be byte-identical.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from fractions import Fraction

# ----------------------------------------------------------------------------------------- constants
ANALYSIS_DATE = "2026-11-13"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G5_DIR = os.path.normcase(os.path.abspath(os.environ.get("C1R_G5_AGGREGATES", os.path.join(_REPO, "data", "g5_aggregates"))))
AUTH_PATH = os.path.normcase(os.path.abspath(os.environ.get("C1R_SCORE_TABLE_HASHES", os.path.join(_REPO, "data", "provenance", "score_table_sha256.json"))))
DATA_FILES = ("G5_per_run_metrics.json", "G5_model_independent_counts.json", "G5_model_summary.json")
SUMS_FILE = "SHA256SUMS.txt"
DEFAULT_OUT = os.path.dirname(os.path.abspath(__file__))

MODELS = ("svm", "lgbm", "xgb", "mlp")                      # Table 2 row order
MODEL_LABEL = {"svm": "SVM", "lgbm": "LightGBM", "xgb": "XGBoost", "mlp": "MLP"}
RUNS = ("1", "2", "3", "4", "5")
IID_UNITS = ("IID_2013", "IID_2014")
FAR_UNITS = tuple(f"FAR_{y}" for y in range(2018, 2026))
NEAR_UNITS = ("NEAR_2016", "NEAR_2017")
ALL_UNITS = IID_UNITS + NEAR_UNITS + FAR_UNITS
METRICS = ("A_IID_F", "A_IID_U", "A_FAR_F", "A_FAR_U", "D_F", "D_U", "Delta_IID", "Delta_FAR", "Delta_deg")
AUC_METRICS = METRICS[:6]                                   # displayed at 4 dp
CONTRAST_METRICS = METRICS[6:]                              # displayed x10^3 at 2 dp

EXPLORATORY = "EXPLORATORY / POST-HOC"

# Published Table 2 (identical in ICICT2027_FINAL_LNNS_20261110/overleaf_project/main.tex L373-376 and
# ICICT2027_10PAGE_CANDIDATE/overleaf_project/main.tex L360-363, sha256 0a3a555c...ea1). AUCs at 4 dp;
# contrasts x10^-3 at 2 dp. Caption: over runs, Delta_deg x10^3 ranges -1.84..-1.69 (LGBM), -0.92..-0.67 (MLP).
PUBLISHED_TABLE2 = {
    "svm":  ("0.9898", "0.9886", "0.8145", "0.8141", "0.1752", "0.1746", "-1.11", "-0.45", "-0.65"),
    "lgbm": ("0.9965", "0.9950", "0.8131", "0.8134", "0.1834", "0.1816", "-1.53", "+0.25", "-1.78"),
    "xgb":  ("0.9963", "0.9947", "0.8466", "0.8463", "0.1497", "0.1484", "-1.60", "-0.29", "-1.31"),
    "mlp":  ("0.9947", "0.9929", "0.8263", "0.8253", "0.1684", "0.1676", "-1.80", "-1.00", "-0.80"),
}
PUBLISHED_CAPTION_RANGES = {"lgbm": ("-1.84", "-1.69"), "mlp": ("-0.92", "-0.67")}

EXACT_RE = re.compile(r"^-?[0-9]+/[0-9]+$")


# ----------------------------------------------------------------------------------------- access audit
class AccessAudit:
    """Records every file open and directory listing made by this process (sys.addaudithook)."""

    def __init__(self):
        self.events = []
        self.py_roots = tuple(sorted({os.path.normcase(os.path.abspath(p)) for p in
                                      (sys.base_prefix, sys.prefix, sys.exec_prefix, sys.base_exec_prefix)}))

    def hook(self, event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
            self.events.append(("open", os.path.normcase(os.path.abspath(p)), mode))
        elif event in ("os.listdir", "os.scandir") and args:
            p = args[0]
            if isinstance(p, (str, bytes, os.PathLike)):
                self.events.append((event, os.path.normcase(os.path.abspath(os.fsdecode(p))), ""))

    def classify(self, out_dir, script_path):
        allowed_inputs = {os.path.normcase(os.path.join(G5_DIR, f)) for f in DATA_FILES + (SUMS_FILE,)} | {AUTH_PATH}
        out_dir = os.path.normcase(os.path.abspath(out_dir))
        script_path = os.path.normcase(os.path.abspath(script_path))
        inputs, outputs, python, violations = set(), set(), 0, []
        for kind, p, mode in self.events:
            if kind == "open" and p in allowed_inputs:
                if any(c in mode for c in "wax+"):
                    violations.append(f"write-mode open of input {p} ({mode})")
                inputs.add(p)
            elif kind == "open" and p == script_path:
                inputs.add(p)                                           # self-hash
            elif kind == "open" and os.path.dirname(p) == out_dir:
                outputs.add(os.path.basename(p))
            elif p.startswith(self.py_roots):
                python += 1
            else:
                violations.append(f"{kind} {p} {mode}".strip())
        return {"inputs_opened": sorted(_display_path(p) for p in inputs),
                "output_basenames_opened": sorted(outputs),
                "python_installation_events": "present (stdlib imports; count omitted, varies with caching)"
                if python else "none",
                "violations": violations}


def _display_path(p):
    if p.startswith(G5_DIR):
        return "data/g5_aggregates/" + os.path.relpath(p, G5_DIR).replace("\\", "/")
    if p == AUTH_PATH:
        return "data/provenance/score_table_sha256.json"
    return os.path.basename(p)


# ----------------------------------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fx(s):
    """Stored exact string -> Fraction; the string must be the canonical reduced 'p/q' form."""
    if not isinstance(s, str) or not EXACT_RE.match(s):
        raise ValueError(f"non-canonical exact representation: {s!r}")
    f = Fraction(s)
    if f"{f.numerator}/{f.denominator}" != s:
        raise ValueError(f"exact string not in lowest terms: {s!r}")
    return f


def ex(f):
    """Fraction -> report form."""
    return {"exact": f"{f.numerator}/{f.denominator}", "value": float(f)}


def mean(vals):
    vals = list(vals)
    return sum(vals, Fraction(0)) / len(vals)


def sign(f):
    return "negative" if f < 0 else "positive" if f > 0 else "zero"


def is_half_tie(f, ndigits):
    """True iff f*10^ndigits lies exactly halfway between two integers (rounding convention would matter)."""
    g = f * 10 ** ndigits
    return (2 * g).denominator == 1 and g.denominator == 2


def fmt_dec(f, nd):
    """Exact decimal string of f rounded to nd places (Fraction.__round__: exact, ties-to-even; ties are
    detected separately and reported, see is_half_tie)."""
    q = round(f, nd)
    sgn = "-" if q < 0 else ""
    q = abs(q)
    scaled = q * 10 ** nd
    assert scaled.denominator == 1
    s = str(scaled.numerator).rjust(nd + 1, "0")
    return f"{sgn}{s[:-nd]}.{s[-nd:]}"


def display_exact(metric, f):
    if metric in AUC_METRICS:
        return fmt_dec(f, 4)
    s = fmt_dec(f * 1000, 2)
    return s if s.startswith("-") else "+" + s


def same_decimal(a, b):
    return Fraction(a) == Fraction(b)


# ----------------------------------------------------------------------------------------- core math
def unit_aucs(prm, model, run):
    """{unit: (A_F, A_U, Delta_u)} as Fractions from the stored per-run metrics."""
    units = prm[model][run]["units"]
    out = {}
    for u in ALL_UNITS:
        out[u] = (fx(units[u]["A_F"]["exact"]), fx(units[u]["A_U"]["exact"]), fx(units[u]["Delta_u"]["exact"]))
    return out


def summarize(aucs, iid_w, far_w):
    """All nine quantities for one model x run under unit weights iid_w, far_w ({unit: Fraction}, sum 1)."""
    assert sum(iid_w.values()) == 1 and sum(far_w.values()) == 1
    A_IID_F = sum((w * aucs[u][0] for u, w in iid_w.items()), Fraction(0))
    A_IID_U = sum((w * aucs[u][1] for u, w in iid_w.items()), Fraction(0))
    A_FAR_F = sum((w * aucs[u][0] for u, w in far_w.items()), Fraction(0))
    A_FAR_U = sum((w * aucs[u][1] for u, w in far_w.items()), Fraction(0))
    r = {"A_IID_F": A_IID_F, "A_IID_U": A_IID_U, "A_FAR_F": A_FAR_F, "A_FAR_U": A_FAR_U,
         "D_F": A_IID_F - A_FAR_F, "D_U": A_IID_U - A_FAR_U,
         "Delta_IID": A_IID_U - A_IID_F, "Delta_FAR": A_FAR_U - A_FAR_F}
    r["Delta_deg"] = r["D_U"] - r["D_F"]
    assert r["Delta_deg"] == r["Delta_IID"] - r["Delta_FAR"]                  # decomposition identity
    return r


def equal_weights(units):
    return {u: Fraction(1, len(units)) for u in units}


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    script_path = os.path.abspath(__file__)

    audit = AccessAudit()
    sys.addaudithook(audit.hook)

    out_names = ("WP1_S0_S4_RESULTS.json", "WP1_S0_S4_RESULTS.csv")
    for n in out_names:
        if os.path.exists(os.path.join(out_dir, n)) and not args.overwrite:
            sys.exit(f"REFUSING to overwrite existing {n} in {out_dir} (pass --overwrite deliberately)")

    gates = {}
    log = []

    def say(msg):
        log.append(msg)
        print(msg)

    # ---------------------------------------------------------------- Gate A: input SHA-256
    with open(os.path.join(G5_DIR, SUMS_FILE), "r", encoding="utf-8") as fh:
        sums = {}
        for line in fh:
            line = line.strip()
            if line:
                h, name = line.split(None, 1)
                sums[name.lstrip("*")] = h.lower()
    hashes_before = {}
    for f in DATA_FILES:
        p = os.path.join(G5_DIR, f)
        hashes_before[f] = sha256_file(p)
        if f not in sums or sums[f] != hashes_before[f]:
            sys.exit(f"GATE A FAIL: SHA-256 mismatch for {f}: observed {hashes_before[f]} vs listed {sums.get(f)}")
    sums_sha = sha256_file(os.path.join(G5_DIR, SUMS_FILE))
    gates["A_input_sha256"] = {"status": "PASS", "files": {f: hashes_before[f] for f in DATA_FILES},
                               "SHA256SUMS.txt_sha256": sums_sha}
    say("Gate A (input SHA-256): PASS")

    def load(name):
        with open(os.path.join(G5_DIR, name), "r", encoding="utf-8") as fh:
            return json.load(fh)
    prm = load("G5_per_run_metrics.json")
    cnt = load("G5_model_independent_counts.json")
    summ = load("G5_model_summary.json")

    # ---------------------------------------------------------------- representation inspection
    rep = {"exact_fields_checked": 0, "noncanonical": [], "float_mismatch": []}

    def walk(o, path):
        if isinstance(o, dict):
            if set(o) == {"exact", "value"}:
                rep["exact_fields_checked"] += 1
                try:
                    f = fx(o["exact"])
                except ValueError as e:
                    rep["noncanonical"].append(f"{path}: {e}")
                    return
                if float(f) != o["value"]:
                    rep["float_mismatch"].append(path)
                return
            for k, v in o.items():
                walk(v, f"{path}.{k}")
    walk(prm, "per_run"); walk(summ, "summary"); walk(cnt, "counts")
    rep_ok = not rep["noncanonical"] and not rep["float_mismatch"]
    say(f"Exact-Fraction representation: {rep['exact_fields_checked']} fields, canonical reduced p/q, "
        f"float copies equal float(exact): {'OK' if rep_ok else 'FAIL'}")

    # ---------------------------------------------------------------- S0 gate
    s0_fail = []
    if not rep_ok:
        s0_fail.append("exact representation check failed")
    if set(prm) != set(MODELS) or set(summ) != set(MODELS):
        s0_fail.append("model set differs")
    for m in MODELS:
        if tuple(sorted(prm[m])) != RUNS:
            s0_fail.append(f"{m}: run set differs")
        for r in RUNS:
            if prm[m][r]["state"] != "ESTIMABLE" or prm[m][r]["undefined_components"] or \
                    prm[m][r]["paired_undefined_primary_units"]:
                s0_fail.append(f"{m} r{r}: not fully ESTIMABLE")
            if set(prm[m][r]["units"]) != set(ALL_UNITS):
                s0_fail.append(f"{m} r{r}: unit set differs")
        if summ[m]["state"] != "COMPLETE":
            s0_fail.append(f"{m}: summary state {summ[m]['state']}")

    # class counts: per-run unit counts must equal the model-independent counts (F and U)
    counts = {}
    for u in ALL_UNITS:
        counts[u] = {a: (cnt[u][a]["malware"], cnt[u][a]["benign"]) for a in ("F", "U")}
        for a in ("F", "U"):
            if cnt[u][a]["malware"] + cnt[u][a]["benign"] != cnt[u][a]["n"]:
                s0_fail.append(f"counts {u} {a}: malware+benign != n")
        for m in MODELS:
            for r in RUNS:
                for a in ("F", "U"):
                    pu = prm[m][r]["units"][u][a]
                    if (pu["malware"], pu["benign"]) != counts[u][a]:
                        s0_fail.append(f"counts {m} r{r} {u} {a} differ from model-independent counts")

    AUCS = {m: {r: unit_aucs(prm, m, r) for r in RUNS} for m in MODELS}
    n_delta_u = 0
    for m in MODELS:
        for r in RUNS:
            for u in ALL_UNITS:
                aF, aU, dU = AUCS[m][r][u]
                n_delta_u += 1
                if aU - aF != dU:
                    s0_fail.append(f"{m} r{r} {u}: stored Delta_u != A_U - A_F")

    primary_iid, primary_far = equal_weights(IID_UNITS), equal_weights(FAR_UNITS)
    n_exact = 0
    S0 = {}
    for m in MODELS:
        S0[m] = {}
        for r in RUNS:
            res = summarize(AUCS[m][r], primary_iid, primary_far)
            S0[m][r] = res
            for k in METRICS:
                n_exact += 1
                if res[k] != fx(prm[m][r][k]["exact"]):
                    s0_fail.append(f"{m} r{r} {k}: recomputed != stored per-run value")
        # model summary (mean/min/max/per_run), mirroring metrics_g5.model_summary()
        for k in METRICS:
            stored = summ[m] if k == "Delta_deg" else summ[m]["components"][k]
            pr = [S0[m][r][k] for r in RUNS]
            for r in RUNS:
                n_exact += 1
                if fx(stored["per_run"][r]["exact"]) != S0[m][r][k]:
                    s0_fail.append(f"summary {m} {k} run {r} mismatch")
            for stat, val in (("mean", mean(pr)), ("min", min(pr)), ("max", max(pr))):
                n_exact += 1
                if fx(stored[stat]["exact"]) != val:
                    s0_fail.append(f"summary {m} {k} {stat} mismatch")

    # Table 2 rounding + caption ranges + text claim "negative in all 20 runs"
    table2 = {}
    ties = []
    for m in MODELS:
        row = {}
        for i, k in enumerate(METRICS):
            v = mean(S0[m][r][k] for r in RUNS)
            nd, val = (4, v) if k in AUC_METRICS else (2, v * 1000)
            if is_half_tie(val, nd):
                ties.append(f"{m} {k}")
            shown = display_exact(k, v)
            pub = PUBLISHED_TABLE2[m][i]
            ok = same_decimal(shown, pub) and (k in AUC_METRICS or shown[0] == pub[0])
            row[k] = {"recomputed_display": shown, "published": pub, "match": ok}
            if not ok:
                s0_fail.append(f"Table 2 {m} {k}: recomputed {shown} vs published {pub}")
        table2[m] = row
    caption = {}
    for m, (lo, hi) in PUBLISHED_CAPTION_RANGES.items():
        vals = [S0[m][r]["Delta_deg"] for r in RUNS]
        got = (display_exact("Delta_deg", min(vals)), display_exact("Delta_deg", max(vals)))
        ok = same_decimal(got[0], lo) and same_decimal(got[1], hi)
        caption[m] = {"recomputed": got, "published": (lo, hi), "match": ok}
        if not ok:
            s0_fail.append(f"Table 2 caption range {m}: {got} vs {(lo, hi)}")
    if ties:
        s0_fail.append(f"exact half-way rounding ties (convention-dependent): {ties}")
    all_neg_20 = all(S0[m][r]["Delta_deg"] < 0 for m in MODELS for r in RUNS)

    gates["B_S0_exact_reproduction"] = {
        "status": "PASS" if not s0_fail else "FAIL",
        "exact_equalities_checked": n_exact,
        "stored_Delta_u_identity_checked": n_delta_u,
        "representation": rep,
        "table2": table2, "table2_caption_ranges": caption, "half_way_rounding_ties": ties,
        "text_claim_negative_in_all_20_runs": all_neg_20,
        "failures": s0_fail,
        "equality": "exact rational equality (fractions.Fraction ==); no floating-point tolerance used",
    }
    if s0_fail:
        say("Gate B (S0 exact reproduction): FAIL -- STOPPING before any exploratory computation")
        for f in s0_fail:
            say("   " + f)
        fail_path = os.path.join(out_dir, "WP1_S0_FAILURE.json")
        with open(fail_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"gates": gates}, fh, indent=1, sort_keys=True, default=str)
        sys.exit(2)
    say(f"Gate B (S0 exact reproduction): PASS ({n_exact} exact equalities, {n_delta_u} Delta_u identities, "
        f"Table 2 36/36 cells + 2 caption ranges)")

    # ---------------------------------------------------------------- distinct score tables (hash strings only)
    with open(AUTH_PATH, "r", encoding="utf-8") as fh:
        auth = json.load(fh)
    auth_sha = sha256_file(AUTH_PATH)
    score_hash = {m: {r: auth["evidence"][f"{m}_r{r}"]["scores_file_sha256"] for r in RUNS} for m in MODELS}
    tables = {}                                                   # hash -> [(m, r), ...]
    for m in MODELS:
        for r in RUNS:
            tables.setdefault(score_hash[m][r], []).append((m, r))
    # consistency: identical score files must carry identical stored per-unit AUCs
    hash_consistency = []
    for h, members in tables.items():
        m0, r0 = members[0]
        for m, r in members[1:]:
            if AUCS[m][r] != AUCS[m0][r0]:
                hash_consistency.append(f"{m} r{r} vs {m0} r{r0}: same score hash, different stored AUCs")
    if hash_consistency:
        sys.exit("score-hash consistency failure: " + "; ".join(hash_consistency))
    distinct_tables = {
        "source": "data/provenance/score_table_sha256.json evidence.<model>_r<run>.scores_file_sha256 (hash strings only)",
        "G5_AUTHORIZATION.json_sha256": auth_sha,
        "n_distinct": len(tables),
        "per_model": {m: len({score_hash[m][r] for r in RUNS}) for m in MODELS},
        "groups": {h: [f"{m}_r{r}" for m, r in mem] for h, mem in sorted(tables.items())},
        "consistency_check": "identical score-file hashes <=> identical stored per-unit AUCs: PASS "
                             "(same-hash runs verified identical)",
    }
    say(f"Distinct score tables (from stored score-file hashes): {len(tables)} "
        f"{distinct_tables['per_model']}")

    # ---------------------------------------------------------------- count bounds (Proposition 1)
    B = {}
    for u in ALL_UNITS:
        (n1, n0), (n1u, n0u) = counts[u]["F"], counts[u]["U"]
        B[u] = 1 - Fraction(n1u * n0u, n1 * n0)
    bound_violations = [f"{m} r{r} {u}" for m in MODELS for r in RUNS for u in ALL_UNITS
                        if abs(AUCS[m][r][u][2]) > B[u]]
    if bound_violations:
        sys.exit(f"|Delta_u| <= B_u violated (would contradict Proposition 1): {bound_violations}")

    # ---------------------------------------------------------------- analyses
    n1F = {u: counts[u]["F"][0] for u in FAR_UNITS}
    tot = sum(n1F.values())
    s4_w = {u: Fraction(n1F[u], tot) for u in FAR_UNITS}
    SPECS = {
        "S0": dict(label="PRIMARY (frozen G5 estimand; reproduction gate, not exploratory)",
                   iid=primary_iid, far=primary_far,
                   definition="IID = equal mean of IID_2013 (Dec 2013) and IID_2014 (Aug 2014); "
                              "FAR = equal-year mean of 2018-2025."),
        "S1": dict(label=EXPLORATORY, iid=equal_weights(("IID_2014",)), far=primary_far,
                   definition="IID = IID_2014 (Aug 2014) only; FAR = equal-year mean of 2018-2025 (primary)."),
        "S2": dict(label=EXPLORATORY, iid=primary_iid, far=equal_weights(FAR_UNITS[:7]),
                   definition="IID = primary two-month mean; FAR = equal-year mean of 2018-2024 (2025 excluded). "
                              "Named as S-FAR24 at G0 but DISABLED; enabling it now is post-hoc (Manifest L529)."),
        "S3": dict(label=EXPLORATORY, iid=primary_iid, far=equal_weights(FAR_UNITS[:6]),
                   definition="IID = primary two-month mean; FAR = equal-year mean of 2018-2023 "
                              "(2024 and 2025 excluded)."),
        "S4": dict(label=EXPLORATORY, iid=primary_iid, far=s4_w,
                   definition="IID = primary two-month mean; FAR = sum_y w_y A(y,S) over 2018-2025 with "
                              "w_y = n1_F(y) / sum_y' n1_F(y') (Full-arm malware counts, "
                              "G5_model_independent_counts.json FAR_y.F.malware); the SAME fixed w_y are "
                              "applied to Full and Unseen. This is a weighted mean of per-year AUCs, NOT a "
                              "pooled ROC-AUC over the concatenated years."),
    }

    def src_fields(arm_letter, units):
        return [f"G5_per_run_metrics.json[<model>][<run>].units.{u}.A_{arm_letter}.exact" for u in units]

    def provenance(spec):
        iw = {u: f"{w.numerator}/{w.denominator}" for u, w in spec["iid"].items()}
        fw = {u: f"{w.numerator}/{w.denominator}" for u, w in spec["far"].items()}
        return {
            "per_run_formulas": {
                "A_IID_S": "sum_{u in IID'} a_u * A(u,S)", "A_FAR_S": "sum_{y in FAR'} b_y * A(y,S)",
                "D_S": "A_IID_S - A_FAR_S", "Delta_IID": "A_IID_U - A_IID_F", "Delta_FAR": "A_FAR_U - A_FAR_F",
                "Delta_deg": "D_U - D_F  (asserted == Delta_IID - Delta_FAR exactly)",
                "A(u,S)": "Fraction(G5_per_run_metrics.json[model][run].units[u].A_S.exact), S in {F,U}"},
            "iid_weights_a": iw, "far_weights_b": fw,
            "source_fields_F": src_fields("F", list(spec["iid"]) + list(spec["far"])),
            "source_fields_U": src_fields("U", list(spec["iid"]) + list(spec["far"])),
            "model_level": "mean / min / max over runs 1-5 of the per-run Fractions (models never pooled)",
            "bound_formulas": {"B_u": "1 - n1_U(u) n0_U(u) / (n1_F(u) n0_F(u))  "
                                      "[G5_model_independent_counts.json u.F/U.malware/benign]",
                               "bound_Delta_IID": "sum_u a_u B_u", "bound_Delta_FAR": "sum_y b_y B_y",
                               "bound_Delta_deg": "bound_Delta_IID + bound_Delta_FAR (triangle inequality)"},
        }

    analyses = {}
    for sid, spec in SPECS.items():
        per_model = {}
        for m in MODELS:
            per_run = {r: summarize(AUCS[m][r], spec["iid"], spec["far"]) for r in RUNS}
            agg = {}
            for k in METRICS:
                vals = [per_run[r][k] for r in RUNS]
                agg[k] = {"mean": mean(vals), "min": min(vals), "max": max(vals)}
            neg = sum(per_run[r]["Delta_deg"] < 0 for r in RUNS)
            pos = sum(per_run[r]["Delta_deg"] > 0 for r in RUNS)
            per_model[m] = {"per_run": per_run, "agg": agg,
                            "Delta_deg_sign_runs": {"negative": neg, "positive": pos, "zero": 5 - neg - pos}}
        # sign over distinct score tables
        tbl_sign = {"negative": 0, "positive": 0, "zero": 0}
        tbl_detail = {}
        for h, mem in sorted(tables.items()):
            m, r = mem[0]
            v = per_model[m]["per_run"][r]["Delta_deg"]
            tbl_sign[sign(v)] += 1
            tbl_detail[h[:12]] = {"runs": [f"{a}_r{b}" for a, b in mem], "Delta_deg_sign": sign(v),
                                  "Delta_deg": ex(v)}
        bIID = sum((w * B[u] for u, w in spec["iid"].items()), Fraction(0))
        bFAR = sum((w * B[u] for u, w in spec["far"].items()), Fraction(0))
        for m in MODELS:
            for r in RUNS:
                pr = per_model[m]["per_run"][r]
                assert abs(pr["Delta_IID"]) <= bIID and abs(pr["Delta_FAR"]) <= bFAR
        analyses[sid] = dict(spec=spec, per_model=per_model, table_sign=tbl_sign, table_detail=tbl_detail,
                             bounds={"Delta_IID": bIID, "Delta_FAR": bFAR, "Delta_deg": bIID + bFAR})

    # ---------------------------------------------------------------- comparison with primary + review flags
    flags = []
    for sid in ("S1", "S2", "S3", "S4"):
        for m in MODELS:
            pm, p0 = analyses[sid]["per_model"][m], analyses["S0"]["per_model"][m]
            dd, dd0 = pm["agg"]["Delta_deg"]["mean"], p0["agg"]["Delta_deg"]["mean"]
            dF = pm["agg"]["D_F"]["mean"]
            ratio = abs(dd) / dF
            cmp = {
                "Delta_deg_mean_primary": ex(dd0), "Delta_deg_mean_analysis": ex(dd),
                "Delta_deg_change": ex(dd - dd0),
                "Delta_deg_relative_change": ex((dd - dd0) / dd0),
                "D_F_mean_primary": ex(p0["agg"]["D_F"]["mean"]), "D_F_mean_analysis": ex(dF),
                "D_F_change": ex(dF - p0["agg"]["D_F"]["mean"]),
                "D_F_relative_change": ex((dF - p0["agg"]["D_F"]["mean"]) / p0["agg"]["D_F"]["mean"]),
                "abs_Delta_deg_over_D_F": ex(ratio),
                "sign_primary": sign(dd0), "sign_analysis": sign(dd),
                "sign_changed_model_mean": sign(dd) != sign(dd0),
                "runs_with_sign_change": [r for r in RUNS if sign(pm["per_run"][r]["Delta_deg"]) !=
                                          sign(p0["per_run"][r]["Delta_deg"])],
                "GF_plan_rule_ratio_gt_5pct": ratio > Fraction(5, 100),
            }
            pm["vs_primary"] = cmp
            if cmp["sign_changed_model_mean"] or cmp["runs_with_sign_change"]:
                flags.append(f"{sid} {MODEL_LABEL[m]}: Delta_deg sign change (model mean {sign(dd0)} -> "
                             f"{sign(dd)}; runs changed: {cmp['runs_with_sign_change'] or 'none'})")
            if cmp["GF_plan_rule_ratio_gt_5pct"]:
                flags.append(f"{sid} {MODEL_LABEL[m]}: |Delta_deg|/D(F) = {float(ratio):.4f} > 0.05 "
                             f"(plan rule G-F trigger)")
    say(f"Owner-review flags raised: {len(flags)}")
    for f in flags:
        say("   FLAG " + f)

    # ---------------------------------------------------------------- serialise
    def ser_analysis(sid, a):
        spec = a["spec"]
        pm_out = {}
        for m in MODELS:
            pm = a["per_model"][m]
            o = {"per_run": {r: {k: ex(pm["per_run"][r][k]) for k in METRICS} for r in RUNS},
                 "five_run_mean": {k: ex(pm["agg"][k]["mean"]) for k in METRICS},
                 "run_min": {k: ex(pm["agg"][k]["min"]) for k in METRICS},
                 "run_max": {k: ex(pm["agg"][k]["max"]) for k in METRICS},
                 "display_table2_convention": {k: display_exact(k, pm["agg"][k]["mean"]) for k in METRICS},
                 "Delta_deg_sign_runs": pm["Delta_deg_sign_runs"],
                 "n_distinct_score_tables": distinct_tables["per_model"][m]}
            if "vs_primary" in pm:
                o["vs_primary"] = pm["vs_primary"]
            pm_out[m] = o
        means = {k: [a["per_model"][m]["agg"][k]["mean"] for m in MODELS] for k in METRICS}
        return {
            "label": spec["label"],
            "definition": spec["definition"],
            "iid_units": list(spec["iid"]), "far_units": list(spec["far"]),
            "per_model": pm_out,
            "range_of_model_means": {k: {"min": ex(min(v)), "max": ex(max(v))} for k, v in means.items()},
            "Delta_deg_sign_over_distinct_score_tables": a["table_sign"],
            "Delta_deg_by_distinct_score_table": a["table_detail"],
            "count_bound": {k: ex(v) for k, v in a["bounds"].items()},
            "provenance": provenance(spec),
        }

    results = {
        "title": "C1-R / ICCIES 2027 V3 -- WP1 sensitivity analyses S0-S4",
        "analysis_date": ANALYSIS_DATE,
        "status": "S0 = primary reproduction gate. S1-S4 = EXPLORATORY / POST-HOC; NOT manuscript-ready; "
                  "pending independent verification and Research Owner acceptance.",
        "authorization": "Research Owner WP1 execution authorization, 2026-11-13 (WP1 only).",
        "primary_estimand_unchanged": True,
        "script": {"name": "wp1_sensitivity.py", "sha256": sha256_file(script_path),
                   "python": sys.version.split()[0]},
        "inputs": {f: {"sha256": hashes_before[f], "listed_in_SHA256SUMS": sums[f]} for f in DATA_FILES},
        "gates": gates,
        "distinct_score_tables": distinct_tables,
        "unit_count_bounds_B_u": {u: ex(B[u]) for u in ALL_UNITS},
        "unit_counts": {u: {"F": {"malware": counts[u]["F"][0], "benign": counts[u]["F"][1]},
                            "U": {"malware": counts[u]["U"][0], "benign": counts[u]["U"][1]}} for u in ALL_UNITS},
        "S4_weights": {u: {"n1_F": n1F[u], "w": ex(s4_w[u])} for u in FAR_UNITS} | {"_sum_n1_F": tot},
        "analyses": {sid: ser_analysis(sid, a) for sid, a in analyses.items()},
        "iid_month_contrasts_stored": {
            "note": "Stored G5 per-unit contrasts Delta_u = A_U - A_F for the two IID months (S-YEAR values already "
                    "in G5_per_run_metrics.json; no new analysis). Shown because S1 drops IID_2013: "
                    "primary Delta_IID = (Delta_IID_2013 + Delta_IID_2014)/2 and S1 Delta_IID = Delta_IID_2014.",
            "per_model": {m: {u: {"per_run": {r: ex(AUCS[m][r][u][2]) for r in RUNS},
                                  "five_run_mean": ex(mean(AUCS[m][r][u][2] for r in RUNS))}
                              for u in IID_UNITS} for m in MODELS}},
        "owner_review_flags": flags,
        "owner_review_flag_rules": [
            "Any change of sign of Delta_deg relative to the primary, at model-mean or run level.",
            "Plan Section D rule G-F (pre-committed before execution): |Delta_deg'| > 5% of D'(F) for any model. "
            "Used ONLY as a flag for Research Owner review; no result is suppressed by any threshold.",
        ],
        "log": log,
    }

    # ---------------------------------------------------------------- Gate C (access audit) + Gate F (byte identity)
    hashes_after = {f: sha256_file(os.path.join(G5_DIR, f)) for f in DATA_FILES}
    unchanged = all(hashes_after[f] == hashes_before[f] == sums[f] for f in DATA_FILES)
    gates["F_inputs_byte_identical_after_run"] = {"status": "PASS" if unchanged else "FAIL",
                                                  "sha256_after": hashes_after}
    acc = audit.classify(out_dir, script_path)
    gates["C_access_boundary"] = {"status": "PASS" if not acc["violations"] else "FAIL", **acc,
                                  "note": "runtime audit (sys.addaudithook 'open', 'os.listdir', 'os.scandir') "
                                          "up to serialisation; see WP1_VALIDATION_REPORT.md for static review"}
    gates["E_exploratory_labels"] = {
        "status": "PASS" if all(results["analyses"][s]["label"] == EXPLORATORY for s in ("S1", "S2", "S3", "S4"))
        else "FAIL"}
    if not unchanged or acc["violations"]:
        sys.exit(f"GATE FAIL: byte-identity={unchanged}, access violations={acc['violations']}")
    say("Gate C (access boundary, runtime audit): PASS; Gate E (labels): PASS; Gate F (inputs unchanged): PASS")

    # JSON
    buf = json.dumps(results, indent=1, sort_keys=False, ensure_ascii=True)
    with open(os.path.join(out_dir, "WP1_S0_S4_RESULTS.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(buf + "\n")

    # CSV (long format; one row per number)
    sio = io.StringIO()
    w = csv.writer(sio, lineterminator="\n")
    w.writerow(["analysis", "label", "model", "statistic", "metric", "exact", "float", "display",
                "display_convention", "provenance"])
    for sid, a in analyses.items():
        lab = a["spec"]["label"]
        prov = provenance(a["spec"])
        wtxt = ("a=" + ";".join(f"{u}:{v}" for u, v in prov["iid_weights_a"].items()) +
                " | b=" + ";".join(f"{u}:{v}" for u, v in prov["far_weights_b"].items()))
        for m in MODELS:
            pm = a["per_model"][m]
            rows = [(f"run_{r}", pm["per_run"][r]) for r in RUNS] + \
                   [(st, {k: pm["agg"][k][st] for k in METRICS}) for st in ("mean", "min", "max")]
            for st, vals in rows:
                for k in METRICS:
                    v = vals[k]
                    src = (f"run {st[4:]}" if st.startswith("run_") else f"{st} over runs 1-5")
                    w.writerow([sid, lab, MODEL_LABEL[m], st, k, f"{v.numerator}/{v.denominator}",
                                repr(float(v)), display_exact(k, v),
                                "4dp" if k in AUC_METRICS else "x1e3 2dp",
                                f"{k} per_run_formulas; {wtxt}; A(u,S) from G5_per_run_metrics.json[{m}]; {src}"])
        for k, v in a["bounds"].items():
            w.writerow([sid, lab, "ALL", "count_bound", f"|{k}|<=", f"{v.numerator}/{v.denominator}",
                        repr(float(v)), fmt_dec(v, 4), "4dp",
                        f"bound_{k}; B_u from G5_model_independent_counts.json; {wtxt}"])
    with open(os.path.join(out_dir, "WP1_S0_S4_RESULTS.csv"), "w", encoding="utf-8", newline="") as fh:
        fh.write(sio.getvalue())
    say("Wrote WP1_S0_S4_RESULTS.json and WP1_S0_S4_RESULTS.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
