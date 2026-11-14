"""C1-R G5 metric module — FROZEN before G4 (Manifest v0.2.4 §6 P8). Not executed at G4.

Implements exactly, with exact rational arithmetic (fractions.Fraction) for every AUC and aggregate:
  * §7.1 ROC-AUC with ties counted 1/2:  A = (R1 - n1(n1+1)/2) / (n1 n0), R1 = sum of positive midranks.
  * §7.3 definedness: an arm AUC is defined iff the arm has >= 1 benign and >= 1 malware row.
  * §7.2 A_IID = mean of 2 IID months, A_FAR = mean of 8 FAR years, D = A_IID - A_FAR,
         Delta_IID, Delta_FAR, Delta_deg = D(U) - D(F); model mean only when 5/5 runs SUCCEEDED.
  * §6 P4/P5: one sealed score table per (model, run); Full and Unseen arms are masks over it; the seal is
         re-verified and the Unseen scores are asserted to be the Full scores restricted to the mask.
  * §7.4/§7.5 retained secondaries: S-YEAR per-unit contrasts; E3 E_min and AUC_max (exact, zero rules).
No statistical inference, confidence interval, p-value, threshold metric or model ranking is computed.
"""
import csv
from fractions import Fraction

import numpy as np

from g4_seal import seal_digest

IID_UNITS = ("IID_2013", "IID_2014")
FAR_UNITS = tuple(f"FAR_{y}" for y in range(2018, 2026))
NEAR_UNITS = ("NEAR_2016", "NEAR_2017")
ALL_UNITS = IID_UNITS + NEAR_UNITS + FAR_UNITS


class MetricContractError(Exception):
    pass


# ----------------------------------------------------------------------------- §7.1 AUC (exact)
def _tie_groups(scores, labels):
    o = np.argsort(scores, kind="mergesort")
    s, lab = scores[o], labels[o].astype(np.int64)
    n = len(s)
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    starts = np.concatenate(([0], np.flatnonzero(s[1:] != s[:-1]) + 1)).astype(np.int64)
    ends = np.concatenate((starts[1:], [n])).astype(np.int64)
    pos = np.add.reduceat(lab, starts)
    return starts, ends, pos


def auc_exact(scores, labels):
    """Exact empirical ROC-AUC as a Fraction (ties 1/2), or None if either class is absent (UNDEFINED)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise MetricContractError("scores/labels shape mismatch")
    if labels.size and not np.isin(labels, (0, 1)).all():
        raise MetricContractError("labels must be 0/1")
    n1 = int(labels.sum())
    n0 = int(labels.size - n1)
    if n1 == 0 or n0 == 0:
        return None
    starts, ends, pos = _tie_groups(scores, labels)
    # doubled 1-based midrank of a group spanning [a, b) is (a + 1) + b
    r1x2 = int(np.sum((starts + 1 + ends) * pos, dtype=object)) if len(pos) else 0
    return Fraction(r1x2 - n1 * (n1 + 1), 2 * n1 * n0)


def auc_bruteforce(scores, labels):
    """O(n1*n0) reference used by the tests."""
    p = [s for s, l in zip(scores, labels) if l == 1]
    q = [s for s, l in zip(scores, labels) if l == 0]
    if not p or not q:
        return None
    num = sum(2 if a > b else 1 if a == b else 0 for a in p for b in q)
    return Fraction(num, 2 * len(p) * len(q))


def tie_diagnostics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    n1 = int(labels.sum())
    n0 = int(labels.size - n1)
    starts, ends, pos = _tie_groups(scores, labels)
    neg = (ends - starts) - pos
    mixed = (pos > 0) & (neg > 0)
    tied_pairs = int(np.sum(pos[mixed] * neg[mixed], dtype=object)) if mixed.any() else 0
    return {"tied_groups_spanning_both_classes": int(mixed.sum()), "tied_pos_neg_pairs": tied_pairs,
            "tied_pair_fraction": (tied_pairs / (n1 * n0)) if n1 and n0 else None}


# ----------------------------------------------------------------------------- inputs (sealed)
def load_seen(keys_csv, expected_sha=None):
    """Sealed G3 keys/seen table -> {gid: 'TRAIN' | 0 | 1}. Hash-verified when expected_sha is given."""
    if expected_sha is not None:
        import hashlib
        h = hashlib.sha256()
        with open(keys_csv, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != expected_sha:
            raise MetricContractError("keys/seen table hash differs from the sealed G3 value")
    out = {}
    with open(keys_csv, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            s = row["seen"]
            out[int(row["gid"])] = "TRAIN" if s == "TRAIN" else int(s)
    return out


class EvalMeta:
    """gid-aligned evaluation metadata: unit, label (1 = malware), seen (0/1) for every evaluation row."""

    def __init__(self, gids, units, labels, seen):
        self.gids = np.asarray(gids, dtype=np.int64)
        if len(self.gids) > 1 and np.any(self.gids[1:] <= self.gids[:-1]):
            raise MetricContractError("EvalMeta gids must be strictly increasing")
        self.units = np.asarray(units, dtype=object)
        self.labels = np.asarray(labels, dtype=np.int8)
        self.seen = np.asarray(seen, dtype=np.int8)
        if not (len(self.units) == len(self.labels) == len(self.seen) == len(self.gids)):
            raise MetricContractError("EvalMeta length mismatch")
        if not np.isin(self.seen, (0, 1)).all():
            raise MetricContractError("seen flags must be 0/1 for evaluation rows")


# ----------------------------------------------------------------------------- §6 P5 arms
def arms(gids, scores, digest, meta, unit):
    """Full and Unseen arms of one unit as masks over ONE sealed table (P5)."""
    if seal_digest(gids, scores) != digest:
        raise MetricContractError("sealed score table digest mismatch (P4)")
    if not np.array_equal(gids, meta.gids):
        raise MetricContractError("score gids differ from evaluation metadata (P3 join)")
    F = meta.units == unit
    U = F & (meta.seen == 0)
    fs, fl = scores[F], meta.labels[F]
    us, ul = scores[U], meta.labels[U]
    sub = (meta.seen[F] == 0)
    if not (np.array_equal(us.view(np.uint8), fs[sub].view(np.uint8)) and np.array_equal(ul, fl[sub])):
        raise MetricContractError("Unseen arm is not the Full arm restricted to the mask (P5)")
    if seal_digest(gids[U], us) != seal_digest(gids[F][sub], fs[sub]):
        raise MetricContractError("Unseen subset digest mismatch (P5)")
    return (fs, fl), (us, ul)


def _counts(lab):
    n1 = int(np.sum(lab))
    return {"n": int(len(lab)), "malware": n1, "benign": int(len(lab) - n1)}


# ----------------------------------------------------------------------------- §7.2 per run
def _mean(vals):
    return sum(vals, Fraction(0)) / len(vals)


SKLEARN_TOLERANCE = 1e-12          # Manifest §7.1: cross-check against sklearn.metrics.roc_auc_score


def sklearn_crosscheck(exact, scores, labels):
    """Runtime cross-check of a DEFINED exact AUC against sklearn (§7.1); returns |difference|."""
    from sklearn.metrics import roc_auc_score
    d = abs(float(exact) - float(roc_auc_score(labels, scores)))
    if not d < SKLEARN_TOLERANCE:
        raise MetricContractError(f"exact AUC differs from sklearn by {d} (> {SKLEARN_TOLERANCE})")
    return d


def _sub(a, b):
    return None if (a is None or b is None) else a - b


def run_metrics(gids, scores, digest, meta, sklearn_check=True):
    """Per-run E2 quantities. Every component is computed independently and is None ONLY when one of its
    own constituents is undefined (§7.3); an undefined component never erases a defined one."""
    units, max_dev = {}, 0.0
    for u in ALL_UNITS:
        (fs, fl), (us, ul) = arms(gids, scores, digest, meta, u)
        aF, aU = auc_exact(fs, fl), auc_exact(us, ul)
        if sklearn_check:
            for a, s, l in ((aF, fs, fl), (aU, us, ul)):
                if a is not None:
                    max_dev = max(max_dev, sklearn_crosscheck(a, s, l))
        units[u] = {"A_F": aF, "A_U": aU, "F": _counts(fl), "U": _counts(ul),
                    "ties_F": tie_diagnostics(fs, fl), "ties_U": tie_diagnostics(us, ul),
                    "Delta_u": _sub(aU, aF)}                                         # S-YEAR

    def agg(names, arm):
        v = [units[u][arm] for u in names]
        return None if any(x is None for x in v) else _mean(v)
    A = {"A_IID_F": agg(IID_UNITS, "A_F"), "A_IID_U": agg(IID_UNITS, "A_U"),
         "A_FAR_F": agg(FAR_UNITS, "A_F"), "A_FAR_U": agg(FAR_UNITS, "A_U")}
    out = {"units": units, **A,
           "D_F": _sub(A["A_IID_F"], A["A_FAR_F"]), "D_U": _sub(A["A_IID_U"], A["A_FAR_U"]),
           "Delta_IID": _sub(A["A_IID_U"], A["A_IID_F"]), "Delta_FAR": _sub(A["A_FAR_U"], A["A_FAR_F"])}
    out["Delta_deg"] = _sub(out["D_U"], out["D_F"])
    out["paired_undefined_primary_units"] = [u for u in IID_UNITS + FAR_UNITS if units[u]["Delta_u"] is None]
    out["undefined_components"] = [k for k in ("A_IID_F", "A_IID_U", "A_FAR_F", "A_FAR_U", "D_F", "D_U",
                                               "Delta_IID", "Delta_FAR", "Delta_deg") if out[k] is None]
    out["state"] = "ESTIMABLE" if out["Delta_deg"] is not None else "NOT ESTIMABLE"
    if out["Delta_deg"] is not None:
        assert out["Delta_deg"] == out["Delta_IID"] - out["Delta_FAR"]
        assert not out["paired_undefined_primary_units"]
    else:
        assert out["paired_undefined_primary_units"]          # §7.3: NOT ESTIMABLE iff a paired contrast is undefined
    out["sklearn_max_abs_deviation"] = max_dev if sklearn_check else None
    return out


# ----------------------------------------------------------------------------- §7.2/§8.2 per model
def model_summary(run_results, run_states, n_runs=5):
    """run_results: {r: run_metrics(...) or None}; run_states: {r: state}. Mean only if all n_runs SUCCEEDED
    and estimable; otherwise INCOMPLETE (k/n) or NOT ESTIMABLE, with individual values only."""
    ok = [r for r in range(1, n_runs + 1) if run_states.get(r) == "SUCCEEDED" and run_results.get(r) is not None]
    vals = {r: run_results[r]["Delta_deg"] for r in ok}
    comps = {}
    for k in ("A_IID_F", "A_IID_U", "A_FAR_F", "A_FAR_U", "D_F", "D_U", "Delta_IID", "Delta_FAR"):
        pr = {r: run_results[r][k] for r in ok}
        full = len(ok) == n_runs and all(v is not None for v in pr.values())
        comps[k] = {"per_run": pr, "mean": _mean(list(pr.values())) if full else None,
                    "min": min(pr.values()) if full else None, "max": max(pr.values()) if full else None}
    if any(run_results[r]["state"] == "NOT ESTIMABLE" for r in ok):
        return {"state": "NOT ESTIMABLE", "per_run": vals, "mean": None, "min": None, "max": None,
                "components": comps}
    if len(ok) < n_runs:
        return {"state": f"INCOMPLETE ({len(ok)}/{n_runs})", "per_run": vals, "mean": None, "min": None, "max": None,
                "components": comps}
    v = [vals[r] for r in ok]
    return {"state": "COMPLETE", "per_run": vals, "mean": _mean(v), "min": min(v), "max": max(v), "components": comps}


# ----------------------------------------------------------------------------- §7.5 E3 (exact)
def e3_scope(kids, labels):
    """E3 on ONE declared labeled scope: per exact vector key g with p_g malware and n_g benign rows.
    E_min = sum_g min(p_g, n_g); rate undefined if P+N = 0; AUC_max undefined if P = 0 or N = 0."""
    kids = np.asarray(kids)
    labels = np.asarray(labels, dtype=np.int64)
    cnt = {}
    for k, l in zip(kids.tolist(), labels.tolist()):
        c = cnt.setdefault(k, [0, 0])
        c[l] += 1                                   # c[0] = benign n_g, c[1] = malware p_g
    P = sum(c[1] for c in cnt.values())
    N = sum(c[0] for c in cnt.values())
    emin = sum(min(c[0], c[1]) for c in cnt.values())
    out = {"P": P, "N": N, "E_min": emin, "rate": Fraction(emin, P + N) if P + N else None,
           "conflict_keys": sum(1 for c in cnt.values() if c[0] and c[1]),
           "rows_in_conflict_keys": sum(c[0] + c[1] for c in cnt.values() if c[0] and c[1])}
    if P == 0 or N == 0:
        out["AUC_max"] = None
        return out
    groups = {}
    for c in cnt.values():                          # group keys by exact malware fraction
        f = Fraction(c[1], c[0] + c[1])
        gq = groups.setdefault(f, [0, 0])
        gq[0] += c[1]
        gq[1] += c[0]
    num2, neg_below = 0, 0                          # doubled numerator
    for f in sorted(groups):                        # ascending fraction
        pG, nG = groups[f]
        num2 += pG * nG + 2 * pG * neg_below
        neg_below += nG
    out["AUC_max"] = Fraction(num2, 2 * P * N)
    return out


def e3_auc_max_bruteforce(kids, labels):
    """Reference: maximum in-sample AUC over all deterministic scorings of the keys (all weak orders)."""
    import itertools
    keys = sorted(set(np.asarray(kids).tolist()))
    best = None
    for assign in itertools.product(range(len(keys)), repeat=len(keys)):
        m = dict(zip(keys, assign))
        a = auc_bruteforce([m[k] for k in kids], labels)
        if a is not None and (best is None or a > best):
            best = a
    return best


def fmt(x):
    """Report form: float plus exact rational string; None stays None."""
    if x is None:
        return None
    return {"value": float(x), "exact": f"{x.numerator}/{x.denominator}"}
