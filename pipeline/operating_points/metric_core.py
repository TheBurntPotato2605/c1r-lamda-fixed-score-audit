"""C1-R V4-WP5 metric core (exploratory, post-hoc; record C1R-V4-WP5-PH-02). Pure numpy; no I/O.

Definitions (scores ranked in DESCENDING order; rows with equal scores form ONE tie group that is always taken whole:
no tie-breaking, no interpolation, no randomization):
  TP_k, FP_k  cumulative malware / benign counts after the k-th tie group (k = 1..G); TP_0 = FP_0 = 0.
  AP          non-interpolated average precision  = (1/n1) * sum_k (TP_k - TP_{k-1}) * TP_k / (TP_k + FP_k).
              Evaluated in IEEE double precision: integer numerators/denominators (exact int64), one floating division per
              term, terms summed with math.fsum. Accuracy is established empirically by independent verification
              (scikit-learn and a separate implementation, tolerance 1e-12) and by exact-Fraction evaluation on a subset.
  TPR@FPR<=a  maximum attainable empirical TPR over tie-group boundaries k = 0..G subject to FP_k <= floor(a * n0)
              (equivalently FP_k / n0 <= a); TP, FP and the budget are exact integers, TPR and FPR exact Fractions.
              The operating point is chosen in-sample per unit and arm (an oracle point), not a prospective threshold.
No ROC-AUC bound (count bound, pair decomposition, conflict oracle) applies to these quantities.
"""
import math
import time
from fractions import Fraction

import numpy as np

ALPHAS = (("fpr_le_1pct", Fraction(1, 100)), ("fpr_le_0p1pct", Fraction(1, 1000)))
LOW_RESOLUTION_FP_BUDGET = 10          # flag: fewer than 10 false positives define the operating point
INT53 = 2 ** 53


class WP5ContractError(Exception):
    """Input-contract violation in the metric core."""


def descending_order(scores):
    """Stable ascending argsort reversed: equal scores stay contiguous (one tie group); within-group order is irrelevant."""
    return np.argsort(scores, kind="stable")[::-1]


def grouped_counts(s_desc, l_desc):
    """Cumulative (TP, FP) at the end of every tie group of a descending-sorted arm. int64 arrays of length G."""
    n = len(s_desc)
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    ends = np.concatenate((np.flatnonzero(s_desc[1:] != s_desc[:-1]) + 1, [n])).astype(np.int64)
    ctp = np.cumsum(l_desc.astype(np.int64))
    tp = ctp[ends - 1]
    return tp, ends - tp


def _check_arm(s, lab):
    if s.shape != lab.shape or s.ndim != 1:
        raise WP5ContractError("scores/labels shape mismatch")
    if s.size and not np.isfinite(s).all():
        raise WP5ContractError("non-finite score")
    if lab.size and not np.isin(lab, (0, 1)).all():
        raise WP5ContractError("labels must be 0/1")
    if s.size > 1 and np.any(s[1:] > s[:-1]):
        raise WP5ContractError("arm is not sorted in descending score order")


def ap_terms(tp, fp):
    """Integer numerators/denominators of the AP terms (only groups that add malware)."""
    dtp = np.diff(tp, prepend=np.int64(0))
    pos = dtp > 0
    num, den = dtp[pos] * tp[pos], tp[pos] + fp[pos]
    if num.size and (int(num.max()) >= INT53 or int(den.max()) >= INT53):
        raise WP5ContractError("AP term exceeds exact int64->float64 range")
    return num, den


def ap_double(tp, fp, n1):
    num, den = ap_terms(tp, fp)
    terms = num.astype(np.float64) / den.astype(np.float64)
    return math.fsum(terms.tolist()) / n1


def ap_exact(tp, fp, n1, deadline=None):
    """Exact rational AP (Fraction). Returns None if `deadline` (time.monotonic value) passes first."""
    num, den = ap_terms(tp, fp)
    acc = Fraction(0)
    for i, (a, b) in enumerate(zip(num.tolist(), den.tolist())):
        if deadline is not None and i % 256 == 0 and time.monotonic() > deadline:
            return None
        acc += Fraction(a, b)
    return acc / n1


def tpr_at_fpr(tp, fp, n1, n0, alpha):
    """Maximum attainable TPR with FP <= floor(alpha*n0) over whole tie groups (k = 0 included)."""
    budget = (alpha.numerator * n0) // alpha.denominator
    j = int(np.searchsorted(fp, budget, side="right")) - 1            # fp is non-decreasing
    TP, FP = (int(tp[j]), int(fp[j])) if j >= 0 else (0, 0)
    nxt = j + 1
    has_next = nxt < len(fp)
    gain = int(tp[nxt] - TP) if has_next else None
    return {"alpha": f"{alpha.numerator}/{alpha.denominator}", "fp_budget": budget, "TP": TP, "FP": FP,
            "tpr": Fraction(TP, n1), "achieved_fpr": Fraction(FP, n0),
            "tpr_resolution": Fraction(1, n1), "fpr_resolution": Fraction(1, n0),
            "low_resolution": budget < LOW_RESOLUTION_FP_BUDGET,
            # the next whole tie group would exceed the FP budget although budget remains and it contains malware rows
            "tie_blocked": bool(has_next and gain > 0 and FP < budget),
            "next_group_fp": int(fp[nxt] - FP) if has_next else None, "next_group_tp": gain}


def arm_metrics(s_desc, l_desc):
    """All WP5 quantities for one descending-sorted arm. Undefined (None) metrics when a class is absent."""
    s_desc, l_desc = np.asarray(s_desc), np.asarray(l_desc)
    _check_arm(s_desc, l_desc)
    n = int(len(s_desc))
    n1 = int(l_desc.sum())
    n0 = n - n1
    out = {"n": n, "malware": n1, "benign": n0, "prevalence": Fraction(n1, n) if n else None}
    if n1 == 0 or n0 == 0:
        out.update({"AP": None, "tie_groups": None, "positive_groups": None, **{k: None for k, _ in ALPHAS}})
        return out, None
    tp, fp = grouped_counts(s_desc, l_desc)
    if int(tp[-1]) != n1 or int(fp[-1]) != n0:
        raise WP5ContractError("cumulative counts do not end at (n1, n0)")
    out["AP"] = ap_double(tp, fp, n1)
    out["tie_groups"] = int(len(tp))
    out["positive_groups"] = int(np.count_nonzero(np.diff(tp, prepend=np.int64(0)) > 0))
    for k, a in ALPHAS:
        out[k] = tpr_at_fpr(tp, fp, n1, n0, a)
    return out, (tp, fp)


def unit_metrics(scores_full, labels_full, unseen_mask_full):
    """Full arm and Unseen arm of one unit, reusing ONE descending sort: the Unseen arm is the order-preserving
    subsequence of the sorted Full arm selected by the mask."""
    scores_full, labels_full = np.asarray(scores_full), np.asarray(labels_full)
    m = np.asarray(unseen_mask_full, dtype=bool)
    if not (len(scores_full) == len(labels_full) == len(m)):
        raise WP5ContractError("unit arrays differ in length")
    o = descending_order(scores_full)
    sF, lF, mF = scores_full[o], labels_full[o], m[o]
    F, cF = arm_metrics(sF, lF)
    U, cU = arm_metrics(sF[mF], lF[mF])
    return {"F": F, "U": U}, {"F": cF, "U": cU}
