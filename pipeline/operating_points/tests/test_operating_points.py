"""Synthetic tests for the operating-point metric core and the public driver (no LAMDA data).
Run (pinned environment): python -I -B pipeline/operating_points/tests/test_operating_points.py
"""
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "frozen", "src"))
import metric_core as C          # noqa: E402
import operating_points as D  # noqa: E402
import metrics_g5 as MG       # noqa: E402  (frozen, read-only)
from g4_seal import seal_digest  # noqa: E402


# ------------------------------------------------------------------ brute-force references (independent of the core)
def brute(scores, labels, alphas=C.ALPHAS):
    s = [float(x) for x in scores]
    lab = [int(x) for x in labels]
    n1 = sum(lab)
    n0 = len(lab) - n1
    th = sorted(set(s), reverse=True)
    pts = [(0, 0)]
    for t in th:                                   # threshold "score >= t" takes whole tie groups
        tp = sum(1 for x, y in zip(s, lab) if x >= t and y == 1)
        fp = sum(1 for x, y in zip(s, lab) if x >= t and y == 0)
        pts.append((tp, fp))
    ap = Fraction(0)
    for (tp0, _), (tp, fp) in zip(pts, pts[1:]):
        if tp > tp0:
            ap += Fraction(tp - tp0, n1) * Fraction(tp, tp + fp)
    out = {"AP": ap}
    for k, a in alphas:
        budget = (a.numerator * n0) // a.denominator
        best = max((tp, fp) for tp, fp in pts if fp <= budget)
        feas = [(tp, fp) for tp, fp in pts if fp <= budget]
        tpmax = max(tp for tp, _ in feas)
        fp_at = max(fp for tp, fp in feas if tp == tpmax)
        out[k] = (tpmax, fp_at, budget)
    return out


def sorted_arm(scores, labels):
    o = C.descending_order(scores)
    return scores[o], labels[o]


def synth(rng, n, prev, levels):
    lab = (rng.random(n) < prev).astype(np.int8)
    if lab.sum() == 0:
        lab[0] = 1
    if lab.sum() == n:
        lab[0] = 0
    raw = rng.normal(size=n) + 1.2 * lab
    s = np.round(raw * levels) / levels                # heavy ties
    return s.astype(np.float32), lab


class CoreTests(unittest.TestCase):
    def test_against_bruteforce_with_ties(self):
        rng = np.random.default_rng(7)
        for trial in range(60):
            n = int(rng.integers(5, 400))
            s, lab = synth(rng, n, rng.choice([0.02, 0.3, 0.5, 0.9]), rng.choice([1, 3, 10, 1000]))
            arm, (tp, fp) = C.arm_metrics(*sorted_arm(s, lab))
            ref = brute(s, lab)
            self.assertLessEqual(abs(Fraction(arm["AP"]) - ref["AP"]), Fraction(1, 10 ** 12))
            self.assertEqual(C.ap_exact(tp, fp, arm["malware"]), ref["AP"])
            for k, _ in C.ALPHAS:
                self.assertEqual((arm[k]["TP"], arm[k]["FP"], arm[k]["fp_budget"]), ref[k], (trial, k))
                self.assertEqual(arm[k]["tpr"], Fraction(ref[k][0], arm["malware"]))

    def test_sklearn_definition_matches(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(11)
        for _ in range(30):
            s, lab = synth(rng, 2000, 0.1, 50)
            arm, _ = C.arm_metrics(*sorted_arm(s, lab))
            self.assertLessEqual(abs(arm["AP"] - average_precision_score(lab, s)), 1e-12)

    def test_all_tied_and_single_positive(self):
        s = np.zeros(10, np.float32)
        lab = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], np.int8)
        arm, _ = C.arm_metrics(*sorted_arm(s, lab))
        self.assertAlmostEqual(arm["AP"], 0.2, places=15)          # one group: precision = 2/10, recall jumps 0 -> 1
        self.assertEqual(arm["fpr_le_1pct"]["TP"], 0)               # whole group would need FP = 8 > budget 0
        self.assertTrue(arm["fpr_le_1pct"]["tie_blocked"] is False)  # budget 0 is exhausted (FP == budget), not tie-blocked
        s2 = np.array([0.9, 0.5, 0.5, 0.1], np.float32)
        l2 = np.array([1, 0, 0, 0], np.int8)
        arm2, _ = C.arm_metrics(*sorted_arm(s2, l2))
        self.assertEqual(arm2["AP"], 1.0)
        self.assertEqual(arm2["fpr_le_1pct"]["tpr"], Fraction(1, 1))

    def test_tie_blocked_flag(self):
        # 1000 benign; budget at 1% = 10. Scores: 5 benign high, then one tie group of 20 rows (10 malware, 10 benign).
        s = np.concatenate([np.full(5, 0.99), np.full(20, 0.8), np.linspace(0.1, 0.5, 988)]).astype(np.float32)
        lab = np.concatenate([np.zeros(5), np.r_[np.ones(10), np.zeros(10)], np.zeros(988)]).astype(np.int8)
        lab[-3:] = 1                                     # benign total = 5 + 10 + 985 = 1000 -> 1% budget = 10
        arm, _ = C.arm_metrics(*sorted_arm(s, lab))
        r = arm["fpr_le_1pct"]
        self.assertEqual((r["FP"], r["fp_budget"]), (5, 10))
        self.assertTrue(r["tie_blocked"])
        self.assertEqual((r["next_group_fp"], r["next_group_tp"]), (10, 10))

    def test_low_resolution_flag_and_undefined(self):
        rng = np.random.default_rng(3)
        s, lab = synth(rng, 300, 0.5, 1000)
        arm, _ = C.arm_metrics(*sorted_arm(s, lab))
        self.assertTrue(arm["fpr_le_0p1pct"]["low_resolution"])       # floor(0.001 * ~150) = 0 < 10
        z = np.zeros(5, np.int8)
        a0, c0 = C.arm_metrics(np.sort(np.arange(5, dtype=np.float32))[::-1], z)
        self.assertIsNone(a0["AP"])
        self.assertIsNone(c0)

    def test_unseen_reuses_full_sort(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            s, lab = synth(rng, 1500, 0.4, 20)
            m = rng.random(1500) < 0.7
            res, _ = C.unit_metrics(s, lab, m)
            direct, _ = C.arm_metrics(*sorted_arm(s[m], lab[m]))
            for k in ("AP", "malware", "benign", "tie_groups", "positive_groups"):
                self.assertEqual(res["U"][k], direct[k])
            for k, _a in C.ALPHAS:
                self.assertEqual(res["U"][k], direct[k])

    def test_contracts(self):
        with self.assertRaises(C.WP5ContractError):
            C.arm_metrics(np.array([0.1, 0.9], np.float32), np.array([0, 1], np.int8))      # not descending
        with self.assertRaises(C.WP5ContractError):
            C.arm_metrics(np.array([np.nan, 0.1], np.float32), np.array([0, 1], np.int8))

    def test_exact_deadline(self):
        rng = np.random.default_rng(9)
        s, lab = synth(rng, 5000, 0.5, 10 ** 6)
        _, (tp, fp) = C.arm_metrics(*sorted_arm(s, lab))
        self.assertIsNone(C.ap_exact(tp, fp, int(lab.sum()), deadline=time.monotonic() - 1))


# ------------------------------------------------------------------ driver-level synthetic pipeline
def synthetic_world(rng, per_unit=400):
    units, labels, seen = [], [], []
    for u in MG.ALL_UNITS:
        units += [u] * per_unit
        prev = 0.01 if u == "FAR_2025" else 0.4
        lab = (rng.random(per_unit) < prev).astype(np.int8)
        lab[:2] = (1, 0)
        labels += lab.tolist()
        seen += (rng.random(per_unit) < (0.35 if u.startswith("IID") else 0.03)).astype(np.int8).tolist()
    n = len(units)
    gids = np.arange(10, 10 + n, dtype=np.int64)
    meta = MG.EvalMeta(gids, np.array(units, dtype=object), np.array(labels, np.int8), np.array(seen, np.int8))
    tables = []
    for i, (m, rs) in enumerate((("svm", [1, 2]), ("lgbm", [1]), ("mlp", [1]))):
        sc = (np.round((rng.normal(size=n) + 1.5 * meta.labels) * 8) / 8).astype(np.float32)
        tables.append({"sha": f"{i:064x}", "slots": [(m, r) for r in rs], "gids": gids.copy(), "scores": sc, "seal": seal_digest(gids, sc)})
    stored = {}
    for t in tables:
        for (m, r) in t["slots"]:
            for u in MG.ALL_UNITS:
                (fs, fl), (us, ul) = MG.arms(t["gids"], t["scores"], t["seal"], meta, u)
                stored[(m, r, u, "A_F")], stored[(m, r, u, "A_U")] = MG.auc_exact(fs, fl), MG.auc_exact(us, ul)
    counts = {}
    for u in MG.ALL_UNITS:
        F = meta.units == u
        counts[u] = {}
        for arm, mask in (("F", F), ("U", F & (meta.seen == 0))):
            n_, n1 = int(mask.sum()), int(meta.labels[mask].sum())
            counts[u][arm] = {"n": n_, "malware": n1, "benign": n_ - n1}
    return meta, tables, stored, counts


class DriverPipelineTests(unittest.TestCase):
    def setUp(self):
        self.meta, self.tables, self.stored, self.counts = synthetic_world(np.random.default_rng(21))
        self.sa = lambda m, r, u, k: self.stored[(m, r, u, k)]

    def test_clean_pipeline(self):
        per, chk = D.evaluate_tables(self.meta, self.tables, self.sa, self.counts, MG, max_groups=8000, large_limit_s=5)
        self.assertEqual(chk["auc_alignment_checks"], 2 * 12 * 4)          # 4 slots x 12 units x 2 arms
        self.assertEqual(chk["count_checks"], 24)
        self.assertLessEqual(chk["sklearn_ap_max_abs_dev"], 1e-12)
        self.assertGreater(chk["exact_subset"]["evaluated"], 0)
        slot_table = {f"{m}_r{r}": t["sha"] for t in self.tables for (m, r) in t["slots"]}
        self.assertEqual(per[slot_table["svm_r1"]], per[slot_table["svm_r2"]])

    def test_alignment_control_stops(self):
        k = ("lgbm", 1, "FAR_2019", "A_F")
        self.stored[k] = self.stored[k] + Fraction(1, 10 ** 9)
        with self.assertRaises(D.MetricStop):
            D.evaluate_tables(self.meta, self.tables, self.sa, self.counts, MG)

    def test_count_check_stops(self):
        self.counts["IID_2014"]["U"]["malware"] += 1
        with self.assertRaises(D.MetricStop):
            D.evaluate_tables(self.meta, self.tables, self.sa, self.counts, MG)

    def test_tampered_scores_stop(self):
        self.tables[1]["scores"] = self.tables[1]["scores"].copy()
        self.tables[1]["scores"][5] += np.float32(0.125)
        with self.assertRaises(D.MetricStop):
            D.evaluate_tables(self.meta, self.tables, self.sa, self.counts, MG)

    def test_misaligned_gids_stop(self):
        t = self.tables[2]
        g = t["gids"][::-1].copy()
        t["gids"], t["seal"] = g, seal_digest(g, t["scores"])
        with self.assertRaises(D.MetricStop):
            D.evaluate_tables(self.meta, self.tables, self.sa, self.counts, MG)


if __name__ == "__main__":
    unittest.main(verbosity=2)
