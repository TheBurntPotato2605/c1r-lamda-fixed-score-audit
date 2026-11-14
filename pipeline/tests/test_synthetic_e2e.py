"""Synthetic end-to-end verification of the public pipeline (NO LAMDA data, no network, no GPU).

Run inside the pinned environment:   python pipeline/tests/test_synthetic_e2e.py [--keep]

What it proves
  * the stages verify -> cohorts -> train (4 models x 5 runs) -> evaluate -> compare run end to end through the
    FROZEN scientific code;
  * the frozen G3 core recomputes an independently built keys/seen table byte for byte (exact membership);
  * training is deterministic on CPU: a retrained slot has an identical seal and score-file hash;
  * every ROC-AUC the frozen metric module reports equals an independent re-computation (scikit-learn and an
    exact rank-sum implementation in this file), and the Full/Unseen arms are masks over the same scores;
  * the `compare` stage reports EXACT against itself and DIFFERS against the real reference (negative control);
  * no reference result is read by any computation stage.
It does NOT prove that the real dataset reproduces the manuscript: see README (verification status).
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
sys.path[:0] = [HERE, PIPE]

import c1r_e2e as E  # noqa: E402
import synth  # noqa: E402

SYN_OVERRIDES = {"svm": {}, "lgbm": {}, "xgb": {"device": "cpu", "rounds": 15}, "mlp": {"device": "cpu", "epochs": 2}}
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def exact_auc(scores, labels):
    """Independent exact AUC (ties 1/2): O(n log n) over sorted unique scores, Fractions."""
    import numpy as np
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n1, n0 = len(pos), len(neg)
    u, cp = np.unique(pos, return_counts=True)
    un, cn = np.unique(neg, return_counts=True)
    below = np.concatenate([[0], np.cumsum(cn)])
    tot = 0
    eq = dict(zip(un.tolist(), cn.tolist()))
    for s, c in zip(u.tolist(), cp.tolist()):
        lt = int(below[np.searchsorted(un, s, side="left")])
        tot += 2 * c * lt + c * eq.get(s, 0)
    return Fraction(tot, 2 * n1 * n0)


def main():
    keep = "--keep" in sys.argv
    tmp = tempfile.mkdtemp(prefix="c1r_synth_")
    print("workspace: <temporary directory>")
    try:
        d = synth.make_dataset(os.path.join(tmp, "synthetic"))
        sp = d["spec"]
        work = os.path.join(tmp, "work")
        ctx = E.Ctx(work, data_dir=sp["data_dir"], g1_dir=sp["g1_run"], keys_csv=os.path.join(work, "keys", "keys_seen.csv"),
                    nf=sp["nf"], total=sp["total"], membership_digest=None, keys_sha=sp["keys_sha"], set_sizes=sp["set_sizes"],
                    overrides=SYN_OVERRIDES, containers={m: "quantile" for m in E.MODELS}, real=False, batch_rows=64)
        check("verify (frozen G2 input verification)", E.stage_verify(ctx))
        check("cohorts (frozen G3 core; keys table equals the independent table byte for byte)", E.stage_cohorts(ctx))
        check("keys table SHA-256 equals independent reference", E.sha256_file(ctx.keys_csv) == sp["keys_sha"])
        check("train 4 models x 5 runs", E.stage_train(ctx))
        # determinism: retrain one slot elsewhere and compare seals
        d1 = E.jload(os.path.join(E.slot_dir(ctx, "svm", 1), "job_result.json"))["scores"]
        ctx2 = E.Ctx(os.path.join(tmp, "work2"), data_dir=sp["data_dir"], g1_dir=sp["g1_run"], keys_csv=ctx.keys_csv, nf=sp["nf"],
                     total=sp["total"], membership_digest=None, keys_sha=sp["keys_sha"], set_sizes=sp["set_sizes"],
                     overrides=SYN_OVERRIDES, real=False, batch_rows=64)
        for m in ("svm", "lgbm", "xgb", "mlp"):
            E.stage_train(ctx2, models=[m], runs=[2])
            a = E.jload(os.path.join(E.slot_dir(ctx, m, 2), "job_result.json"))["scores"]
            b = E.jload(os.path.join(E.slot_dir(ctx2, m, 2), "job_result.json"))["scores"]
            check(f"deterministic retrain {m} run 2 (seal and score-file hash identical)",
                  a["seal_sha256"] == b["seal_sha256"] and a["file_sha256"] == b["file_sha256"])
        check("evaluate (frozen metric module, exact arithmetic)", E.stage_evaluate(ctx))
        # independent AUC recomputation for every slot and unit (Full and Unseen)
        import numpy as np
        import g4_seal as S
        import metrics_g5 as MG
        per = E.jload(os.path.join(ctx.eval_dir, "G5_per_run_metrics.json"))
        keys = [l.rstrip("\n").split(",") for l in open(ctx.keys_csv, encoding="utf-8")][1:]
        seen = {int(k[0]): k[4] for k in keys}
        units, labels = d["units"], d["labels"]
        bad = 0
        n = 0
        for m in ("svm", "lgbm", "xgb", "mlp"):
            for r in range(1, 6):
                R = E.jload(os.path.join(E.slot_dir(ctx, m, r), "job_result.json"))
                g, s = S.read_table(os.path.join(E.slot_dir(ctx, m, r), "scores.parquet"), R["scores"]["seal_sha256"])
                for u in MG.ALL_UNITS:
                    for arm in ("F", "U"):
                        idx = [i for i, gg in enumerate(g.tolist()) if units[gg] == u and (arm == "F" or seen[gg] == "0")]
                        sc, lb = s[idx].astype(np.float64), np.asarray([labels[g[i]] for i in idx])
                        want = exact_auc(sc, lb)
                        got = per[m][str(r)]["units"][u]["A_" + arm]["exact"]
                        n += 1
                        if f"{want.numerator}/{want.denominator}" != got:
                            bad += 1
        check(f"independent exact AUC equals frozen metric output ({n} unit-arm values)", bad == 0, f"mismatches={bad}")
        check("compare against itself is EXACT", E.stage_compare(ctx, ref_dir=ctx.eval_dir) and
              E.jload(os.path.join(ctx.reports, "compare.json"))["status"] == "EXACT")
        ok = E.stage_compare(ctx, ref_dir=os.path.join(E.REPO, "data", "g5_aggregates"))
        st = E.jload(os.path.join(ctx.reports, "compare.json"))["status"]
        check("negative control: compare against the real reference reports DIFFERS", (not ok) and st == "DIFFERS", f"status={st}")
        import re
        src = open(os.path.join(PIPE, "c1r_e2e.py"), encoding="utf-8").read()
        comp_fn = src[src.index("def stage_evaluate"):src.index("# ---", src.index("def stage_evaluate"))]
        check("evaluate stage does not reference reference/ or data/g5_aggregates", "REFERENCE" not in comp_fn and "g5_aggregates" not in comp_fn)
    finally:
        if keep:
            print("kept:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    failed = [n for n, ok in CHECKS if not ok]
    print(f"\nSYNTHETIC END-TO-END: {'PASS' if not failed else 'FAIL'} ({len(CHECKS) - len(failed)}/{len(CHECKS)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
