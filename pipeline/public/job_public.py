"""Public fit-and-score job: one (model, run) slot. Child process started by `c1r_e2e.py train`.

This is the scientific sequence of the original `g4_job.py` with the authorization gate, fault-injection hooks,
runtime-pin comparison, phase/ledger files and resource monitor removed (pipeline/PARITY.md lists each removal).
Unchanged, in the same order: seeding, adapter construction, verified inputs, TRAIN load in the frozen order,
fit on TRAIN only, resolved-configuration contract check, save, reload, two scoring passes over the evaluation
universe, immutable-score seal, bit-identity of the two passes.

Usage: python job_public.py job.json      (job.json is written by c1r_e2e.py)
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "frozen", "src")]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import g4_common as K  # noqa: E402

K.forbidden_path = lambda p: False      # the original guard protected an unrelated private project (PARITY.md)


def main(job_path):
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    out_dir, model, seed = job["out_dir"], job["model"], job["run"]
    R = {"state": "RUNNING", "model": model, "run": seed, "phase_seconds": {}}
    t = [time.time(), None]

    def phase(name):
        if t[1] is not None:
            R["phase_seconds"][t[1]] = round(time.time() - t[0], 2)
        t[0], t[1] = time.time(), name

    code = 0
    try:
        import random

        import numpy as np

        from g4_adapters import ADAPTERS
        from g4_data import DataSpec, Dataset
        from g4_seal import SCORE_DTYPES, ScoreAccumulator, assert_passes_identical, seal_digest, write_table
        R["python"] = sys.version.split()[0]
        random.seed(seed)
        np.random.seed(seed)
        adapter_cls = ADAPTERS[model]
        adapter = (adapter_cls(job["overrides"], job["container"]) if model == "xgb" else adapter_cls(job["overrides"]))
        phase("inputs")
        ds = Dataset(DataSpec(**job["spec"]))
        R["inputs"] = {"file_sha256": {f["rel"]: f["sha256"] for f in ds.files}, "membership_digest": ds.membership_digest,
                       "keys_table_sha256": ds.keys_sha, "m_star": ds.mstar, "eval_rows": int(len(ds.eval_gids))}
        phase("train_load")
        X, y, order, fp = ds.load_train()
        R["train"] = {"rows": int(len(y)), "malware": int(y.sum()), "fingerprint_sha256": fp}
        phase("fit")
        model_obj = adapter.fit(X, y, seed=seed)
        del X
        phase("contract")
        resolved = adapter.resolved(model_obj)
        with open(os.path.join(out_dir, "resolved_config.json"), "w", encoding="utf-8") as fh:
            json.dump(resolved, fh, indent=1, default=str)
        adapter.check_contract(resolved, seed)
        phase("save")
        art = adapter.save(model_obj, out_dir)
        R["model_artifacts_sha256"] = {os.path.basename(p): K.sha256_file(p) for p in art}
        del model_obj

        def score_pass():
            acc = ScoreAccumulator(SCORE_DTYPES[model])
            loaded = adapter.load(out_dir)                  # canonical scores come from the reloaded artifact
            for gids, Xb in ds.iter_eval():
                acc.add(gids, adapter.predict(loaded, Xb))
            return acc.finalize(ds.eval_gids)

        phase("score_pass1")
        g1, s1 = score_pass()
        d1 = seal_digest(g1, s1)
        write_table(os.path.join(out_dir, "scores.parquet"), g1, s1)
        phase("score_pass2")
        g2, s2 = score_pass()
        d2 = seal_digest(g2, s2)
        assert_passes_identical(d1, d2)
        with open(os.path.join(out_dir, "scores.sha256"), "w", encoding="utf-8") as fh:
            fh.write(d1 + "\n")
        R["scores"] = {"rows": int(len(g1)), "dtype": str(s1.dtype), "seal_sha256": d1, "pass2_sha256": d2,
                       "file_sha256": K.sha256_file(os.path.join(out_dir, "scores.parquet"))}
        phase("done")
        R["state"] = "SUCCEEDED"
    except K.G4Error as e:
        R["state"], R["error"], code = e.state, str(e), e.exit_code
    except Exception as e:  # noqa: BLE001
        R["state"], R["error"], code = "FAILED-TECHNICAL", f"{type(e).__name__}: {e}", 11
        R["traceback"] = traceback.format_exc()
    R["exit_code"], R["last_phase"] = code, t[1]
    with open(os.path.join(out_dir, "job_result.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
