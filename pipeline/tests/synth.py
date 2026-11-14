"""Synthetic LAMDA-shaped dataset for the end-to-end test (NO LAMDA data): 24 Parquet files with the exact G2 schema,
a G1-style inventory, LOCAL_SHA256SUMS.txt and an INDEPENDENT keys/seen table built by exact vector comparison
(the pipeline recomputes the table with the frozen G3 core and must match it byte for byte)."""
import collections
import hashlib
import json
import os

import numpy as np

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "public"), os.path.join(os.path.dirname(HERE), "frozen", "src")]
import g4_common as K  # noqa: E402

G2_PATH = os.path.join(os.path.dirname(HERE), "frozen", "src", "g2_analysis.py")

YEARS = (2013, 2014) + tuple(range(2016, 2026))
MONTHS = {2013: ["2013-06", "2013-09", "2013-10", "2013-11", "2013-12"],
          2014: [f"2014-0{m}" for m in range(1, 9)]}


def _months(y):
    return MONTHS.get(y, [f"{y}-{m:02d}" for m in range(1, 13)])


def make_dataset(root, nf=24, seed=0, train_value2=False, eval_value2=True, n_train_year=(100, 25),
                 n_other=(40, 10)):
    import pyarrow as pa
    import pyarrow.parquet as pq
    G2 = K.load_locked_module("g2_synth", G2_PATH, K.sha256_file(G2_PATH))
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "Baseline")
    g1 = os.path.join(root, "g1_run")
    os.makedirs(g1, exist_ok=True)
    names, types = G2.expected_schema(nf)
    rows = []   # (Y, S, ym, label, vec bytes)
    train_vecs = []
    for y in YEARS:
        for s in ("train", "test"):
            n = (n_train_year if y in (2013, 2014) else n_other)[0 if s == "train" else 1]
            ms = _months(y)
            for r in range(n):
                ym = ms[r % len(ms)] if r < len(ms) else ms[int(rng.integers(len(ms)))]
                lab = int(r % 2) if r < 4 else int(rng.integers(2))
                v = (rng.random(nf) < 0.25).astype(np.int8)
                v[0] = lab if rng.random() < 0.8 else 1 - lab      # signal
                rows.append([y, s, ym, lab, v])
    # cohort assignment (P-A) to know TRAIN rows
    mstar = {y: max(r[2] for r in rows if r[0] == y) for y in (2013, 2014)}
    is_train = [r[0] in (2013, 2014) and r[2] != mstar[r[0]] for r in rows]
    tr_idx = [i for i, t in enumerate(is_train) if t]
    ev_idx = [i for i, t in enumerate(is_train) if not t]
    # some evaluation rows copy a TRAIN vector (seen); keep >= 1 unseen of each class per unit
    for i in ev_idx:
        if rng.random() < 0.25:
            rows[i][4] = rows[tr_idx[int(rng.integers(len(tr_idx)))]][4].copy()
    if eval_value2:
        far = [i for i in ev_idx if rows[i][0] >= 2018]
        for i in far[:2]:
            rows[i][4] = rows[i][4].copy()
            rows[i][4][nf - 1] = 2
    if train_value2:
        rows[tr_idx[3]][4] = rows[tr_idx[3]][4].copy()
        rows[tr_idx[3]][4][1] = 2
    # write files
    inv = {"revision": G2.REV, "verdict": "PASS", "files": []}
    local = []
    gid = 0
    per_file = collections.OrderedDict()
    for i, r in enumerate(rows):
        per_file.setdefault((r[0], r[1]), []).append(i)
    for (y, s), idx in per_file.items():
        d = os.path.join(base, str(y))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{y}_{s}.parquet")
        cols = {"hash": pa.array([hashlib.sha256(f"{y}{s}{j}".encode()).hexdigest() for j in idx], pa.string()),
                "label": pa.array([rows[j][3] for j in idx], pa.int64()),
                "family": pa.array(["benign" if rows[j][3] == 0 else "famx" for j in idx], pa.string()),
                "vt_count": pa.array([float(rows[j][3] * 5) for j in idx], pa.float64()),
                "year_month": pa.array([rows[j][2] for j in idx], pa.string())}
        M = np.stack([rows[j][4] for j in idx])
        for k in range(nf):
            cols[f"feat_{k}"] = pa.array(M[:, k], pa.int8())
        t = pa.table(cols, schema=pa.schema(list(zip(names, types))))
        pq.write_table(t, p)
        size = os.path.getsize(p)
        with open(p, "rb") as fh:
            data = fh.read()
        flen = int.from_bytes(data[-8:-4], "little")
        sha = hashlib.sha256(data).hexdigest()
        rel = f"Baseline/{y}/{y}_{s}.parquet"
        inv["files"].append({"path": rel, "size": size, "footer_len": flen,
                             "raw_footer_sha256": hashlib.sha256(data[size - 8 - flen:size - 8]).hexdigest(),
                             "head": {"provenance": {"x-linked-etag (server-provided; algorithm not asserted)": f'"{sha}"'}},
                             "decode": {"summary": {"num_rows": len(idx)}}})
        local.append(f"{sha}  {y}/{y}_{s}.parquet")
        gid += len(idx)
    with open(os.path.join(g1, "inventory.json"), "w", encoding="utf-8") as fh:
        json.dump(inv, fh)
    with open(os.path.join(base, "LOCAL_SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(local) + "\n")
    with open(os.path.join(base, "SYNTHETIC_DATA.json"), "w", encoding="utf-8") as fh:   # V4 access-control marker
        json.dump({"synthetic": True, "lamda_data": False, "generator": "tests/synth.py"}, fh)
    # canonical gid order = per_file order (Y asc, train<test, r asc) -> rows order already matches
    order = [i for idx in per_file.values() for i in idx]
    units = []
    for i in order:
        y = rows[i][0]
        if y in (2013, 2014):
            units.append("TRAIN" if is_train[i] else f"IID_{y}")
        else:
            units.append(("NEAR_" if y < 2018 else "FAR_") + str(y))
    vecs = [rows[i][4].tobytes() for i in order]
    labels = [rows[i][3] for i in order]
    ktrain = {vecs[g] for g, u in enumerate(units) if u == "TRAIN"}
    first = {}
    keys_csv = os.path.join(root, "keys_seen.csv")
    with open(keys_csv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("gid,kid,vector_sha256,all_zero,seen\n")
        for g, v in enumerate(vecs):
            kid = first.setdefault(v, g)
            seen = "TRAIN" if units[g] == "TRAIN" else ("1" if v in ktrain else "0")
            fh.write(f"{g},{kid},{hashlib.sha256(v).hexdigest()},{int(v == bytes(nf))},{seen}\n")
    sizes = dict(collections.Counter(units))
    spec = {"data_dir": base, "g1_run": g1, "keys_csv": keys_csv, "nf": nf, "total": len(order),
            "membership_digest": None, "keys_sha": K.sha256_file(keys_csv), "set_sizes": sizes, "batch_rows": 64}
    return {"spec": spec, "units": units, "labels": labels, "vecs": vecs, "mstar": mstar}
