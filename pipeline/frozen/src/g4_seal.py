"""C1-R immutable score tables (Manifest §6 P1-P5).

Seal (P4): SHA-256 over rows in ascending gid order, each row = gid as int64 little-endian followed by the
score's native IEEE-754 little-endian bytes (float64 for SVM/LightGBM, float32 for XGBoost/MLP).
Arms (P5) are boolean masks over ONE sealed table; the metric code re-verifies the seal before use.
"""
import hashlib

import numpy as np

import g4_common as K

SCORE_DTYPES = {"svm": np.float64, "lgbm": np.float64, "xgb": np.float32, "mlp": np.float32}


def _records(gids, scores):
    dt = np.dtype([("g", "<i8"), ("s", np.dtype(scores.dtype).newbyteorder("<"))])
    rec = np.empty(len(gids), dtype=dt)
    rec["g"] = gids
    rec["s"] = scores
    return rec


def seal_digest(gids, scores):
    return hashlib.sha256(_records(gids, scores).tobytes()).hexdigest()


class ScoreAccumulator:
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        self.g, self.s = [], []

    def add(self, gids, scores):
        scores = np.asarray(scores)
        if scores.dtype != self.dtype:
            raise K.ContractFailure(f"score dtype {scores.dtype} != frozen {self.dtype}")
        if scores.ndim != 1 or len(scores) != len(gids):
            raise K.ContractFailure("score vector shape mismatch")
        self.g.append(np.asarray(gids, dtype=np.int64))
        self.s.append(scores)

    def finalize(self, expected_gids):
        """Return (gids, scores) sorted by gid after the P3 checks; never drops or deduplicates rows."""
        g = np.concatenate(self.g) if self.g else np.empty(0, np.int64)
        s = np.concatenate(self.s) if self.s else np.empty(0, self.dtype)
        if not np.all(np.isfinite(s)):
            raise K.TechnicalFailure("NaN or infinite score (P3; never imputed)")
        if s.size and (s.min() < 0 or s.max() > 1):
            raise K.ContractFailure("score outside [0, 1] (probability contract)")
        o = np.argsort(g, kind="stable")
        g, s = g[o], s[o]
        if len(g) != len(expected_gids):
            raise K.ContractFailure(f"score rows {len(g)} != evaluation universe {len(expected_gids)}")
        if len(g) > 1 and np.any(g[1:] == g[:-1]):
            raise K.ContractFailure("duplicate gid in score table")
        if not np.array_equal(g, np.sort(np.asarray(expected_gids, dtype=np.int64))):
            raise K.ContractFailure("score gids differ from the evaluation universe")
        return g, s


def write_table(path, gids, scores):
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table({"gid": pa.array(gids, type=pa.int64()),
                  "score": pa.array(scores, type=pa.from_numpy_dtype(scores.dtype))})
    pq.write_table(t, path, compression="zstd", write_statistics=False)


def read_table(path, expected_digest=None):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    g = t.column("gid").to_numpy()
    s = t.column("score").to_numpy()
    if expected_digest is not None and seal_digest(g, s) != expected_digest:
        raise K.ContractFailure(f"sealed score table digest mismatch: {path}")
    return g, s


def assert_passes_identical(d1, d2):
    if d1 != d2:
        raise K.ContractFailure("P2: second inference pass is not bit-identical to the first")
