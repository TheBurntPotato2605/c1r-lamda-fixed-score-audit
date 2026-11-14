"""C1-R G4 data layer: verified inputs, frozen P-A cohorts, I/O-guarded feature streaming, TRAIN in frozen order.

Rules reused verbatim from the locked G2 module (verify_inputs, projected_read, ym_status, cohort_of, REV) and
the locked G3 core (TracingFile, _chunk_ranges). Columns read: `label`, `year_month`, `feat_*` only.
`hash`, `vt_count` and `family` chunks are never touched (any touch is an IntegrityStop).
Feature values are kept exactly as released (int8, domain {0,1,2}; G3-DEV-01). No binarizing or clipping.
"""
import collections
import hashlib
import os

import numpy as np

import g4_common as K

FORBIDDEN_COLS = ("hash", "vt_count", "family")
META_COLS = ("label", "year_month")


class DataSpec:
    def __init__(self, data_dir, g1_run, g2_module, g3core_module, keys_csv, nf=K.NF, total=K.TOTAL_ROWS,
                 membership_digest=K.MEMBERSHIP_DIGEST, keys_sha=K.KEYS_TABLE_SHA256, set_sizes=None,
                 g2_sha=K.G2_ANALYSIS_SHA256, g3_sha=K.G3CORE_SHA256, batch_rows=8192):
        self.data_dir, self.g1_run, self.keys_csv = data_dir, g1_run, keys_csv
        self.g2_module, self.g3core_module, self.g2_sha, self.g3_sha = g2_module, g3core_module, g2_sha, g3_sha
        self.nf, self.total, self.membership_digest, self.keys_sha = nf, total, membership_digest, keys_sha
        self.set_sizes = dict(K.SET_SIZES) if set_sizes is None else set_sizes
        self.batch_rows = batch_rows

    @classmethod
    def real(cls):
        p = K.REAL_PATHS
        return cls(p["data_dir"], p["g1_run"], p["g2_module"], p["g3core_module"], p["keys_csv"])

    def to_dict(self):
        return {"data_dir": self.data_dir, "g1_run": self.g1_run, "g2_module": self.g2_module,
                "g3core_module": self.g3core_module, "keys_csv": self.keys_csv, "nf": self.nf, "total": self.total,
                "membership_digest": self.membership_digest, "keys_sha": self.keys_sha, "set_sizes": self.set_sizes,
                "g2_sha": self.g2_sha, "g3_sha": self.g3_sha, "batch_rows": self.batch_rows}


class Dataset:
    """Verified, cohort-assigned view of the 24 files. Row identity: canonical gid (Y asc, train<test, r_phys)."""

    def __init__(self, spec):
        for p in (spec.data_dir, spec.g1_run, spec.keys_csv, spec.g2_module, spec.g3core_module):
            if K.forbidden_path(p):
                raise K.IntegrityStop(f"forbidden path refused: {p}")
        import g4_access                      # V4 (finding D): real inputs only through the child-side G4 gate
        g4_access.assert_dataset_access(spec.to_dict())          # path/fingerprint checks; no protected file opened
        self.spec = spec
        self.G2 = K.load_locked_module("g2_analysis_locked", spec.g2_module, spec.g2_sha)
        self.G3 = K.load_locked_module("g3core_locked", spec.g3core_module, spec.g3_sha)
        self.nf = spec.nf
        self.trace = []
        try:
            self.files, total = self.G2.verify_inputs(spec.data_dir, spec.g1_run, spec.nf, spec.total)
        except Exception as e:  # G2's own IntegrityStop or any I/O error
            raise K.IntegrityStop(f"input verification failed: {e}") from e
        if total != spec.total:
            raise K.IntegrityStop("total rows differ")
        self._cohorts()
        self._verify_keys_table()

    # ------------------------------------------------------------ cohorts (Manifest §3.6; G2/G3 digest)
    def _cohorts(self):
        G = self.G2
        labels, yms, status, Ys, Ss, rphys = [], [], [], [], [], []
        for f in self.files:
            try:
                t, _ = G.projected_read(f["path"], f["size"], f["footer_len"], META_COLS)
            except Exception as e:
                raise K.IntegrityStop(f"metadata read failed: {e}") from e
            labs, ym = t.column("label").to_pylist(), t.column("year_month").to_pylist()
            for r, (lab, v) in enumerate(zip(labs, ym)):
                st, _ = G.ym_status(v, f["Y"])
                if lab not in (0, 1) or st == "INVALID" or (f["Y"] in (2013, 2014) and st != "VALID"):
                    raise K.IntegrityStop("metadata violates the G2 contract")
                labels.append(lab); yms.append(v); status.append(st)
                Ys.append(f["Y"]); Ss.append(f["S"]); rphys.append(r)
        n = len(labels)
        mstar = {Y: max(yms[i] for i in range(n) if Ys[i] == Y and status[i] == "VALID") for Y in (2013, 2014)}
        units = []
        for i in range(n):
            y = Ys[i]
            if y in (2013, 2014):
                units.append(f"IID_{y}" if yms[i] == mstar[y] else "TRAIN")
            else:
                units.append(("NEAR_" if y < 2018 else "FAR_") + str(y))
        h = hashlib.sha256()
        for i in range(n):
            a = units[i]
            h.update(f"{i}|{G.REV}/{Ys[i]}/{Ss[i]}/{rphys[i]}|{G.cohort_of(a)}|{a}|{labels[i]}|"
                     f"{'' if yms[i] is None else yms[i]}|{status[i]}\n".encode())
        self.membership_digest = h.hexdigest()
        if self.spec.membership_digest is not None and self.membership_digest != self.spec.membership_digest:
            raise K.IntegrityStop("membership digest differs from the frozen G2/G3 digest")
        sizes = dict(collections.Counter(units))
        if sizes != self.spec.set_sizes:
            raise K.IntegrityStop(f"cohort sizes differ from frozen sizes: {sizes}")
        self.mstar = mstar
        self.labels = np.asarray(labels, dtype=np.int8)
        self.units = np.asarray(units, dtype=object)
        self.yms = yms
        self.is_train = self.units == "TRAIN"
        self.eval_gids = np.flatnonzero(~self.is_train).astype(np.int64)
        self.train_gids_frozen = np.asarray(sorted(np.flatnonzero(self.is_train).tolist(),
                                                   key=lambda g: (yms[g], g)), dtype=np.int64)  # (ym, gid)

    def _verify_keys_table(self):
        if K.forbidden_path(self.spec.keys_csv) or not os.path.isfile(self.spec.keys_csv):
            raise K.IntegrityStop("sealed keys table missing")
        self.keys_sha = K.sha256_file(self.spec.keys_csv)
        if self.spec.keys_sha is not None and self.keys_sha != self.spec.keys_sha:
            raise K.IntegrityStop("sealed keys/seen table hash differs from G3")

    # ------------------------------------------------------------ I/O-guarded feature batches
    def iter_file_batches(self, f, extra_cols=()):
        """Yield (labels, X int8 [m, nf]) for one file in physical row order. extra_cols exists only so
        tests can prove that requesting a forbidden column triggers the guard."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        C, nf = self.G3, self.nf
        feats = [f"feat_{k}" for k in range(nf)]
        cols = list(META_COLS) + list(extra_cols) + feats
        size, flen = f["size"], f["footer_len"]
        tf = C.TracingFile(f["path"])
        try:
            tf.seek(size - 8 - flen)
            footer = tf.read(flen + 8)
            if len(footer) != flen + 8 or footer[-4:] != b"PAR1":
                raise K.IntegrityStop(f"{f['rel']}: footer read failed")
            md = pq.read_metadata(pa.BufferReader(b"PAR1" + footer))
            allowed = [(size - 8 - flen, size)] + C._chunk_ranges(md, set(META_COLS) | set(feats))
            forbidden = C._chunk_ranges(md, set(FORBIDDEN_COLS))
            pf = pq.ParquetFile(pa.PythonFile(tf, mode="r"), metadata=md, pre_buffer=False)
            for rb in pf.iter_batches(batch_size=self.spec.batch_rows, columns=cols, use_threads=False,
                                      use_pandas_metadata=False):
                if rb.schema.names != cols:
                    raise K.IntegrityStop("unexpected batch columns")
                m = rb.num_rows
                labs = np.asarray(rb.column(0).to_pylist(), dtype=np.int8)
                X = np.empty((m, nf), dtype=np.int8)
                base = 2 + len(extra_cols)
                for k in range(nf):
                    a = rb.column(base + k)
                    if a.type != pa.int8() or a.null_count:
                        raise K.IntegrityStop(f"{f['rel']}: feature column type/null violation")
                    X[:, k] = a.to_numpy(zero_copy_only=True)
                if X.size and (X.min() < 0 or X.max() > 2):
                    raise K.IntegrityStop(f"{f['rel']}: feature value outside {{0,1,2}} (G3-DEV-01)")
                yield labs, X
        finally:
            tf.close()
        bad = [(a, z) for a, z in tf.reads if not any(lo <= a and z <= hi for lo, hi in allowed)]
        touch = [(a, z) for a, z in tf.reads for lo, hi in forbidden if a < hi and z > lo]
        if bad or touch:
            raise K.IntegrityStop(f"{f['rel']}: {len(bad)} unauthorized read(s), {len(touch)} touching "
                                  f"hash/vt_count/family")
        self.trace.append({"file": f["rel"], "reads": len(tf.reads),
                           "bytes": sum(z - a for a, z in tf.reads), "unauthorized": 0, "touch_forbidden": 0})

    def _file_offsets(self):
        off, out = 0, []
        for f in self.files:
            out.append(off)
            off += f["rows"]
        return out

    # ------------------------------------------------------------ TRAIN in frozen order
    def load_train(self):
        """Dense int8 TRAIN matrix in the frozen (ym, gid) order, labels, gids and a fingerprint.
        STOP if any TRAIN cell equals 2 (G3 found none) or if label alignment fails."""
        order = self.train_gids_frozen
        n = len(order)
        pos = np.full(len(self.labels), -1, dtype=np.int64)
        pos[order] = np.arange(n, dtype=np.int64)
        X = np.empty((n, self.nf), dtype=np.int8)
        filled = 0
        for f, off in zip(self.files, self._file_offsets()):
            if f["Y"] not in (2013, 2014):
                continue
            g0 = off
            for labs, Xb in self.iter_file_batches(f):
                m = len(labs)
                gids = np.arange(g0, g0 + m, dtype=np.int64)
                if not np.array_equal(labs, self.labels[gids]):
                    raise K.IntegrityStop("row alignment check failed (label mismatch)")
                sel = self.is_train[gids]
                if sel.any():
                    rows = Xb[sel]
                    if (rows == 2).any():
                        raise K.IntegrityStop("TRAIN contains feature value 2 (contradicts G3; STOP)")
                    X[pos[gids[sel]]] = rows
                    filled += int(sel.sum())
                g0 += m
        if filled != n:
            raise K.IntegrityStop(f"TRAIN rows loaded {filled} != {n}")
        y = self.labels[order].astype(np.int32)
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(X).tobytes())
        h.update(y.astype("<i4").tobytes())
        h.update(order.astype("<i8").tobytes())
        return X, y, order, h.hexdigest()

    # ------------------------------------------------------------ evaluation universe stream
    def iter_eval(self):
        """Yield (gids int64, X int8) for evaluation rows (all 12 units) in canonical gid order."""
        seen = 0
        for f, off in zip(self.files, self._file_offsets()):
            g0 = off
            for labs, Xb in self.iter_file_batches(f):
                m = len(labs)
                gids = np.arange(g0, g0 + m, dtype=np.int64)
                if not np.array_equal(labs, self.labels[gids]):
                    raise K.IntegrityStop("row alignment check failed (label mismatch)")
                sel = ~self.is_train[gids]
                if sel.any():
                    seen += int(sel.sum())
                    yield gids[sel], Xb[sel]
                g0 += m
        if seen != len(self.eval_gids):
            raise K.IntegrityStop("evaluation stream row count differs")
