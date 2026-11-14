"""C1-R G3 core: I/O-traced streaming of feature batches, binary validation, exact 4,561-byte keys.

Key semantics (Manifest §4): key(i) = bytes(int8 feat_0 .. feat_{nf-1}) in ascending k, no metadata.
For validated binary values each byte is 0x00 or 0x01.
"""
import ctypes
import hashlib
import io
import os

import pyarrow as pa
import pyarrow.parquet as pq

FORBIDDEN_COLS = ("hash", "vt_count")
META_COLS = ("label", "family", "year_month")
BIN2ASC = bytes.maketrans(b"\x00\x01", b"01")
ASC2BIN = bytes.maketrans(b"01", b"\x00\x01")


class IntegrityStop(Exception):
    pass


def feat_names(nf):
    return [f"feat_{k}" for k in range(nf)]


ALLOWED_VALUES = (0, 1, 2)          # G3-DEV-01: released int8 domain accepted as-is
ALLOWED_BYTES = bytes(ALLOWED_VALUES)
LOW_PLANE = bytes.maketrans(b"\x00\x01\x02", b"\x00\x01\x00")   # bit 0 of the value
HIGH_PLANE = bytes.maketrans(b"\x00\x01\x02", b"\x00\x00\x01")  # bit 1 of the value


def plane_len(nf):
    return (nf + 7) // 8


def packed_len(nf):
    return 2 * plane_len(nf)


def _pack_bits(bits01, nf):
    return int(bits01.translate(BIN2ASC), 2).to_bytes(plane_len(nf), "big")


def _unpack_bits(p, nf):
    return format(int.from_bytes(p, "big"), f"0{nf}b").encode("ascii").translate(ASC2BIN)


def pack(key, nf):
    """Two-bit-plane packing of a key with values in {0,1,2} (storage only; exact round trip)."""
    if len(key) != nf or key.translate(None, ALLOWED_BYTES):
        raise IntegrityStop("pack: key outside the validated domain")
    return _pack_bits(key.translate(LOW_PLANE), nf) + _pack_bits(key.translate(HIGH_PLANE), nf)


def unpack(p, nf):
    pl = plane_len(nf)
    low, high = _unpack_bits(p[:pl], nf), _unpack_bits(p[pl:], nf)
    # per-byte value = low + 2*high (each byte <= 2, so no carry between bytes)
    return (int.from_bytes(low, "big") + 2 * int.from_bytes(high, "big")).to_bytes(nf, "big")


def sha256_digest(key):
    return hashlib.sha256(key).digest()


class TracingFile(io.RawIOBase):
    def __init__(self, path):
        self.f = open(path, "rb")
        self.reads = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        return self.f.seek(off, whence)

    def tell(self):
        return self.f.tell()

    def readinto(self, b):
        pos = self.f.tell()
        n = self.f.readinto(b)
        if n:
            self.reads.append((pos, pos + n))
        return n

    def close(self):
        self.f.close()
        super().close()


def _chunk_ranges(md, cols):
    out = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        for j in range(rg.num_columns):
            c = rg.column(j)
            if c.path_in_schema in cols:
                b = c.dictionary_page_offset if c.has_dictionary_page else c.data_page_offset
                out.append((b, b + c.total_compressed_size))
    return out


def iter_batches(path, size, flen, nf, batch_rows):
    """Yield (labels, families, yms, feature_arrays) per batch. Footer is read explicitly (no
    speculative tail read); afterwards every read is checked to lie inside the footer or an
    authorized column chunk, and none may touch hash/vt_count chunks."""
    feats = feat_names(nf)
    cols = list(META_COLS) + feats
    tf = TracingFile(path)
    try:
        tf.seek(size - 8 - flen)
        footer = tf.read(flen + 8)
        if len(footer) != flen + 8 or footer[-4:] != b"PAR1":
            raise IntegrityStop(f"{os.path.basename(path)}: footer read failed")
        md = pq.read_metadata(pa.BufferReader(b"PAR1" + footer))
        allowed = [(size - 8 - flen, size)] + _chunk_ranges(md, set(cols))
        forbidden = _chunk_ranges(md, set(FORBIDDEN_COLS))
        pf = pq.ParquetFile(pa.PythonFile(tf, mode="r"), metadata=md, pre_buffer=False)
        for rb in pf.iter_batches(batch_size=batch_rows, columns=cols, use_threads=False,
                                  use_pandas_metadata=False):
            if rb.schema.names != cols:
                raise IntegrityStop("unexpected batch columns")
            yield (rb.column(0).to_pylist(), rb.column(1).to_pylist(), rb.column(2).to_pylist(),
                   [rb.column(3 + k) for k in range(nf)])
    finally:
        tf.close()
    bad = [(a, z) for a, z in tf.reads if not any(lo <= a and z <= hi for lo, hi in allowed)]
    touch = [(a, z) for a, z in tf.reads for lo, hi in forbidden if a < hi and z > lo]
    if bad or touch:
        raise IntegrityStop(f"{os.path.basename(path)}: {len(bad)} unauthorized read(s), "
                            f"{len(touch)} touching hash/vt_count")
    TRACE_LOG.append({"file": os.path.basename(path), "reads": len(tf.reads),
                      "bytes": sum(z - a for a, z in tf.reads), "unauthorized": 0, "touch_forbidden": 0})


TRACE_LOG = []


def column_views(arrays):
    """memoryviews of int8 value buffers (offset-aware). Caller must have checked nulls/type."""
    out = []
    for a in arrays:
        if a.type != pa.int8():
            raise IntegrityStop(f"feature column type {a.type}")
        buf = a.buffers()[1]
        out.append(memoryview(buf)[a.offset:a.offset + len(a)])
    return out


def keys_from_views(views, n, block):
    """Row-major exact keys from column-major int8 buffers: key_i[k] = col_k[i].
    Transposition by C-level strided slicing of a small column-major block (cache-local)."""
    nf = len(views)
    for r0 in range(0, n, block):
        r1 = min(n, r0 + block)
        w = r1 - r0
        cm = b"".join([v[r0:r1] for v in views])  # nf * w bytes, column-major
        for i in range(w):
            yield cm[i::w]


def peak_rss_bytes():
    """Peak working set of this process (Windows), else None."""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        if fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return int(c.PeakWorkingSetSize)
    except Exception:  # noqa: BLE001
        return None
    return None
