"""C1-R G2 LOCAL — row-metadata verification and canonical P-A cohort construction.

Authority: owner approval "G2 LOCAL" (2026-11-05); frozen rules = Manifest v0.2.4 §3.3-§3.7, §7.3, §9 G2
and Protocol V2.2.1 §D/§I G2. No scientific rule is changed here.

Data access: ONLY columns hash, label, family, year_month, read by PyArrow column projection
(pre_buffer off, single-threaded) through an I/O tracer that STOPs if any byte outside the footer
or those four column chunks is read. feat_* and vt_count are never read or decoded.
Whole-file SHA-256 hashing (byte-level, no decoding) is used only for identity verification.

Outputs contain aggregates only. No hash value or other row-level value is written to any ordinary
report, table or log. A restricted gid-only duplicate listing (no hashes) is written under restricted/.

Usage: <pinned venv python> -I g2_analysis.py --data-dir ... --g1-run ... --review-dir ... --out-root ...
Exit: 0 PASS; 2 STOP (scientific/data); 3 STOP (integrity/pre-read refusal).
"""
import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import os
import platform
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "1.1.0"
REV = "ad9614bdd5556767f97ced2fce797c2f06408ebf"
YEARS = (2013, 2014) + tuple(range(2016, 2026))
SPLITS = ("train", "test")
READ_COLS = ("hash", "label", "family", "year_month")
EXPECTED_TOTAL = 1_008_381
PIN = {"python": "3.13.3", "pyarrow": "25.0.1"}
ANCHORS = {
    "C1-R_G0_APPROVAL_RECORD_2026-11-04.md": "abe279b46b038b77e6f34394b2964cd377eaf7271427d647adbbc1d0c444a764",
    "C1-R_G0_Execution_Manifest_CANDIDATE_v0.2.4_2026-11-04.md": "ac8fee5f014778533d12359d2052f985277d6a614f181c6c1fdad9e163721898",
    "C1-R_Protocol_V2.2.1_CANDIDATE_2026-11-04.md": "47fb8da6e856ef1fbdd9790fb337e7d985802bbad70b8a1b0f472623d1befaea",
    "G0_DECISION_PACKAGE.md": "91eb5537c1f6528bf92cde6bef55857a0fbb39aab3deffdc670adc0ede850984",
    "C1-R_G1_APPROVAL_RECORD_2026-11-04.md": "0d5f659a49d25557d56db493564f3d890079eb5b184d1a355b947020cde09c2c",
    "C1-R_G1_AUTHORIZATION_REQUEST_CANDIDATE_2026-11-04.md": "88d27c736fdeba42a49cbfb0de9cc42e025ebaff37811478f39bb1876fb34706",
    "C1-R_G1_HOST_ALLOWLIST_AMENDMENT_2026-11-05.md": "6e0d29c8bc4d8866ffe28bc9214dbda03eb29ef076e1bf6fccfacbdff0cd6b4a",
}
YM_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
UNITS = ("IID_2013", "IID_2014", "NEAR_2016", "NEAR_2017") + tuple(f"FAR_{y}" for y in range(2018, 2026))
PRIMARY_UNITS = ("IID_2013", "IID_2014") + tuple(f"FAR_{y}" for y in range(2018, 2026))
SETS = ("TRAIN",) + UNITS
LABEL_SEMANTICS = {0: "benign", 1: "malware"}  # Manifest §3.4: L1 official documentation (S6), L3 code (S5)
MISSING = "<MISSING>"
# Family values that embed a sample hash (AVClass "singleton:<sha256>") are row-level identifiers;
# they are counted but only ever exported as an aggregate placeholder.
SINGLETON = "<SINGLETON>"
HASH_BEARING = "<HASH-BEARING>"
SINGLETON_RE = re.compile(r"^singleton:[0-9a-fA-F]{64}$", re.IGNORECASE)
HEXRUN_RE = re.compile(r"[0-9a-fA-F]{32,}")


class IntegrityStop(Exception):
    """File identity / schema / access-scope failure: nothing scientific is computed."""


def expected_schema(nfeat):
    names = ["hash", "label", "family", "vt_count", "year_month"] + [f"feat_{i}" for i in range(nfeat)]
    types = [pa.string(), pa.int64(), pa.string(), pa.float64(), pa.string()] + [pa.int8()] * nfeat
    return names, types


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ I/O-traced projected read
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


def allowed_ranges(md, size, footer_len, cols):
    rs = [(size - 8 - footer_len, size)]  # footer + length + magic
    per_col = collections.defaultdict(list)
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        for j in range(rg.num_columns):
            c = rg.column(j)
            if c.path_in_schema in cols:
                b = c.dictionary_page_offset if c.has_dictionary_page else c.data_page_offset
                rs.append((b, b + c.total_compressed_size))
                per_col[c.path_in_schema].append((b, b + c.total_compressed_size))
    return rs, per_col


def check_trace(reads, ranges):
    bad = [(a, z) for a, z in reads if not any(lo <= a and z <= hi for lo, hi in ranges)]
    return bad


def projected_read(path, size, footer_len, cols):
    tf = TracingFile(path)
    try:
        # Read the exact footer span ourselves and decode metadata from it, so PyArrow performs no
        # speculative tail read; then only the projected column chunks are fetched by PyArrow.
        tf.seek(size - 8 - footer_len)
        footer = tf.read(footer_len + 8)
        if len(footer) != footer_len + 8 or footer[-4:] != b"PAR1":
            raise IntegrityStop(f"{os.path.basename(path)}: footer read failed")
        md = pq.read_metadata(pa.BufferReader(b"PAR1" + footer))
        ranges, per_col = allowed_ranges(md, size, footer_len, cols)
        pf = pq.ParquetFile(pa.PythonFile(tf, mode="r"), metadata=md, pre_buffer=False)
        t = pf.read(columns=list(cols), use_threads=False, use_pandas_metadata=False)
    finally:
        tf.close()
    bad = check_trace(tf.reads, ranges)
    if bad:
        raise IntegrityStop(f"{os.path.basename(path)}: {len(bad)} read(s) outside footer/authorized column chunks")
    data_reads = [(a, z) for a, z in tf.reads if a < size - 8 - footer_len]
    by_col = {c: sum(z - a for a, z in data_reads if any(lo <= a and z <= hi for lo, hi in per_col[c]))
              for c in cols}
    trace = {"reads": len(tf.reads), "bytes_read_total": sum(z - a for a, z in tf.reads),
             "bytes_read_footer_region": sum(z - a for a, z in tf.reads if a >= size - 8 - footer_len),
             "bytes_read_by_column": by_col, "authorized_chunk_bytes": {c: sum(z - a for a, z in per_col[c]) for c in cols},
             "reads_outside_authorized": 0, "read_spans": tf.reads}
    if t.column_names != list(cols):
        raise IntegrityStop(f"{os.path.basename(path)}: projected table columns {t.column_names}")
    return t, trace


# ------------------------------------------------------------------ stage 1: identity + schema
def verify_inputs(data_dir, g1_run, nfeat, expected_total, local_manifest_name="LOCAL_SHA256SUMS.txt"):
    inv = json.load(open(os.path.join(g1_run, "inventory.json"), encoding="utf-8"))
    if inv.get("revision") != REV or inv.get("verdict") != "PASS":
        raise IntegrityStop("G1 evidence is not a PASS at the pinned revision")
    g1 = {f["path"]: f for f in inv["files"]}
    local = {}
    lm = os.path.join(data_dir, local_manifest_name)
    with open(lm, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                h, p = line.split(None, 1)
                local["Baseline/" + p.strip().lstrip("*")] = h
    names, types = expected_schema(nfeat)
    files, total = [], 0
    for y in YEARS:
        for s in SPLITS:
            rel = f"Baseline/{y}/{y}_{s}.parquet"
            p = os.path.join(data_dir, str(y), f"{y}_{s}.parquet")
            if not os.path.isfile(p):
                raise IntegrityStop(f"missing file {rel}")
            if rel not in g1:
                raise IntegrityStop(f"{rel} not in G1 evidence")
            g = g1[rel]
            size = os.path.getsize(p)
            if size != g["size"]:
                raise IntegrityStop(f"{rel}: size {size} != G1 {g['size']}")
            sha = sha256_file(p)
            etag = g["head"]["provenance"]["x-linked-etag (server-provided; algorithm not asserted)"].strip('"')
            if sha != local.get(rel):
                raise IntegrityStop(f"{rel}: SHA-256 differs from {local_manifest_name}")
            etag_match = sha == etag
            if not etag_match:
                raise IntegrityStop(f"{rel}: SHA-256 differs from G1 server X-Linked-Etag")
            flen = g["footer_len"]
            with open(p, "rb") as fh:
                fh.seek(size - 8)
                tail = fh.read(8)
                fh.seek(size - 8 - flen)
                footer = fh.read(flen)
            if tail[4:] != b"PAR1" or int.from_bytes(tail[:4], "little") != flen:
                raise IntegrityStop(f"{rel}: tail/footer length differs from G1")
            if hashlib.sha256(footer).hexdigest() != g["raw_footer_sha256"]:
                raise IntegrityStop(f"{rel}: footer bytes differ from G1 raw footer")
            md = pq.read_metadata(p)
            sch = md.schema.to_arrow_schema()
            if sch.names != names or list(sch.types) != types:
                raise IntegrityStop(f"{rel}: schema mismatch")
            g1rows = g["decode"]["summary"]["num_rows"]
            if md.num_rows != g1rows:
                raise IntegrityStop(f"{rel}: rows {md.num_rows} != G1 {g1rows}")
            files.append({"rel": rel, "path": p, "Y": y, "S": s, "size": size, "sha256": sha,
                          "g1_x_linked_etag_equals_sha256": etag_match, "footer_len": flen,
                          "footer_matches_g1": True, "rows": md.num_rows, "row_groups": md.num_row_groups})
            total += md.num_rows
    if total != expected_total:
        raise IntegrityStop(f"total rows {total} != expected {expected_total}")
    return files, total


# ------------------------------------------------------------------ stage 2-4: metadata, cohorts
def ym_status(v, y):
    if v is None:
        return "INVALID", "null"
    if v == "unknown":
        return "UNKNOWN_SENTINEL", None
    if v == "":
        return "INVALID", "empty"
    m = YM_RE.match(v)
    if not m:
        return "INVALID", ("unknown_variant" if v.strip().lower() == "unknown" else "malformed")
    if int(m.group(1)) != y:
        return "INVALID", "year_mismatch"
    return "VALID", None


def analyze(files, nfeat=4561):
    stops, review = [], []
    rows = []          # per row: (gid, Y, S, r_phys, label, ym, status, fam_cat, family, hash_lc)
    per_file, traces = [], {}
    month_label = collections.Counter()      # (Y, S, ym_or_status, label)
    gid = 0
    for f in files:
        t, tr = projected_read(f["path"], f["size"], f["footer_len"], READ_COLS)
        traces[f["rel"]] = tr
        hashes = t.column("hash").to_pylist()
        labels = t.column("label").to_pylist()
        fams = [None if v is None else sys.intern(v) for v in t.column("family").to_pylist()]
        yms = [None if v is None else sys.intern(v) for v in t.column("year_month").to_pylist()]
        del t
        c = collections.Counter()
        fam_distinct = set()
        y, s = f["Y"], f["S"]
        for r in range(len(hashes)):
            h, lab, fam, ym = hashes[r], labels[r], fams[r], yms[r]
            if h is None:
                c["hash_null"] += 1
                hlc = None
            elif not HASH_RE.match(h):
                c["hash_malformed"] += 1
                hlc = h.lower()
            else:
                hlc = h.lower()
                if h != hlc:
                    c["hash_uppercase_present"] += 1
            if lab is None:
                c["label_null"] += 1
            elif lab not in (0, 1):
                c["label_out_of_domain"] += 1
            else:
                c[f"label_{lab}"] += 1
            st, why = ym_status(ym, y)
            c[f"ym_{st}"] += 1
            if why:
                c[f"ym_invalid_{why}"] += 1
            if fam is None or fam == "":
                fcat = "missing"
                c["family_missing_null" if fam is None else "family_missing_empty"] += 1
            elif fam == "unknown":
                fcat = "unknown"
                c["family_unknown"] += 1
                fam_distinct.add(fam)
            elif SINGLETON_RE.match(fam):
                fcat = "singleton"            # AVClass singleton: value embeds a sample hash -> never exported
                c["family_singleton"] += 1
                if hlc is not None and hlc in fam.lower():
                    c["family_singleton_embeds_own_hash"] += 1
                fam = SINGLETON
            elif HEXRUN_RE.search(fam):
                fcat = "present"
                c["family_present_hash_bearing"] += 1
                fam = HASH_BEARING
            else:
                fcat = "present"
                c["family_present_other"] += 1
                fam_distinct.add(fam)
            month_label[(y, s, ym if st == "VALID" else st, lab)] += 1
            rows.append((gid, y, s, r, lab, ym, st, fcat, fam if fcat != "missing" else None, hlc))
            gid += 1
        per_file.append(dict(f, counts=dict(c), family_distinct=len(fam_distinct)))
        for k, code in (("hash_null", "HASH_NULL"), ("hash_malformed", "HASH_MALFORMED"),
                        ("label_null", "LABEL_NULL"), ("label_out_of_domain", "LABEL_OUT_OF_DOMAIN"),
                        ("ym_INVALID", "YEAR_MONTH_INVALID")):
            if c.get(k):
                stops.append({"code": code, "file": f["rel"], "count": c[k]})
        if y in (2013, 2014) and c.get("ym_UNKNOWN_SENTINEL"):
            stops.append({"code": "YEAR_MONTH_SENTINEL_IN_2013_2014", "file": f["rel"],
                          "count": c["ym_UNKNOWN_SENTINEL"]})
    return rows, per_file, traces, month_label, stops, review


def build_cohorts(rows):
    mstar = {}
    for Y in (2013, 2014):
        vals = [r[5] for r in rows if r[1] == Y and r[6] == "VALID"]
        if not vals:
            return None, None, [{"code": "M_STAR_UNDEFINED", "year": Y}]
        mstar[Y] = max(vals)  # lexicographic YYYY-MM == chronological
    assign = []
    for r in rows:
        y = r[1]
        if y in (2013, 2014):
            assign.append(f"IID_{y}" if r[5] == mstar[y] else "TRAIN")
        elif y in (2016, 2017):
            assign.append(f"NEAR_{y}")
        else:
            assign.append(f"FAR_{y}")
    return mstar, assign, []


def cohort_of(unit):
    return "TRAIN" if unit == "TRAIN" else unit.split("_")[0]


def validate_partition(rows, assign, per_file, mstar, month_label, expected_total):
    problems = []
    n = len(rows)
    if n != expected_total or len(assign) != n:
        problems.append({"code": "ROW_ACCOUNTING", "rows": n, "assigned": len(assign)})
    if [r[0] for r in rows] != list(range(n)):
        problems.append({"code": "GID_NOT_CONTIGUOUS"})
    if any(a not in SETS for a in assign):
        problems.append({"code": "UNASSIGNED_OR_UNKNOWN_SET"})
    sizes = collections.Counter(assign)
    if sum(sizes.values()) != n:
        problems.append({"code": "PARTITION_SUM"})
    # independent recount from file-level and month-level aggregates
    file_rows = collections.Counter()
    for f in per_file:
        file_rows[f["Y"]] += f["rows"]
    exp = {}
    for Y in (2013, 2014):
        iid = sum(v for (y, s, m, lab), v in month_label.items() if y == Y and m == mstar[Y])
        exp[f"IID_{Y}"] = iid
        exp.setdefault("TRAIN", 0)
        exp["TRAIN"] += file_rows[Y] - iid
    for Y in (2016, 2017):
        exp[f"NEAR_{Y}"] = file_rows[Y]
    for Y in range(2018, 2026):
        exp[f"FAR_{Y}"] = file_rows[Y]
    for k in SETS:
        if sizes.get(k, 0) != exp.get(k):
            problems.append({"code": "RECOUNT_MISMATCH", "set": k, "assigned": sizes.get(k, 0), "expected": exp.get(k)})
    for k in UNITS:
        if sizes.get(k, 0) == 0:
            problems.append({"code": "EMPTY_UNIT", "unit": k})
    train_labels = {rows[i][4] for i, a in enumerate(assign) if a == "TRAIN"}
    if train_labels != {0, 1}:
        problems.append({"code": "TRAIN_NOT_BOTH_CLASSES", "classes": sorted(train_labels)})
    return sizes, exp, problems


def duplicates(rows, assign):
    groups = collections.defaultdict(list)
    for r in rows:
        if r[9] is not None:
            groups[r[9]].append(r[0])
    dup = {h: g for h, g in groups.items() if len(g) > 1}
    st = collections.Counter()
    for k in ("groups", "rows_in_groups", "excess_rows", "max_group_size", "groups_within_single_file",
              "groups_spanning_files", "groups_spanning_train_test_same_year", "groups_spanning_years",
              "groups_spanning_sets", "groups_touching_TRAIN_and_an_eval_unit"):
        st[k] = 0  # explicit zeros in every report
    conflict_groups = []
    listing = []
    pair_sets = collections.Counter()
    for h, g in dup.items():
        files = {(rows[i][1], rows[i][2]) for i in g}
        years = {rows[i][1] for i in g}
        sets_ = {assign[i] for i in g}
        labs = {rows[i][4] for i in g}
        st["groups"] += 1
        st["rows_in_groups"] += len(g)
        st["excess_rows"] += len(g) - 1
        st["max_group_size"] = max(st["max_group_size"], len(g))
        st["groups_within_single_file"] += len(files) == 1
        st["groups_spanning_files"] += len(files) > 1
        st["groups_spanning_train_test_same_year"] += any((y, "train") in files and (y, "test") in files for y in years)
        st["groups_spanning_years"] += len(years) > 1
        st["groups_spanning_sets"] += len(sets_) > 1
        st["groups_touching_TRAIN_and_an_eval_unit"] += ("TRAIN" in sets_ and len(sets_) > 1)
        for a in sorted(sets_):
            for b in sorted(sets_):
                if a < b:
                    pair_sets[f"{a}|{b}"] += 1
        if len(labs) > 1:
            conflict_groups.append(g)
        listing.append({"gids": g, "sets": sorted(sets_), "labels": sorted(labs, key=str)})
    st["label_conflict_groups"] = len(conflict_groups)
    st["label_conflict_rows"] = sum(len(g) for g in conflict_groups)
    # rows in each eval unit whose hash also occurs in TRAIN (hash-level only; NOT G3 seen status)
    train_hashes = {rows[i][9] for i, a in enumerate(assign) if a == "TRAIN" and rows[i][9] is not None}
    overlap = collections.Counter()
    for i, a in enumerate(assign):
        if a != "TRAIN" and rows[i][9] in train_hashes:
            overlap[(a, rows[i][4])] += 1
    st["distinct_hashes"] = len(groups)
    return dict(st), dict(pair_sets), overlap, listing


def membership_digest(rows, assign):
    h = hashlib.sha256()
    for r, a in zip(rows, assign):  # rows already in gid order
        row_uid = f"{REV}/{r[1]}/{r[2]}/{r[3]}"
        ym = "" if r[5] is None else r[5]
        h.update(f"{r[0]}|{row_uid}|{cohort_of(a)}|{a}|{r[4]}|{ym}|{r[6]}\n".encode("utf-8"))
    return h.hexdigest()


# ------------------------------------------------------------------ outputs
def write_csv(path, header, rows_):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows_)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)


def unit_table(rows, assign, overlap):
    agg = collections.defaultdict(collections.Counter)
    fam = collections.defaultdict(collections.Counter)
    famset = collections.defaultdict(set)
    for r, a in zip(rows, assign):
        lab = r[4]
        A = agg[a]
        A["n"] += 1
        A[f"n_{lab}"] += 1
        A[f"partition_{r[2]}_{lab}"] += 1
        if r[6] == "UNKNOWN_SENTINEL":
            A[f"unknown_month_{lab}"] += 1
        A[f"family_{r[7]}_{lab}"] += 1
        fam[(a, lab)][r[8] if r[8] is not None else MISSING] += 1
        if r[8] is not None and r[8] not in (SINGLETON, HASH_BEARING):
            famset[a].add(r[8])
    out = []
    for k in SETS:
        A = agg[k]
        n0, n1 = A["n_0"], A["n_1"]
        out.append({
            "set": k, "cohort": cohort_of(k), "primary_unit": k in PRIMARY_UNITS,
            "n": A["n"], "n_benign": n0, "n_malware": n1,
            "prevalence_malware": round(n1 / A["n"], 6) if A["n"] else None,
            "train_partition_benign": A["partition_train_0"], "train_partition_malware": A["partition_train_1"],
            "test_partition_benign": A["partition_test_0"], "test_partition_malware": A["partition_test_1"],
            "unknown_month_benign": A["unknown_month_0"], "unknown_month_malware": A["unknown_month_1"],
            "family_missing_benign": A["family_missing_0"], "family_missing_malware": A["family_missing_1"],
            "family_unknown_benign": A["family_unknown_0"], "family_unknown_malware": A["family_unknown_1"],
            "family_singleton_benign": A["family_singleton_0"], "family_singleton_malware": A["family_singleton_1"],
            "distinct_family_values_excluding_singletons": len(famset[k]),
            "hash_also_in_TRAIN_benign": overlap.get((k, 0), 0) if k != "TRAIN" else None,
            "hash_also_in_TRAIN_malware": overlap.get((k, 1), 0) if k != "TRAIN" else None,
            "full_arm_auc_defined": (n0 >= 1 and n1 >= 1) if k != "TRAIN" else None,
        })
    return out, fam


def run(args, nfeat=4561, expected_total=EXPECTED_TOTAL):
    started = now_utc()
    for p in (args.data_dir, args.g1_run, args.review_dir, args.out_root):
        ap = os.path.abspath(p)
        if os.path.splitdrive(ap)[0].upper() == "D:" or "android-malware-detection" in ap.lower():
            print("G2: STOP - forbidden path refused")
            return 3
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(args.out_root, f"{stamp}_G2LOCAL{('_' + args.label) if args.label else ''}")
    os.makedirs(os.path.join(out, "tables"))
    os.makedirs(os.path.join(out, "restricted"))
    summary = {"task": "C1-R G2 LOCAL", "version": VERSION, "revision": REV, "started_utc": started,
               "python": platform.python_version(), "pyarrow": pa.__version__, "read_columns": list(READ_COLS),
               "never_read": ["vt_count", "feat_*"], "label_semantics": LABEL_SEMANTICS}
    code = 0
    try:
        if not args.skip_pins and (platform.python_version() != PIN["python"] or pa.__version__ != PIN["pyarrow"]):
            raise IntegrityStop("environment does not match pin")
        anchors = {}
        for n, h in ANCHORS.items():
            p = os.path.join(args.review_dir, n)
            if not os.path.isfile(p) or sha256_file(p) != h:
                raise IntegrityStop(f"anchor missing or changed: {n}")
            anchors[n] = h
        summary["anchors"] = anchors
        g1_sums = os.path.join(args.g1_run, "SHA256SUMS.txt")
        summary["g1_evidence"] = {"run": os.path.basename(args.g1_run), "SHA256SUMS_sha256": sha256_file(g1_sums),
                                  "inventory_sha256": sha256_file(os.path.join(args.g1_run, "inventory.json"))}
        files, total = verify_inputs(args.data_dir, args.g1_run, nfeat, expected_total)
        summary["inputs"] = {"files": 24, "total_rows": total, "file_sha256": {f["rel"]: f["sha256"] for f in files},
                             "local_manifest_sha256": sha256_file(os.path.join(args.data_dir, "LOCAL_SHA256SUMS.txt"))}
        rows, per_file, traces, month_label, stops, _ = analyze(files, nfeat)
        summary["io_trace"] = {"files": len(traces),
                               "reads_outside_authorized_total": sum(t["reads_outside_authorized"] for t in traces.values()),
                               "bytes_read_by_column_total": {c: sum(t["bytes_read_by_column"][c] for t in traces.values()) for c in READ_COLS},
                               "authorized_chunk_bytes_total": {c: sum(t["authorized_chunk_bytes"][c] for t in traces.values()) for c in READ_COLS}}
        write_json(os.path.join(out, "io_trace_per_file.json"), traces)
        # ---- table 1: files
        keys = ["label_0", "label_1", "label_null", "label_out_of_domain", "ym_VALID", "ym_UNKNOWN_SENTINEL",
                "ym_INVALID", "family_missing_null", "family_missing_empty", "family_unknown", "family_present_other",
                "family_singleton", "family_singleton_embeds_own_hash", "family_present_hash_bearing",
                "hash_null", "hash_malformed", "hash_uppercase_present"]
        write_csv(os.path.join(out, "tables", "t01_files.csv"),
                  ["file", "Y", "S", "rows", "size_bytes", "sha256", "sha256_equals_g1_x_linked_etag", "footer_matches_g1"]
                  + keys + ["family_distinct"],
                  [[f["rel"], f["Y"], f["S"], f["rows"], f["size"], f["sha256"], f["g1_x_linked_etag_equals_sha256"],
                    f["footer_matches_g1"]] + [f["counts"].get(k, 0) for k in keys] + [f["family_distinct"]] for f in per_file])
        write_csv(os.path.join(out, "tables", "t02_year_month_label.csv"), ["Y", "S", "year_month_or_status", "label", "rows"],
                  [[y, s, m, lab, v] for (y, s, m, lab), v in sorted(month_label.items(), key=lambda kv: tuple(map(str, kv[0])))])
        summary["file_counts"] = {f["rel"]: f["counts"] for f in per_file}
        summary["data_stops"] = stops
        if stops:
            summary["verdict"] = "STOP"
            summary["stop_reason"] = "scientific/data STOP condition (Manifest §3.3-§3.5, §9 G2); cohorts not constructed"
            code = 2
        else:
            mstar, assign, mprob = build_cohorts(rows)
            if mprob:
                raise IntegrityStop(f"M* undefined: {mprob}")
            summary["m_star"] = {"2013": mstar[2013], "2014": mstar[2014],
                                 "pc_a_reference": {"2013": "2013-12", "2014": "2014-08"},
                                 "differs_from_pc_a": {"2013": mstar[2013] != "2013-12", "2014": mstar[2014] != "2014-08"}}
            sizes, exp, problems = validate_partition(rows, assign, per_file, mstar, month_label, expected_total)
            summary["partition"] = {"sizes": dict(sizes), "independent_recount": exp, "sum": sum(sizes.values()),
                                    "problems": problems, "sets": list(SETS)}
            dst, pairs, overlap, listing = duplicates(rows, assign)
            ut, fam = unit_table(rows, assign, overlap)
            summary["units"] = ut
            summary["duplicates"] = dst
            summary["duplicate_set_pairs"] = pairs
            summary["membership_digest_sha256"] = membership_digest(rows, assign)
            summary["membership_digest_format"] = "UTF-8 lines gid|row_uid|cohort|unit|label|ym|ym_status (row_uid=revision/Y/S/r_phys), gid order"
            cols = list(ut[0].keys())
            write_csv(os.path.join(out, "tables", "t03_cohort_units.csv"), cols, [[u[c] for c in cols] for u in ut])
            write_csv(os.path.join(out, "tables", "t04_family_by_unit_label.csv"), ["set", "label", "family", "rows"],
                      [[a, lab, fname, v] for (a, lab), cnt in sorted(fam.items(), key=lambda kv: (SETS.index(kv[0][0]), kv[0][1]))
                       for fname, v in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))])
            write_json(os.path.join(out, "tables", "t05_duplicates.json"), {"stats": dst, "set_pairs": pairs})
            write_json(os.path.join(out, "tables", "t06_m_star.json"), summary["m_star"])
            with open(os.path.join(out, "restricted", "duplicate_groups_by_gid.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
                for g in listing:
                    fh.write(json.dumps(g) + "\n")
            summary["definedness"] = {
                "full_arm": {u["set"]: u["full_arm_auc_defined"] for u in ut if u["set"] != "TRAIN"},
                "primary_units_all_full_arm_defined": all(u["full_arm_auc_defined"] for u in ut if u["primary_unit"]),
                "unseen_arm": "NOT DETERMINABLE AT G2 - requires G3 vector keys and seen status"}
            if problems:
                summary["verdict"], summary["stop_reason"], code = "STOP", "cohort invariant failure (Manifest §3.6)", 2
            elif dst["label_conflict_groups"] > 0:
                summary["verdict"] = "STOP (SUPERVISOR REVIEW REQUIRED)"
                summary["stop_reason"] = "hash anomaly: same hash with different labels (Manifest §3.7, §9 G2)"
                code = 2
            else:
                summary["verdict"] = "PASS"
    except IntegrityStop as e:
        summary["verdict"], summary["stop_reason"], code = "STOP (INTEGRITY)", str(e), 3
    summary["finished_utc"] = now_utc()
    write_json(os.path.join(out, "g2_summary.json"), summary)
    write_report(out, summary)
    write_input_sums(out, args, summary)  # SHA-256 records: inputs, then outputs
    write_sums(out)
    print(f"G2 LOCAL: {summary['verdict']}" + (f" - {summary.get('stop_reason')}" if summary.get("stop_reason") else ""))
    print(f"out={out}")
    return code


def write_input_sums(out, args, summary):
    lines = []
    for y in YEARS:
        for s in SPLITS:
            p = os.path.join(args.data_dir, str(y), f"{y}_{s}.parquet")
            rel = f"Baseline/{y}/{y}_{s}.parquet"
            known = summary.get("inputs", {}).get("file_sha256", {}).get(rel)
            if known or os.path.isfile(p):
                lines.append(f"{known or sha256_file(p)} *{rel}")
    for name in ("LOCAL_SHA256SUMS.txt",):
        p = os.path.join(args.data_dir, name)
        if os.path.isfile(p):
            lines.append(f"{sha256_file(p)} *Baseline/{name}")
    for name in ("inventory.json", "SHA256SUMS.txt"):
        lines.append(f"{sha256_file(os.path.join(args.g1_run, name))} *G1:{os.path.basename(args.g1_run)}/{name}")
    for n in ANCHORS:
        p = os.path.join(args.review_dir, n)
        if os.path.isfile(p):
            lines.append(f"{sha256_file(p)} *review/{n}")
    me = os.path.abspath(__file__)
    lines.append(f"{sha256_file(me)} *src/{os.path.basename(me)}")
    with open(os.path.join(out, "INPUT_SHA256.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def write_sums(out):
    lines = []
    for root, _, fs in os.walk(out):
        for f in sorted(fs):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out).replace("\\", "/")
            if rel != "SHA256SUMS.txt":
                lines.append(f"{sha256_file(p)} *{rel}")
    with open(os.path.join(out, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sorted(lines, key=lambda s: s.split("*", 1)[1])) + "\n")


def write_report(out, S):
    L = ["# C1-R G2 LOCAL — Final Report", "",
         f"- **Verdict:** **{S.get('verdict')}**" + (f" — {S['stop_reason']}" if S.get("stop_reason") else ""),
         f"- Revision `{REV}`; analysis v{VERSION}; Python {S['python']}, pyarrow {S['pyarrow']}",
         f"- Started/finished (UTC): {S['started_utc']} / {S.get('finished_utc')}",
         "- Columns read: `hash`, `label`, `family`, `year_month` (projection, I/O-traced). Never read: `vt_count`, `feat_*`.",
         "- Report contains aggregates only; no hash or other row-level value.", ""]
    if "inputs" in S:
        it = S.get("io_trace", {"reads_outside_authorized_total": "STOPPED", "bytes_read_by_column_total": "n/a",
                                "authorized_chunk_bytes_total": "n/a"})
        L += ["## 1. Input verification", "",
              f"- 24/24 files; total rows **{S['inputs']['total_rows']:,}**; each file's SHA-256 equals the local manifest "
              "**and** the G1 server `X-Linked-Etag`; footer bytes equal the G1 raw footers; schema and row counts equal G1.",
              f"- I/O trace: reads outside footer/authorized chunks = **{it['reads_outside_authorized_total']}**; "
              f"bytes read per column = {it['bytes_read_by_column_total']} (authorized chunk bytes = {it['authorized_chunk_bytes_total']}).", ""]
    if S.get("data_stops"):
        L += ["## STOP conditions", ""] + [f"- `{s['code']}` in `{s['file']}`: {s['count']} rows" for s in S["data_stops"]] + [""]
    if "file_counts" in S:
        tot = collections.Counter()
        for c in S["file_counts"].values():
            tot.update(c)
        L += ["## 2. Metadata quality (all 24 files)", "", "| Check | Rows |", "|---|---|"]
        for k in ("label_0", "label_1", "label_null", "label_out_of_domain", "ym_VALID", "ym_UNKNOWN_SENTINEL", "ym_INVALID",
                  "family_missing_null", "family_missing_empty", "family_unknown", "family_present_other",
                  "family_singleton", "family_singleton_embeds_own_hash", "family_present_hash_bearing",
                  "hash_null", "hash_malformed", "hash_uppercase_present"):
            L.append(f"| {k} | {tot.get(k, 0):,} |")
        L += ["", "Label semantics: 0 = benign, 1 = malware (Manifest §3.4: official documentation S6; code S5); "
              "not inferred from class frequencies.", ""]
    if "m_star" in S:
        m = S["m_star"]
        L += ["## 3. P-A cohorts", "",
              f"- M\\*_2013 = **{m['2013']}**, M\\*_2014 = **{m['2014']}** (over both partitions; PC-A reference 2013-12 / 2014-08; "
              f"differs: {m['differs_from_pc_a']}).",
              f"- Partition: 13 sets, sum = **{S['partition']['sum']:,}**; independent recount equal; problems: {S['partition']['problems'] or 'none'}.",
              f"- Membership digest: `{S['membership_digest_sha256']}`", "",
              "| Set | n | benign | malware | prev. | train-part. (b/m) | test-part. (b/m) | unknown-month (b/m) | missing family (b/m) | full-arm AUC defined |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for u in S["units"]:
            L.append(f"| {u['set']} | {u['n']:,} | {u['n_benign']:,} | {u['n_malware']:,} | {u['prevalence_malware']} | "
                     f"{u['train_partition_benign']:,}/{u['train_partition_malware']:,} | {u['test_partition_benign']:,}/{u['test_partition_malware']:,} | "
                     f"{u['unknown_month_benign']:,}/{u['unknown_month_malware']:,} | {u['family_missing_benign']:,}/{u['family_missing_malware']:,} | "
                     f"{'—' if u['full_arm_auc_defined'] is None else ('YES' if u['full_arm_auc_defined'] else 'NO')} |")
        d = S["duplicates"]
        L += ["", "## 4. Hash duplicates (case-insensitive; counted, never dropped)", ""] + \
             [f"- {k}: {v:,}" for k, v in sorted(d.items())] + \
             ["", "## 5. Definedness", "",
              f"- Full (F) arm: all 10 primary units defined = **{S['definedness']['primary_units_all_full_arm_defined']}**.",
              "- **Unseen-only (U) arm definedness cannot be established at G2.** It requires G3 vector keys and seen status.",
              "- `hash_also_in_TRAIN_*` columns are a hash-level overlap descriptor only, not the G3 seen status.", ""]
    L += ["## Limitations", "",
          "- Local files were obtained outside the G1 runner; identity is established by whole-file SHA-256 = G1 server digest and footer equality.",
          "- `feat_*` {0,1} validity, vector keys, `K_TRAIN`, seen status, conflicts and U-arm estimability are G3 facts.",
          "- Storage-partition composition does not authenticate upstream representation-fit membership (AM-05).",
          "- G3–G6 not authorized; nothing beyond G2 was executed."]
    with open(os.path.join(out, "G2_FINAL_REPORT.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"<PRIVATE-PATH-NOT-PUBLISHED>")
    ap.add_argument("--g1-run", default=r"<PRIVATE-PATH-NOT-PUBLISHED>")
    ap.add_argument("--review-dir", default=r"<PRIVATE-PATH-NOT-PUBLISHED>")
    ap.add_argument("--out-root", default=r"<PRIVATE-PATH-NOT-PUBLISHED>")
    ap.add_argument("--label", default="")
    ap.add_argument("--skip-pins", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse()))
