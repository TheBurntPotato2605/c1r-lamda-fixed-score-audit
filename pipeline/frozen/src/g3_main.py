"""C1-R G3 LOCAL — feature validation, exact TRAIN-vector membership (seen/unseen) and E2 estimability.

Authority: owner approval "G3 LOCAL". Frozen rules: Manifest v0.2.4 §3.6 (cohorts), §4 (keys, exact
equality, K_TRAIN, seen, coverage, sealing), §7.3 (definedness, AM-03), §7.5 (conflict/all-zero counts),
§9 G3. No model, score or AUC is computed. vt_count and hash are never read.

Exit: 0 PASS (E2 ESTIMABLE); 2 STOP (scientific: invalid features / NOT ESTIMABLE / exactness);
3 STOP (integrity: inputs, schema, cohorts, access scope).
"""
import argparse
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g3core as C  # noqa: E402
import pyarrow as pa  # noqa: E402

VERSION = "1.1.0"
NF = 4561
EXPECTED_TOTAL = 1_008_381
G2_SOURCE_SHA = "3b7615c844c1cda7804b11baaf44570396faaa0a73677ad6fdddeba74c85df38"  # g2_analysis.py v1.1.0
G2_DIGEST = "f4b62175292564f08288ce19e05db2974cf4659464146757b190c73cbd569bc8"
DEFAULTS = {
    "data_dir": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "g1_run": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "g2_work": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "g2_run": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "review_dir": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "out_root": r"<PRIVATE-PATH-NOT-PUBLISHED>",
}
LAB = {0: "benign", 1: "malware"}
TWO_TO_ONE = bytes.maketrans(b"\x02", b"\x01")   # used ONLY for a labelled descriptive counterfactual
VALUE2_ROWS = []
TRAIN_HAS_TWO = [False]


class ScientificStop(Exception):
    pass


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_g2_module(g2_work):
    p = os.path.join(g2_work, "src", "g2_analysis.py")
    if C_sha(p) != G2_SOURCE_SHA:
        raise C.IntegrityStop("G2 analysis source hash differs from the final G2 version")
    spec = importlib.util.spec_from_file_location("g2_analysis", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def C_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ Phase A: inputs + cohorts
def verify_g2_run(g2_run):
    with open(os.path.join(g2_run, "SHA256SUMS.txt"), encoding="utf-8") as fh:
        for line in fh:
            h, rel = line.strip().split(" *", 1)
            if C_sha(os.path.join(g2_run, rel)) != h:
                raise C.IntegrityStop(f"G2 output changed: {rel}")
    S = json.load(open(os.path.join(g2_run, "g2_summary.json"), encoding="utf-8"))
    if S.get("verdict") != "PASS":
        raise C.IntegrityStop("G2 verdict is not PASS")
    return S


def reconstruct_cohorts(G, files, expected_digest, g2_sizes):
    """Metadata-only pass (label, family, year_month; I/O-traced) and canonical P-A assignment."""
    meta = []  # (gid, Y, S, r_phys, label, ym, status, fam_missing)
    gid = 0
    for f in files:
        t, _ = G.projected_read(f["path"], f["size"], f["footer_len"], ("label", "family", "year_month"))
        labs, fams, yms = (t.column(c).to_pylist() for c in ("label", "family", "year_month"))
        for r, (lab, fam, ym) in enumerate(zip(labs, fams, yms)):
            st, _why = G.ym_status(ym, f["Y"])
            if lab not in (0, 1) or st == "INVALID" or (f["Y"] in (2013, 2014) and st != "VALID"):
                raise C.IntegrityStop("metadata no longer satisfies G2 contract")
            meta.append((gid, f["Y"], f["S"], r, lab, ym, st, fam is None or fam == ""))
            gid += 1
    mstar = {Y: max(m[5] for m in meta if m[1] == Y and m[6] == "VALID") for Y in (2013, 2014)}
    assign = []
    for m in meta:
        y = m[1]
        if y in (2013, 2014):
            assign.append(f"IID_{y}" if m[5] == mstar[y] else "TRAIN")
        else:
            assign.append(("NEAR_" if y < 2018 else "FAR_") + str(y))
    h = hashlib.sha256()
    for m, a in zip(meta, assign):
        h.update(f"{m[0]}|{G.REV}/{m[1]}/{m[2]}/{m[3]}|{G.cohort_of(a)}|{a}|{m[4]}|{'' if m[5] is None else m[5]}|{m[6]}\n".encode())
    digest = h.hexdigest()
    sizes = dict(collections.Counter(assign))
    if expected_digest is not None and digest != expected_digest:
        raise C.IntegrityStop("reconstructed membership digest differs from G2")
    if g2_sizes is not None and sizes != g2_sizes:
        raise C.IntegrityStop("reconstructed cohort sizes differ from G2")
    return meta, assign, mstar, digest, sizes


# ------------------------------------------------------------------ Phase B: feature validation
def validate_features(files, nf, batch_rows):
    """G3-DEV-01: accepted domain is exactly {0,1,2} (int8, as released). Anything else, any null
    or any non-int8 column is a STOP. Value counts are aggregate only."""
    tot = collections.Counter()
    bad_values = collections.Counter()
    bad_cols, two_cols = set(), set()
    for f in files:
        n = 0
        for labs, _fams, _yms, arrs in C.iter_batches(f["path"], f["size"], f["footer_len"], nf, batch_rows):
            m = len(labs)
            nulls = sum(a.null_count for a in arrs)
            tot["null_cells"] += nulls
            if any(a.type != pa.int8() for a in arrs):
                tot["non_int8_columns"] += sum(a.type != pa.int8() for a in arrs)
                continue
            views = C.column_views(arrs)
            blob = b"".join(views)
            c0, c1, c2 = blob.count(b"\x00"), blob.count(b"\x01"), blob.count(b"\x02")
            tot["value_0_cells"] += c0
            tot["value_1_cells"] += c1
            tot["value_2_cells"] += c2
            left = blob.translate(None, C.ALLOWED_BYTES)
            if left or c2:
                for k, v in enumerate(views):
                    b = bytes(v)
                    if b"\x02" in b:
                        two_cols.add(k)
                    lk = b.translate(None, C.ALLOWED_BYTES)
                    if lk:
                        bad_cols.add(k)
                        for x in lk:
                            bad_values[x - 256 if x > 127 else x] += 1
            tot["out_of_domain_cells"] += len(left)
            tot["cells"] += m * nf
            n += m
        if n != f["rows"]:
            raise C.IntegrityStop(f"{f['rel']}: validated rows {n} != expected {f['rows']}")
        tot["rows"] += n
    tot["columns_with_value_2"] = len(two_cols)
    return dict(tot), {str(k): v for k, v in sorted(bad_values.items())}, len(bad_cols)


# ------------------------------------------------------------------ Phase C: exact membership
class DiskStore:
    """Append-only store of bit-packed representative vectors (restricted temp; deleted at end)."""

    def __init__(self, path, nf):
        self.path, self.nf, self.pl = path, nf, C.packed_len(nf)
        self.fh = open(path, "w+b")
        self.n = 0

    def add(self, key):
        off = self.n * self.pl
        self.fh.seek(off)
        self.fh.write(C.pack(key, self.nf))
        self.n += 1
        return off

    def get(self, off):
        self.fh.seek(off)
        return C.unpack(self.fh.read(self.pl), self.nf)

    def close(self):
        self.fh.close()


def build_k_train(files, assign, meta_labels, nf, batch_rows, block, digest_fn, stats):
    """K_TRAIN over every TRAIN row (any label). Exact: each digest hit is confirmed by full
    4,561-byte comparison; a mismatch splits the bucket and is logged as a collision."""
    KT = {}   # digest -> list of [packed, n_benign, n_malware]
    gid = 0
    for f in files:
        if f["Y"] not in (2013, 2014):
            gid += f["rows"]
            continue
        for labs, _f, _y, arrs in C.iter_batches(f["path"], f["size"], f["footer_len"], nf, batch_rows):
            views = C.column_views(arrs)
            for j, key in enumerate(C.keys_from_views(views, len(labs), block)):
                g = gid + j
                if labs[j] != meta_labels[g]:
                    raise C.IntegrityStop("row alignment check failed (label mismatch)")
                if assign[g] != "TRAIN":
                    continue
                if b"\x02" in key:
                    stats["train_rows_with_value_2"] += 1
                    TRAIN_HAS_TWO[0] = True
                d = digest_fn(key)
                bucket = KT.get(d)
                hit = None
                if bucket is not None:
                    for ent in bucket:
                        stats["comparisons_train"] += 1
                        if C.unpack(ent[0], nf) == key:   # full-vector byte equality
                            hit = ent
                            break
                    if hit is None:
                        stats["collisions_train"] += 1
                if hit is None:
                    hit = [C.pack(key, nf), 0, 0]
                    KT.setdefault(d, []).append(hit)
                hit[1 + labs[j]] += 1
                stats["train_rows_keyed"] += 1
            gid += len(labs)
    return KT


def canonical_pass(files, assign, meta, nf, batch_rows, block, digest_fn, KT, tmpdir, keys_csv, stats):
    """All rows in canonical gid order: kid (first occurrence, exact), seen (exact vs K_TRAIN), all-zero."""
    zero = bytes(nf)
    store = DiskStore(os.path.join(tmpdir, "global_reps.bin"), nf)
    GI = {}                                   # digest -> list of [kid, offset]
    per = collections.defaultdict(collections.Counter)   # (set, label) -> counters
    kid_counts = collections.defaultdict(dict)           # set -> kid -> [n0, n1]
    u_kids = collections.defaultdict(set)                # unit -> kids among unseen rows
    coverage = []
    gid = 0
    with open(keys_csv, "w", encoding="utf-8", newline="\n") as kf:
        kf.write("gid,kid,vector_sha256,all_zero,seen\n")
        for f in files:
            n_file = 0
            for labs, _f, _y, arrs in C.iter_batches(f["path"], f["size"], f["footer_len"], nf, batch_rows):
                views = C.column_views(arrs)
                for j, key in enumerate(C.keys_from_views(views, len(labs), block)):
                    g = gid + j
                    m = meta[g]
                    lab = labs[j]
                    if lab != m[4]:
                        raise C.IntegrityStop("row alignment check failed (label mismatch)")
                    a = assign[g]
                    d = digest_fn(key)
                    # ---- kid: exact first occurrence
                    kid = None
                    bucket = GI.get(d)
                    if bucket is not None:
                        for ent in bucket:
                            stats["comparisons_global"] += 1
                            if store.get(ent[1]) == key:
                                kid = ent[0]
                                break
                        if kid is None:
                            stats["collisions_global"] += 1
                    if kid is None:
                        kid = g
                        GI.setdefault(d, []).append([g, store.add(key)])
                    # ---- seen: exact membership in K_TRAIN
                    seen = False
                    kb = KT.get(d)
                    if kb is not None:
                        for ent in kb:
                            stats["comparisons_seen"] += 1
                            if C.unpack(ent[0], nf) == key:
                                seen = True
                                break
                        if not seen:
                            stats["collisions_seen"] += 1
                    az = key == zero
                    if b"\x02" in key:   # G3-DEV-01 explicit verification of value-2 rows
                        alt = key.translate(TWO_TO_ONE)
                        alt_seen = any(C.unpack(e[0], nf) == alt for e in KT.get(digest_fn(alt), []))
                        VALUE2_ROWS.append({"gid": g, "unit": a, "label": LAB[lab], "partition": m[2],
                                            "cells_with_value_2": key.count(b"\x02"), "seen_exact": seen,
                                            "any_TRAIN_vector_contains_2": TRAIN_HAS_TWO[0],
                                            "descriptive_only_seen_if_2_were_1": alt_seen})
                    if a == "TRAIN":
                        if not seen:
                            raise ScientificStop("exactness failure: a TRAIN row is not in K_TRAIN")
                        seen_txt = "TRAIN"
                    else:
                        seen_txt = "1" if seen else "0"
                    c = per[(a, lab)]
                    c["n"] += 1
                    c[f"part_{m[2]}"] += 1
                    c["all_zero"] += az
                    if a != "TRAIN":
                        arm = "seen" if seen else "unseen"
                        c[arm] += 1
                        c[f"{arm}_part_{m[2]}"] += 1
                        c[f"{arm}_all_zero"] += az
                        if not seen:
                            c["unseen_unknown_month"] += m[6] == "UNKNOWN_SENTINEL"
                            c["unseen_missing_family"] += m[7]
                            u_kids[a].add(kid)
                    kc = kid_counts[a].get(kid)
                    if kc is None:
                        kc = kid_counts[a][kid] = [0, 0]
                    kc[lab] += 1
                    kf.write(f"{g},{kid},{d.hex() if len(d) == 32 else hashlib.sha256(key).hexdigest()},{int(az)},{seen_txt}\n")
                n = len(labs)
                gid += n
                n_file += n
            coverage.append({"file": f["rel"], "expected_rows_g1": f["rows"], "keyed_rows": n_file,
                             "coverage": n_file / f["rows"] if f["rows"] else None})
    store.close()
    stats["global_distinct_keys"] = sum(len(b) for b in GI.values())
    stats["rows_keyed"] = gid
    return per, kid_counts, u_kids, coverage


def conflict_table(kid_counts, KT_rows_by_kid, sets):
    out = []

    def summarize(scope, counts):
        keys = len(counts)
        conf = [v for v in counts.values() if v[0] and v[1]]
        return {"scope": scope, "distinct_keys": keys, "conflict_keys": len(conf),
                "rows_in_conflict_keys": sum(v[0] + v[1] for v in conf)}

    T = kid_counts.get("TRAIN", {})
    out.append(summarize("TRAIN", T))
    for s in sets:
        if s == "TRAIN":
            continue
        U = kid_counts.get(s, {})
        out.append(summarize(s, U))
        merged = {k: list(v) for k, v in T.items()}
        for k, v in U.items():
            mv = merged.setdefault(k, [0, 0])
            mv[0] += v[0]
            mv[1] += v[1]
        out.append(summarize(f"TRAIN+{s}", merged))
    return out


# ------------------------------------------------------------------ run
def run(args, nf=NF, expected_total=EXPECTED_TOTAL, digest_fn=C.sha256_digest, check_g2=True):
    t_start = time.time()
    for p in (args.data_dir, args.g1_run, args.g2_work, args.g2_run, args.review_dir, args.out_root):
        ap = os.path.abspath(p)
        if os.path.splitdrive(ap)[0].upper() == "D:" or "android-malware-detection" in ap.lower():
            print("G3: STOP - forbidden path refused")
            return 3
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(args.out_root, f"{stamp}_G3LOCAL{('_' + args.label) if args.label else ''}")
    restricted = os.path.join(out, "restricted")
    tmpdir = os.path.join(restricted, "tmp")
    os.makedirs(tmpdir)
    S = {"task": "C1-R G3 LOCAL", "version": VERSION, "started_utc": now_utc(), "python": platform.python_version(),
         "pyarrow": pa.__version__, "nf": nf, "batch_rows": args.batch_rows, "block_rows": args.block_rows,
         "digest": getattr(digest_fn, "__name__", "custom"), "never_read": ["hash", "vt_count"],
         "phase_seconds": {}}
    stats = collections.Counter()
    code = 0
    C.TRACE_LOG.clear()
    VALUE2_ROWS.clear()
    TRAIN_HAS_TWO[0] = False
    S["deviations"] = ["G3-DEV-01: accepted feature domain {0,1,2}, values kept exactly as released (Manifest §3.3; §8.4)"]
    try:
        G = load_g2_module(args.g2_work)
        if not args.skip_pins and (platform.python_version() != "3.13.3" or pa.__version__ != "25.0.1"):
            raise C.IntegrityStop("environment does not match pin")
        anchors = {}
        for n, h in G.ANCHORS.items():
            p = os.path.join(args.review_dir, n)
            if not os.path.isfile(p) or C_sha(p) != h:
                raise C.IntegrityStop(f"anchor missing or changed: {n}")
            anchors[n] = h
        S["anchors"] = anchors
        g2S = verify_g2_run(args.g2_run) if check_g2 else None
        t = time.time()
        files, total = G.verify_inputs(args.data_dir, args.g1_run, nf, expected_total)
        S["inputs"] = {"files": len(files), "total_rows": total, "file_sha256": {f["rel"]: f["sha256"] for f in files}}
        meta, assign, mstar, mdigest, sizes = reconstruct_cohorts(
            G, files, G2_DIGEST if check_g2 else None, g2S["partition"]["sizes"] if check_g2 else None)
        S["cohorts"] = {"m_star": mstar, "membership_digest": mdigest, "matches_g2": check_g2, "sizes": sizes}
        S["phase_seconds"]["A_inputs_cohorts"] = round(time.time() - t, 1)

        t = time.time()
        vt, bad_vals, n_bad_cols = validate_features(files, nf, args.batch_rows)
        S["feature_validation"] = dict(vt, out_of_domain_value_counts=bad_vals, columns_with_out_of_domain=n_bad_cols,
                                       expected_cells=total * nf, accepted_domain=list(C.ALLOWED_VALUES),
                                       deviation="G3-DEV-01 (Manifest §3.3 amended; §8.4)")
        S["phase_seconds"]["B_validation"] = round(time.time() - t, 1)
        if (vt.get("null_cells") or vt.get("out_of_domain_cells") or vt.get("non_int8_columns")
                or vt.get("cells") != total * nf
                or vt.get("value_0_cells", 0) + vt.get("value_1_cells", 0) + vt.get("value_2_cells", 0) != total * nf):
            raise ScientificStop("feature validation failed (null / non-int8 / value outside {0,1,2})")

        t = time.time()
        meta_labels = [m[4] for m in meta]
        KT = build_k_train(files, assign, meta_labels, nf, args.batch_rows, args.block_rows, digest_fn, stats)
        S["k_train"] = {"train_rows": sizes.get("TRAIN"), "train_rows_keyed": stats["train_rows_keyed"],
                        "distinct_train_vectors": sum(len(b) for b in KT.values())}
        if stats["train_rows_keyed"] != sizes.get("TRAIN"):
            raise C.IntegrityStop("K_TRAIN coverage below 100%")
        S["phase_seconds"]["C1_k_train"] = round(time.time() - t, 1)

        t = time.time()
        keys_csv = os.path.join(restricted, "keys_seen.csv")
        per, kid_counts, u_kids, coverage = canonical_pass(files, assign, meta, nf, args.batch_rows,
                                                           args.block_rows, digest_fn, KT, tmpdir, keys_csv, stats)
        S["phase_seconds"]["C2_canonical_seen"] = round(time.time() - t, 1)
        S["coverage"] = {"per_file": coverage, "authoritative": all(c["keyed_rows"] == c["expected_rows_g1"] for c in coverage)
                         and stats["rows_keyed"] == total}
        if not S["coverage"]["authoritative"]:
            raise C.IntegrityStop("coverage below 100% (PROVISIONAL not permitted for primary findings)")
        S["exactness"] = {k: stats[k] for k in ("comparisons_train", "collisions_train", "comparisons_global",
                                                "collisions_global", "comparisons_seen", "collisions_seen")}
        S["global_distinct_vectors"] = stats["global_distinct_keys"]
        S["keys_table_sha256"] = C_sha(keys_csv)
        # ---- G3-DEV-01: explicit verification of rows containing value 2 (gid only in restricted/)
        S["k_train"]["train_rows_with_value_2"] = stats["train_rows_with_value_2"]
        if sum(r["cells_with_value_2"] for r in VALUE2_ROWS) != vt.get("value_2_cells", 0):
            raise C.IntegrityStop("value-2 cell accounting mismatch between validation and membership passes")
        S["value_2_rows"] = [{k: v for k, v in r.items() if k != "gid"} for r in VALUE2_ROWS]
        if VALUE2_ROWS:
            write_csv(os.path.join(restricted, "value2_rows.csv"), VALUE2_ROWS)

        # ---- aggregates
        sets = ["TRAIN"] + list(G.UNITS)
        rows_out = []
        est = {}
        for s in sets:
            tot = collections.Counter()
            for lab in (0, 1):
                c = per[(s, lab)]
                tot.update(c)
                rows_out.append(agg_row(s, G, LAB[lab], c, None))
            rows_out.append(agg_row(s, G, "all", tot, (len(kid_counts[s]), len(u_kids[s]) if s != "TRAIN" else None)))
            if s != "TRAIN":
                f0, f1 = per[(s, 0)]["n"], per[(s, 1)]["n"]
                u0, u1 = per[(s, 0)]["unseen"], per[(s, 1)]["unseen"]
                for lab in (0, 1):
                    c = per[(s, lab)]
                    if c["seen"] + c["unseen"] != c["n"]:
                        raise C.IntegrityStop(f"seen+unseen != full for {s}/{lab}")
                est[s] = {"primary": s in G.PRIMARY_UNITS, "F_benign": f0, "F_malware": f1, "U_benign": u0, "U_malware": u1,
                          "F_defined": f0 >= 1 and f1 >= 1, "U_defined": u0 >= 1 and u1 >= 1}
        total_assigned = sum(per[(s, lab)]["n"] for s in sets for lab in (0, 1))
        if total_assigned != total:
            raise C.IntegrityStop("row-count conservation failed")
        prim = {k: v for k, v in est.items() if v["primary"]}
        undefined = [k for k, v in prim.items() if not (v["F_defined"] and v["U_defined"])]
        S["estimability"] = {"units": est, "primary_undefined": undefined,
                             "E2": "ESTIMABLE" if not undefined else "NOT ESTIMABLE"}
        S["conflicts"] = conflict_table(kid_counts, None, sets)
        S["row_conservation"] = {"rows": total, "assigned": total_assigned}
        write_csv(os.path.join(out, "G3_AGGREGATE_TABLES.csv"), rows_out)
        write_csv(os.path.join(out, "g3_conflicts.csv"), S["conflicts"])
        write_csv(os.path.join(out, "g3_coverage.csv"), coverage)
        write_json(os.path.join(out, "g3_estimability.json"), S["estimability"])
        S["aggregates"] = rows_out
        if undefined:
            S["verdict"] = "STOP"
            S["stop_reason"] = f"E2 NOT ESTIMABLE: primary U arm(s) lack a class: {undefined} (Manifest §7.3, AM-03)"
            code = 2
        else:
            S["verdict"] = "PASS"
    except ScientificStop as e:
        S["verdict"], S["stop_reason"], code = "STOP", str(e), 2
    except C.IntegrityStop as e:
        S["verdict"], S["stop_reason"], code = "STOP (INTEGRITY)", str(e), 3
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)   # packed vector store: temp only
    S["temp_vector_store_deleted"] = not os.path.exists(tmpdir)
    S["io_trace"] = {"passes_files": len(C.TRACE_LOG), "unauthorized_reads": sum(x["unauthorized"] for x in C.TRACE_LOG),
                     "reads_touching_hash_or_vt_count": sum(x["touch_forbidden"] for x in C.TRACE_LOG)}
    S["peak_working_set_bytes"] = C.peak_rss_bytes()
    S["elapsed_seconds"] = round(time.time() - t_start, 1)
    S["finished_utc"] = now_utc()
    write_json(os.path.join(out, "g3_summary.json"), {k: v for k, v in S.items() if k != "aggregates"})
    write_report(out, S)
    write_inputs(out, args, S)
    write_sums(out)
    msg = f"G3 LOCAL: {S['verdict']}" + (f" - {S.get('stop_reason')}" if S.get("stop_reason") else "")
    print(msg.encode("ascii", "replace").decode("ascii"))
    print(f"out={out}")
    return code


def agg_row(s, G, lab, c, distinct):
    n = c["n"]
    r = {"set": s, "cohort": G.cohort_of(s), "year": s.split("_")[-1] if s != "TRAIN" else "2013-2014",
         "primary_unit": s in G.PRIMARY_UNITS, "label": lab, "n_full": n,
         "n_seen": c["seen"] if s != "TRAIN" else None, "n_unseen": c["unseen"] if s != "TRAIN" else None,
         "pct_seen": round(100 * c["seen"] / n, 3) if n and s != "TRAIN" else None,
         "pct_unseen": round(100 * c["unseen"] / n, 3) if n and s != "TRAIN" else None,
         "all_zero_full": c["all_zero"], "all_zero_unseen": c["unseen_all_zero"] if s != "TRAIN" else None,
         "train_partition_full": c["part_train"], "test_partition_full": c["part_test"],
         "train_partition_unseen": c["unseen_part_train"] if s != "TRAIN" else None,
         "test_partition_unseen": c["unseen_part_test"] if s != "TRAIN" else None,
         "unknown_month_unseen": c["unseen_unknown_month"] if s != "TRAIN" else None,
         "missing_family_unseen": c["unseen_missing_family"] if s != "TRAIN" else None,
         "distinct_vectors_full": distinct[0] if distinct else None,
         "distinct_vectors_unseen": distinct[1] if distinct else None}
    return r


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)


def write_inputs(out, args, S):
    lines = [f"{h} *{rel}" for rel, h in sorted(S.get("inputs", {}).get("file_sha256", {}).items())]
    for p, tag in ((os.path.join(args.data_dir, "LOCAL_SHA256SUMS.txt"), "Baseline/LOCAL_SHA256SUMS.txt"),
                   (os.path.join(args.g1_run, "inventory.json"), "G1/inventory.json"),
                   (os.path.join(args.g1_run, "SHA256SUMS.txt"), "G1/SHA256SUMS.txt"),
                   (os.path.join(args.g2_run, "g2_summary.json"), "G2/g2_summary.json"),
                   (os.path.join(args.g2_run, "SHA256SUMS.txt"), "G2/SHA256SUMS.txt"),
                   (os.path.join(args.g2_work, "src", "g2_analysis.py"), "G2/src/g2_analysis.py")):
        if os.path.isfile(p):
            lines.append(f"{C_sha(p)} *{tag}")
    for n, h in S.get("anchors", {}).items():
        lines.append(f"{h} *review/{n}")
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("g3core.py", "g3_main.py"):
        lines.append(f"{C_sha(os.path.join(here, f))} *src/{f}")
    with open(os.path.join(out, "INPUT_SHA256.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def write_sums(out):
    lines = []
    for root, _, fs in os.walk(out):
        for f in sorted(fs):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out).replace("\\", "/")
            if rel != "SHA256SUMS.txt":
                lines.append(f"{C_sha(p)} *{rel}")
    with open(os.path.join(out, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sorted(lines, key=lambda s: s.split("*", 1)[1])) + "\n")


def write_report(out, S):
    L = ["# C1-R G3 LOCAL — Run Report (machine-generated)", "",
         f"- **Verdict:** **{S.get('verdict')}**" + (f" — {S['stop_reason']}" if S.get("stop_reason") else ""),
         f"- v{VERSION}; Python {S['python']}; pyarrow {S['pyarrow']}; batch {S['batch_rows']} rows; block {S['block_rows']}",
         f"- Elapsed {S.get('elapsed_seconds')} s; peak working set {(S.get('peak_working_set_bytes') or 0) / 2**20:.0f} MiB",
         "- Aggregates only. No sample hash, feature vector or row-level membership record is in this report.", ""]
    if "feature_validation" in S:
        v = S["feature_validation"]
        L += ["## Feature validation", "", f"- cells validated: {v.get('cells', 0):,} of expected {v['expected_cells']:,}; "
              f"null {v.get('null_cells', 0)}; non-int8 columns {v.get('non_int8_columns', 0)}; out-of-domain {v.get('out_of_domain_cells', 0)}; "
              f"value counts 0/1/2 = {v.get('value_0_cells', 0):,}/{v.get('value_1_cells', 0):,}/{v.get('value_2_cells', 0):,} "
              f"(accepted domain {v.get('accepted_domain')}, {v.get('deviation')})", ""]
    if S.get("value_2_rows") is not None:
        L += ["## Rows containing value 2 (G3-DEV-01)", "", "| Unit | Label | Partition | cells=2 | seen (exact) | any TRAIN vector has a 2 | descriptive only: seen if 2->1 |",
              "|---|---|---|---|---|---|---|"]
        for r in S["value_2_rows"]:
            L.append(f"| {r['unit']} | {r['label']} | {r['partition']} | {r['cells_with_value_2']} | {r['seen_exact']} | "
                     f"{r['any_TRAIN_vector_contains_2']} | {r['descriptive_only_seen_if_2_were_1']} |")
        L.append("")
    if "estimability" in S:
        L += ["## Estimability (U arm needs >=1 benign and >=1 malware)", "",
              "| Unit | primary | F benign | F malware | U benign | U malware | U defined |", "|---|---|---|---|---|---|---|"]
        for k, v in S["estimability"]["units"].items():
            L.append(f"| {k} | {v['primary']} | {v['F_benign']:,} | {v['F_malware']:,} | {v['U_benign']:,} | {v['U_malware']:,} | "
                     f"{'YES' if v['U_defined'] else 'NO'} |")
        L += ["", f"**E2: {S['estimability']['E2']}**", ""]
    with open(os.path.join(out, "G3_RUN_REPORT.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def parse(argv=None):
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=v)
    ap.add_argument("--batch-rows", type=int, default=8192)
    ap.add_argument("--block-rows", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--skip-pins", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse()))
