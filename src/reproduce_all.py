"""Re-run every analysis that the published aggregates support and compare with the archived outputs.

Usage (from the repository root):
    python -I src/reproduce_all.py [--skip-figures]

Steps
  1. verify the SHA-256 manifest of this repository (MANIFEST.sha256);
  2. src/wp1_sensitivity.py  -> temporary directory; the CSV must be byte-identical to
     data/wp1_aggregates/WP1_S0_S4_RESULTS.csv and the JSON identical after masking five provenance labels
     (script hash, interpreter version, the label and hash of the score-table hash file, input listing);
  3. src/wp1_independent_check.py on that re-run (separate rational-arithmetic implementation);
  4. src/make_tables.py -> temporary directory; must equal outputs/tables/*;
  5. src/make_figures.py (needs matplotlib) -> temporary directory; plotted values must equal
     outputs/figures/plotted_values.json.  Figure PDFs are rendered with the local matplotlib/fonts and are not
     expected to be byte-identical across machines.
Exit code 0 only if every executed step passes. Nothing outside this repository is read.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def step(name, ok, detail=""):
    print("%-58s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILS.append(name)


def run(args, **kw):
    return subprocess.run([sys.executable, "-I"] + args, cwd=REPO, capture_output=True, text=True, **kw)


def main():
    skip_fig = "--skip-figures" in sys.argv
    # 1 manifest
    bad = []
    for ln in open(os.path.join(REPO, "MANIFEST.sha256"), encoding="utf-8"):
        if ln.strip():
            h, n = ln.strip().split("  ", 1)
            p = os.path.join(REPO, n)
            if not os.path.isfile(p) or sha(p) != h:
                bad.append(n)
    step("1 MANIFEST.sha256", not bad, "mismatches: %s" % bad if bad else "")

    with tempfile.TemporaryDirectory() as td:
        # 2 S0-S4
        out = os.path.join(td, "wp1")
        os.makedirs(out)
        r = run(["src/wp1_sensitivity.py", "--out-dir", out])
        step("2a wp1_sensitivity.py exit status (gates A-F)", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
        arch = os.path.join(REPO, "data", "wp1_aggregates")
        step("2b S0-S4 CSV byte-identical to archive", sha(os.path.join(out, "WP1_S0_S4_RESULTS.csv")) == sha(os.path.join(arch, "WP1_S0_S4_RESULTS.csv")))
        a = json.load(open(os.path.join(out, "WP1_S0_S4_RESULTS.json"), encoding="utf-8"))
        b = json.load(open(os.path.join(arch, "WP1_S0_S4_RESULTS.json"), encoding="utf-8"))
        for path in (("script", "sha256"), ("script", "python"), ("distinct_score_tables", "source"),
                     ("distinct_score_tables", "G5_AUTHORIZATION.json_sha256"), ("gates", "C_access_boundary", "inputs_opened")):
            for d in (a, b):
                x = d
                for k in path[:-1]:
                    x = x[k]
                x.pop(path[-1], None)
        step("2c S0-S4 JSON identical (5 provenance labels masked)", a == b)
        # 3 independent check
        r = run(["src/wp1_independent_check.py", "--results-dir", out])
        step("3 independent rational-arithmetic check", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
        # 4 tables
        tdir = os.path.join(td, "tables")
        r = run(["src/make_tables.py", "--out-dir", tdir])
        same = r.returncode == 0 and all(
            sha(os.path.join(tdir, f)) == sha(os.path.join(REPO, "outputs", "tables", f)) for f in sorted(os.listdir(tdir)))
        step("4 tables and bounds equal outputs/tables", same)
        # 5 figures
        if not skip_fig:
            fdir = os.path.join(td, "fig")
            r = run(["src/make_figures.py", os.path.join(REPO, "data", "g5_aggregates"), fdir, os.path.join(fdir, "v.json")])
            if r.returncode != 0 and "matplotlib" in r.stderr:
                print("%-58s SKIPPED (matplotlib not installed)" % "5 figures")
            else:
                ok = r.returncode == 0
                if ok:
                    x = json.load(open(os.path.join(fdir, "v.json"), encoding="utf-8"))
                    y = json.load(open(os.path.join(REPO, "outputs", "figures", "plotted_values.json"), encoding="utf-8"))
                    ok = x == y
                step("5 figures: plotted values equal archive", ok)
    print("RESULT:", "ALL PASS" if not FAILS else "FAILED: %s" % FAILS)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
