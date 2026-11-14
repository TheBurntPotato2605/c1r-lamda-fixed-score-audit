"""Run every automated check of this repository and print one summary table.

Usage: python pipeline/run_checks.py [--no-network] [--skip-synthetic] [--with-figures]
The synthetic end-to-end test needs the pinned environment (numpy, pyarrow, scikit-learn, lightgbm, xgboost, torch).
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def run(name, args):
    r = subprocess.run([sys.executable, "-I"] + args, cwd=REPO, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1]
    print("%-52s %s  %s" % (name, "PASS" if r.returncode == 0 else "FAIL", tail[:90]))
    return r.returncode == 0


def main():
    aggregate_args = ["src/reproduce_all.py"]
    if "--with-figures" not in sys.argv:
        aggregate_args.append("--skip-figures")
        print("SKIP: figure rendering (use --with-figures to include)")
    ok = [run("source parity (frozen sources)", ["pipeline/verify_source_parity.py"]),
          run("aggregate-only workflow (manifest, S0-S4, tables)", aggregate_args),
          run("manifest check (make_manifest.py --check)", ["pipeline/make_manifest.py", "--check"]),
          run("comparison and archive integrity (lightweight)", ["-S", "-B", "pipeline/tests/test_reproduction_checks.py"])]
    if "--no-network" not in sys.argv:
        with tempfile.TemporaryDirectory() as td:
            ok.append(run("download dry run (remote metadata == pins)", ["pipeline/c1r_e2e.py", "download", "--dry-run", "--work-dir", td]))
    else:
        print("SKIP: network metadata check (--no-network)")
    if "--skip-synthetic" not in sys.argv:
        ok.append(run("synthetic end-to-end", ["pipeline/tests/test_synthetic_e2e.py"]))
        ok.append(run("operating-point metrics (synthetic)", ["-B", "pipeline/operating_points/tests/test_operating_points.py"]))
    else:
        print("SKIP: synthetic end-to-end and operating-point tests (--skip-synthetic)")
    print("ALL EXECUTED CHECKS PASS" if all(ok) else "SOME CHECKS FAILED")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
