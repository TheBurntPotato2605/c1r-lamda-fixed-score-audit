"""Run every automated check of this repository and print one summary table.

Usage: python pipeline/run_checks.py [--no-network] [--skip-synthetic]
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
    ok = [run("source parity (frozen sources)", ["pipeline/verify_source_parity.py"]),
          run("aggregate-only workflow (manifest, S0-S4, tables)", ["src/reproduce_all.py", "--skip-figures"])]
    if "--no-network" not in sys.argv:
        with tempfile.TemporaryDirectory() as td:
            ok.append(run("download dry run (remote metadata == pins)", ["pipeline/c1r_e2e.py", "download", "--dry-run", "--work-dir", td]))
    if "--skip-synthetic" not in sys.argv:
        ok.append(run("synthetic end-to-end", ["pipeline/tests/test_synthetic_e2e.py"]))
    print("ALL CHECKS PASS" if all(ok) else "SOME CHECKS FAILED")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
