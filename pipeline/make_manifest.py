"""Deterministic integrity manifest of the repository.

MANIFEST.sha256 lists "<sha256>  <path>" for every file tracked by git except MANIFEST.sha256 itself, sorted by path
(byte order), hashing the bytes stored in the working tree (the repository stores files byte-exact: `* -text`).
Usage: python pipeline/make_manifest.py            # rewrite MANIFEST.sha256
       python pipeline/make_manifest.py --check    # verify every entry and that the file list is complete
"""
import hashlib
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def entries():
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO).decode("utf-8").split("\0")
    files = sorted(f for f in files if f and f != "MANIFEST.sha256")
    return "".join("%s  %s\n" % (hashlib.sha256(open(os.path.join(REPO, f), "rb").read()).hexdigest(), f) for f in files)


def main():
    path = os.path.join(REPO, "MANIFEST.sha256")
    want = entries()
    if "--check" in sys.argv:
        have = open(path, encoding="utf-8").read()
        ok = have == want
        print("MANIFEST CHECK: %s (%d entries)" % ("PASS" if ok else "FAIL", want.count("\n")))
        return 0 if ok else 1
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(want)
    print("MANIFEST written (%d entries)" % want.count("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
