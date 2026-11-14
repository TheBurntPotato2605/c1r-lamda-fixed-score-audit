"""Scientific source parity of the published pipeline with the frozen C1-R implementation.

Usage: python pipeline/verify_source_parity.py        (standard library only)

Checks, from `pipeline/frozen/frozen_source_manifest.json` (written once from the original private sources):
  1. every published frozen file has the recorded published SHA-256;
  2. every published frozen file has the same abstract syntax tree as the ORIGINAL file once string literals that
     held private machine paths are replaced by a placeholder (`original_normalized_ast_sha256_by_python`, recorded for Python 3.12 and 3.13), i.e. the
     scientific code is unchanged; files with `byte_identical_to_original` are literally the original bytes;
  3. the original files' SHA-256 are the ones recorded in the manifests of all 20 original runs (V6 for run 1,
     V6.1 for runs 2-5; the gate files that differ between V6 and V6.1 are not part of this package);
  4. the public job `pipeline/public/job_public.py` performs the same ordered scientific call sequence as the
     original `g4_job.py` (seeding, adapter, verified dataset, TRAIN load, fit, contract, save, reload, two scoring
     passes, seal, identity of passes).
Exit code 0 only if everything holds.
"""
import ast
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
SCI = {"random.seed", "np.random.seed", "adapter_cls", "adapter.fit", "adapter.resolved", "adapter.check_contract",
       "adapter.save", "adapter.load", "adapter.predict", "Dataset", "DataSpec", "ds.load_train", "ds.iter_eval",
       "acc.add", "acc.finalize", "ScoreAccumulator", "seal_digest", "write_table", "assert_passes_identical"}
PLACEHOLDER = "<PRIVATE-PATH-NOT-PUBLISHED>"


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Norm(ast.NodeTransformer):
    def visit_Constant(self, n):
        if isinstance(n.value, str) and ("ADMIN" in n.value or PLACEHOLDER in n.value):
            return ast.copy_location(ast.Constant(value="<PATH>"), n)
        return n


def norm_ast_hash(text):
    return sha(ast.dump(Norm().visit(ast.parse(text)), include_attributes=False).encode())


def dotted(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        d = dotted(n.value)
        return d + "." + n.attr if d else None
    return None


def sequence(path):
    t = ast.parse(open(path, encoding="utf-8").read())
    main = [n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    calls = [(n.end_lineno, n.end_col_offset, dotted(n.func)) for n in ast.walk(main) if isinstance(n, ast.Call) and dotted(n.func) in SCI]
    return [d for _, _, d in sorted(calls)]


def main():
    fm = json.load(open(os.path.join(FROZEN, "frozen_source_manifest.json")))
    prov = fm.pop("_provenance")
    ok, rows, note_ast = True, [], []
    for name, m in sorted(fm.items()):
        raw = open(os.path.join(FROZEN, "src", name), "rb").read()
        c1 = sha(raw) == m["published_sha256"]
        rec_ast = m["original_normalized_ast_sha256_by_python"].get("%d.%d" % sys.version_info[:2])
        if rec_ast is None:
            c2 = True       # ast.dump differs between Python versions; hashes recorded for 3.12 and 3.13 only
            note_ast.append(name)
        else:
            c2 = norm_ast_hash(raw.decode("utf-8")) == rec_ast
        c3 = not m["byte_identical_to_original"] or sha(raw) == m["original_sha256"]
        runs = prov["run_manifest_source_sha256"]
        rec = [r.get(name) for r in runs.values()]
        c4 = name not in runs["run1_frozen_V6"] or all(x == m["original_sha256"] for x in rec)
        rows.append((name, c1, c2, c3, c4, "identical bytes" if m["byte_identical_to_original"] else "path literals neutralised"))
        ok &= c1 and c2 and c3 and c4
    print("%-16s %-9s %-10s %-9s %-11s %s" % ("file", "sha256", "ast==orig", "bytes", "in-run-mf", "note"))
    for r in rows:
        print("%-16s %-9s %-10s %-9s %-11s %s" % (r[0], *("ok" if x else "FAIL" for x in r[1:5]), r[5]))
    if note_ast:
        print("NOTE: AST comparison skipped on Python %d.%d (recorded for 3.12 and 3.13); byte/hash checks were enforced" % sys.version_info[:2])
    seq = sequence(os.path.join(HERE, "public", "job_public.py"))
    c = seq == prov["original_g4_job_scientific_call_sequence"]
    print("public job scientific call sequence == original g4_job (%d calls): %s" % (len(seq), "ok" if c else "FAIL"))
    ok &= c
    print("SOURCE PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
