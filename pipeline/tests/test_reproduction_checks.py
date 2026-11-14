"""Lightweight tests of comparison and source-archive integrity checks (no data, training, or network)."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import c1r_e2e as E
import make_manifest as M


class CompareChecks(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        repo = root / 'repo'
        self.refs = repo / 'data/g5_aggregates'
        reference = repo / 'pipeline/reference'
        published_tables = repo / 'outputs/tables'
        self.ctx = SimpleNamespace(models=E.MODELS, runs=E.RUNS, work=str(root / 'work'),
            runs_dir=str(root / 'work/runs'), eval_dir=str(root / 'work/eval'), reports=str(root / 'work/reports'))
        for p in (self.refs, reference, published_tables, Path(self.ctx.eval_dir), Path(self.ctx.work) / 'tables'):
            p.mkdir(parents=True, exist_ok=True)
        self.patch1 = patch.object(E, 'REPO', str(repo))
        self.patch2 = patch.object(E, 'REFERENCE', str(reference))
        for p in (self.patch1, self.patch2):
            p.start()
            self.addCleanup(p.stop)
        self.pins = {}
        for m in E.MODELS:
            for r in E.RUNS:
                d = Path(E.slot_dir(self.ctx, m, r))
                d.mkdir(parents=True)
                # Mock bytes, not real Parquet; this tests hash comparison, not scoring.
                payload = f'synthetic {m} {r}'.encode()
                h = hashlib.sha256(payload).hexdigest()
                seal = hashlib.sha256(b'seal' + payload).hexdigest()
                (d / 'scores.parquet').write_bytes(payload)
                (d / 'scores.sha256').write_text(seal + '\n')
                E.jdump(d / 'job_result.json', {'state': 'SUCCEEDED', 'scores': {
                    'file_sha256': h, 'seal_sha256': seal, 'pass2_sha256': seal}})
                self.pins[f'{m}_r{r}'] = {'scores_file_sha256': h, 'seal_sha256': seal}
        E.jdump(reference / 'original_seals.json', {'slots': self.pins})
        for n in ('G5_per_run_metrics.json', 'G5_model_summary.json', 'G5_model_independent_counts.json', 'G5_E3.json'):
            for p in (self.refs, Path(self.ctx.eval_dir)):
                E.jdump(p / n, {'auc': {'exact': '1/2', 'value': 0.5}})
        for n in ('table4_cohorts.tex', 'table5_primary.tex', 'table6_sensitivity.tex'):
            for p in (published_tables, Path(self.ctx.work) / 'tables'):
                (p / n).write_bytes(b'table\n')

    def compare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            ok = E.stage_compare(self.ctx, **kwargs)
        return ok, E.jload(Path(self.ctx.reports) / 'compare.json')

    def test_exact_full_reference(self):
        ok, report = self.compare(require_exact=True)
        self.assertTrue(ok)
        self.assertTrue(report['reference_artifacts_exact'])
        self.assertEqual(report['n_score_files_equal'], 20)

    def test_custom_reference_not_full_exact(self):
        ok, report = self.compare(ref_dir=self.ctx.eval_dir)
        self.assertTrue(ok)
        self.assertEqual(report['status'], 'EXACT')
        self.assertFalse(report['reference_artifacts_exact'])

    def test_missing_table_not_full_exact(self):
        (Path(self.ctx.work) / 'tables/table4_cohorts.tex').unlink()
        ok, report = self.compare(require_exact=True)
        self.assertFalse(ok)
        self.assertEqual(report['status'], 'EXACT')
        self.assertFalse(report['reference_artifacts_exact'])

    def test_score_byte_tamper_fails(self):
        d = Path(E.slot_dir(self.ctx, 'svm', 1))
        (d / 'scores.parquet').write_bytes(b'changed bytes')
        ok, report = self.compare()
        self.assertFalse(ok)
        self.assertEqual(report['status'], 'EXACT')
        self.assertFalse(report['score_slot_integrity']['svm_r1'])
        self.assertFalse(report['reference_artifacts_exact'])

    def test_valid_nonmatching_score_is_only_aggregate_exact(self):
        d = Path(E.slot_dir(self.ctx, 'svm', 1))
        payload = b'fresh different score bytes'
        (d / 'scores.parquet').write_bytes(payload)
        job = E.jload(d / 'job_result.json')
        job['scores']['file_sha256'] = hashlib.sha256(payload).hexdigest()
        E.jdump(d / 'job_result.json', job)
        ok, report = self.compare()
        self.assertTrue(ok)
        self.assertEqual(report['n_score_files_equal'], 19)
        self.assertFalse(report['reference_artifacts_exact'])
        self.assertFalse(self.compare(require_exact=True)[0])


class ManifestChecks(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        p = patch.object(M, 'REPO', self.root)
        p.start()
        self.addCleanup(p.stop)
        (self.root / 'data.txt').write_bytes(b'data\n')
        self.valid = hashlib.sha256(b'data\n').hexdigest() + '  data.txt\n'
        (self.root / M.MANIFEST).write_text(self.valid)

    def check(self):
        with patch.object(sys, 'argv', ['make_manifest.py', '--check']), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return M.main()

    def test_source_archive(self):
        self.assertEqual(self.check(), 0)

    def test_tampered_file(self):
        (self.root / 'data.txt').write_bytes(b'changed')
        self.assertEqual(self.check(), 1)

    def test_duplicate_entry(self):
        (self.root / M.MANIFEST).write_text(self.valid * 2)
        self.assertEqual(self.check(), 1)

    def test_traversal_rejected(self):
        (self.root / M.MANIFEST).write_text(self.valid.replace('data.txt', '../data.txt'))
        self.assertEqual(self.check(), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
