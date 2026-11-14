"""Deterministic integrity manifest of the repository.

In a Git checkout, verify tracked-file coverage and hashes, or generate after staging files.
In an extracted source ZIP, --check verifies all listed file hashes without requiring Git;
tracked-file completeness cannot be independently inferred without the Git index.
"""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
MANIFEST = 'MANIFEST.sha256'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tracked_files():
    raw = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO)
    return sorted(set(f for f in raw.decode('utf-8').split('\0') if f and f != MANIFEST))


def checked_path(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or '..' in p.parts or p.as_posix() != name
            or '\\' in name or ':' in name or name == MANIFEST or '.git' in p.parts):
        raise ValueError('invalid manifest path: ' + name)
    actual = REPO.joinpath(*p.parts)
    if actual.resolve() != actual or not actual.is_file():
        raise ValueError('missing or redirected file: ' + name)
    return actual


def parse_manifest():
    entries = {}
    for line in (REPO / MANIFEST).read_text(encoding='utf-8').splitlines():
        h, name = line.split('  ', 1)
        if not re.fullmatch(r'[0-9a-f]{64}', h) or name in entries:
            raise ValueError('invalid or duplicate manifest entry: ' + name)
        checked_path(name)
        entries[name] = h
    if not entries or list(entries) != sorted(entries):
        raise ValueError('empty or unsorted manifest')
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    git_checkout = (REPO / '.git').exists()
    try:
        if args.check:
            entries = parse_manifest()
            bad = [name for name, h in entries.items() if digest(checked_path(name)) != h]
            if git_checkout and set(tracked_files()) != set(entries):
                raise ValueError('manifest file list differs from Git index')
            if bad:
                raise ValueError('SHA-256 mismatch: ' + ', '.join(bad))
            scope = 'tracked-file coverage and hashes' if git_checkout else 'listed-file hashes only; Git index unavailable'
            print(f'MANIFEST CHECK: PASS ({len(entries)} entries; {scope})')
            return 0
        if not git_checkout:
            raise ValueError('generation requires a Git checkout; use --check for a source archive')
        files = tracked_files()
        if not files:
            raise ValueError('no tracked files')
        text = ''.join(f'{digest(checked_path(name))}  {name}\n' for name in files)
        (REPO / MANIFEST).write_text(text, encoding='utf-8', newline='\n')
        print(f'MANIFEST written ({len(files)} entries)')
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'MANIFEST CHECK: FAIL ({exc})', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
