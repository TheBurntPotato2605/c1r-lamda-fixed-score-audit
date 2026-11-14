"""Public execution interface (replaces the private authorization gate `g4_access.py` of the C1-R workspace).

The original C1-R runner required private authorization records, a global lock and an append-only ledger before
real data could be opened. Those records are governance artifacts of the original study, are not published, and
are not needed to compute anything: the gate only decides WHETHER a run may start, it never changes WHAT is
computed (see pipeline/PARITY.md). A researcher running this repository on their own machine is their own
execution authority, so this module accepts every data specification.

The frozen modules call exactly two things from this module: `assert_dataset_access(spec)` (g4_data.Dataset) and the
attribute `_GRANT` (g5_eval.build_metadata). Both are kept, as no-ops.
"""
_GRANT = None


def assert_dataset_access(spec):
    return None
