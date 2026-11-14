"""C1-R G4 common definitions: frozen constants, run states, hashing, locked module loading.

Authority: Manifest v0.2.4 (ac8fee5f...), Protocol V2.2.1 (47fb8da6...), G0 approval record, G3-DEV-01.
Nothing in this package changes a frozen scientific choice. Real-data execution additionally requires an
explicit G4 authorization record (see g4_runner.check_authorization); without it the runner refuses.
"""
import hashlib
import importlib.util
import os

VERSION = "1.0.0"

# ---------------------------------------------------------------- frozen data facts (G2/G3, verified)
NF = 4561
TOTAL_ROWS = 1_008_381
TRAIN_ROWS = 172_328
EVAL_ROWS = TOTAL_ROWS - TRAIN_ROWS            # 836,053
MEMBERSHIP_DIGEST = "f4b62175292564f08288ce19e05db2974cf4659464146757b190c73cbd569bc8"
KEYS_TABLE_SHA256 = "cdc267890721e306cd6714540f3724bb5e303a63d76ee27f73a82d60e652551a"
SET_SIZES = {"TRAIN": 172328, "IID_2013": 9489, "IID_2014": 5797, "NEAR_2016": 109193, "NEAR_2017": 99144,
             "FAR_2018": 104292, "FAR_2019": 91050, "FAR_2020": 102073, "FAR_2021": 81155,
             "FAR_2022": 86416, "FAR_2023": 54354, "FAR_2024": 48427, "FAR_2025": 44663}
UNITS = ("IID_2013", "IID_2014", "NEAR_2016", "NEAR_2017") + tuple(f"FAR_{y}" for y in range(2018, 2026))
PRIMARY_UNITS = ("IID_2013", "IID_2014") + tuple(f"FAR_{y}" for y in range(2018, 2026))
IID_UNITS = ("IID_2013", "IID_2014")
FAR_UNITS = tuple(f"FAR_{y}" for y in range(2018, 2026))
ALLOWED_FEATURE_VALUES = (0, 1, 2)               # G3-DEV-01 (Manifest §3.3 amended)

# ---------------------------------------------------------------- locked upstream code (reused, not copied)
G2_ANALYSIS_SHA256 = "3b7615c844c1cda7804b11baaf44570396faaa0a73677ad6fdddeba74c85df38"   # g2_analysis.py v1.1.0
G3CORE_SHA256 = "95e22dab4b5efd277377eb2a15093858569403f32bd8f170ffe93d3f0844111c"        # g3core.py v1.1.0

REAL_PATHS = {
    "data_dir": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "g1_run": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "g2_module": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "g3core_module": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "keys_csv": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "review_dir": r"<PRIVATE-PATH-NOT-PUBLISHED>",
    "work_dir": r"<PRIVATE-PATH-NOT-PUBLISHED>",
}


# ---------------------------------------------------------------- run states (Manifest §8.2)
class G4Error(Exception):
    state = "FAILED-TECHNICAL"
    exit_code = 11


class NotAuthorized(G4Error):
    state = "NOT-AUTHORIZED"
    exit_code = 5


class Refused(G4Error):
    """Retry/slot rule refusal (e.g. a SUCCEEDED slot, or attempt > 2)."""
    state = "REFUSED"
    exit_code = 6


class IntegrityStop(G4Error):
    """Pre-fit STOP: inputs, digests, schema, domain, access scope, environment, preconditions."""
    state = "STOP-PREFIT"
    exit_code = 10


class TechnicalFailure(G4Error):
    state = "FAILED-TECHNICAL"
    exit_code = 11


class ContractFailure(G4Error):
    """Violation of the immutable prediction contract P1-P8 or of the frozen resolved configuration."""
    state = "FAILED-CONTRACT"
    exit_code = 12


class CapExceeded(G4Error):
    state = "CAP-EXCEEDED"
    exit_code = 13


class Interrupted(G4Error):
    state = "INTERRUPTED"
    exit_code = 14


RETRYABLE_STATES = ("FAILED-TECHNICAL", "INTERRUPTED")   # Manifest §8.2: <= 1 retry, identical config


# ---------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def forbidden_path(p):
    ap = os.path.abspath(p)
    return os.path.splitdrive(ap)[0].upper() == "D:" or "android-malware-detection" in ap.lower()


def load_locked_module(name, path, expected_sha):
    """Import an upstream module only if its bytes equal the frozen hash (G2/G3 provenance)."""
    if forbidden_path(path):
        raise IntegrityStop(f"forbidden path refused: {path}")
    if not os.path.isfile(path) or sha256_file(path) != expected_sha:
        raise IntegrityStop(f"locked module missing or changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lgbm_derived_seeds(seed):
    """LightGBM 4.6.0 Config::Set (src/io/config.cpp L259-268) + utils/random.h: when `seed` is set, six
    sub-seeds are drawn from an LCG x <- 214013*x + 2531011 (uint32), NextShort(0, 32767) =
    ((x >> 16) & 0x7FFF) % 32767. Used to assert the resolved bagging/feature-fraction seeds."""
    x = seed & 0xFFFFFFFF
    out = {}
    for name in ("data_random_seed", "bagging_seed", "drop_seed", "feature_fraction_seed",
                 "objective_seed", "extra_seed"):
        x = (214013 * x + 2531011) & 0xFFFFFFFF
        out[name] = ((x >> 16) & 0x7FFF) % 32767
    return out
