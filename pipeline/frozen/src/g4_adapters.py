"""C1-R G4 model adapters: exact frozen specifications (Manifest v0.2.4 §5.2-§5.4, approved under U-8).

Each adapter: fit(X_int8_train, y, seed) -> model (TRAIN rows only; internal validation/calibration drawn
inside TRAIN), save(model, dir), load(dir), predict(model, X_int8_batch) -> native-dtype probability of
label 1, resolved(model) -> dict, check_contract(resolved, seed) -> raises ContractFailure.

`overrides` exist ONLY for synthetic tests (e.g. device='cpu', fewer rounds); the runner rejects any
override in real mode. Feature values are passed exactly (int8 {0,1,2} -> float32/float64 casts only).
"""
import json
import os
import warnings

import numpy as np

import g4_common as K

ROW_BLOCK = 8192
EXPECTED_CUDA_ORDINAL = 0          # the single approved GPU (RTX 3050 Laptop, verified by provenance at run time)


def normalize_device(spec):
    """Canonical (kind, ordinal) for a device string. 'cuda' and 'cuda:0' -> ('cuda', 0); 'cpu' -> ('cpu', None).
    Anything else (e.g. 'gpu', 'cuda:1', 'sycl', '') -> ValueError, so unexpected devices are never accepted."""
    s = str(spec).strip().lower()
    if s == "cpu":
        return ("cpu", None)
    if s == "cuda":
        return ("cuda", 0)
    if s.startswith("cuda:") and s[5:].isdigit():
        return ("cuda", int(s[5:]))
    raise ValueError(f"unrecognized device specification: {spec!r}")


def device_matches(requested, resolved):
    """True only if the resolved device is the same kind AND the same CUDA ordinal as requested, and a CUDA
    request resolves to the approved ordinal. Silent CPU fallback or another GPU is rejected."""
    try:
        rq, rs = normalize_device(requested), normalize_device(resolved)
    except ValueError:
        return False
    if rq[0] == "cuda" and rq[1] != EXPECTED_CUDA_ORDINAL:
        return False
    return rq == rs


# ===================================================================== Linear SVM (calibrated)
class SVMAdapter:
    name = "svm"
    score_dtype = np.float64
    needs_gpu = False
    FROZEN_SVC = dict(penalty="l2", loss="squared_hinge", C=1.0, dual=False, tol=1e-4, max_iter=10000,
                      fit_intercept=True, intercept_scaling=1, class_weight=None)
    FROZEN_CAL = dict(method="sigmoid", n_splits=5, shuffle=False, ensemble=True)

    def __init__(self, overrides=None):
        self.overrides = dict(overrides or {})
        self.diag = {}

    @staticmethod
    def to_csr64(X):
        import scipy.sparse as sp
        blocks = [sp.csr_matrix(X[i:i + ROW_BLOCK], dtype=np.float64) for i in range(0, len(X), ROW_BLOCK)]
        return sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, X.shape[1]))

    def fit(self, X, y, seed, monitor=None):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.model_selection import StratifiedKFold
        from sklearn.svm import LinearSVC
        if len(X) <= X.shape[1]:
            raise K.ContractFailure("predeclared dual contingency (N_TRAIN <= n_features) would apply; not expected")
        Xc = self.to_csr64(X)
        self.diag["train_nnz"] = int(Xc.nnz)
        svc = LinearSVC(**self.FROZEN_SVC, random_state=seed)
        model = CalibratedClassifierCV(estimator=svc, method="sigmoid",
                                       cv=StratifiedKFold(n_splits=5, shuffle=False), ensemble=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(Xc, y)
        self.diag["convergence_warnings"] = sum(issubclass(x.category, ConvergenceWarning) for x in w)
        return model

    def save(self, model, d):
        import joblib
        p = os.path.join(d, "model_svm.joblib")
        joblib.dump(model, p)
        return [p]

    def load(self, d):
        import joblib
        return joblib.load(os.path.join(d, "model_svm.joblib"))

    def predict(self, model, Xb):
        p = model.predict_proba(self.to_csr64(Xb))[:, list(model.classes_).index(1)]
        return np.ascontiguousarray(p, dtype=np.float64)

    def resolved(self, model):
        pr = model.get_params(deep=True)
        cv = pr["cv"]
        out = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in pr.items()}
        out.update({"cv_class": type(cv).__name__, "cv_n_splits": cv.n_splits, "cv_shuffle": cv.shuffle,
                    "n_calibrated_classifiers": len(model.calibrated_classifiers_),
                    "classes_": [int(c) for c in model.classes_], **self.diag})
        return out

    def check_contract(self, r, seed):
        exp = {f"estimator__{k}": v for k, v in self.FROZEN_SVC.items()}
        exp.update({"estimator__random_state": seed, "method": "sigmoid", "ensemble": True, "cv_class": "StratifiedKFold",
                    "cv_n_splits": 5, "cv_shuffle": False, "n_calibrated_classifiers": 5, "classes_": [0, 1]})
        bad = {k: (r.get(k), v) for k, v in exp.items() if r.get(k) != v}
        if bad:
            raise K.ContractFailure(f"SVM resolved configuration differs from frozen spec: {bad}")


# ===================================================================== LightGBM
def parse_lgbm_parameters(model_str):
    """Parse the 'parameters:' block of a LightGBM model string into {name: value-string}."""
    out, inside = {}, False
    for line in model_str.splitlines():
        if line.strip() == "parameters:":
            inside = True
            continue
        if inside:
            if line.strip() == "end of parameters":
                break
            s = line.strip()
            if s.startswith("[") and s.endswith("]") and ": " in s:
                k, v = s[1:-1].split(": ", 1)
                out[k] = v
    return out


class LGBMAdapter:
    name = "lgbm"
    score_dtype = np.float64
    needs_gpu = False
    FROZEN = dict(boosting_type="gbdt", objective="binary", n_estimators=5000, learning_rate=0.02, num_leaves=256,
                  max_depth=-1, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_alpha=1.5, reg_lambda=1.5, max_bin=255, n_jobs=8, deterministic=True, force_row_wise=True,
                  metric="auc", verbose=-1)
    SPLIT = dict(test_size=0.1, shuffle=True, random_state=42)
    ES = dict(stopping_rounds=100, first_metric_only=True)

    def __init__(self, overrides=None):
        self.overrides = dict(overrides or {})
        self.diag = {}

    def fit(self, X, y, seed, monitor=None):
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, **self.SPLIT)   # TRAIN-internal only
        self.diag.update({"n_fit": int(len(y_tr)), "n_val": int(len(y_val))})
        params = dict(self.FROZEN, random_state=seed)
        params.update(self.overrides)
        m = lgb.LGBMClassifier(**params)
        evals = {}
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(**self.ES), lgb.record_evaluation(evals)])
        self.diag.update({"best_iteration_": int(m.best_iteration_),
                          "trees_built": int(m.booster_.num_trees()),
                          "early_stopped": int(m.booster_.num_trees()) < params["n_estimators"],
                          "val_auc_trace_len": len(next(iter(next(iter(evals.values())).values()))) if evals else 0})
        self._evals = evals
        return m

    def save(self, model, d):
        import joblib
        p1, p2 = os.path.join(d, "model_lgbm.joblib"), os.path.join(d, "model_lgbm.txt")
        joblib.dump(model, p1)
        model.booster_.save_model(p2)
        with open(os.path.join(d, "lgbm_validation_trace.json"), "w", encoding="utf-8") as fh:
            json.dump(self._evals, fh)                 # TRAIN-internal validation AUC only
        return [p1, p2]

    def load(self, d):
        import joblib
        return joblib.load(os.path.join(d, "model_lgbm.joblib"))

    def predict(self, model, Xb):
        p = model.predict_proba(Xb, num_iteration=model.best_iteration_)[:, 1]
        return np.ascontiguousarray(p, dtype=np.float64)

    def resolved(self, model):
        out = {"model_string_parameters": parse_lgbm_parameters(model.booster_.model_to_string()),
               "sklearn_params": {k: v for k, v in model.get_params().items()
                                  if isinstance(v, (int, float, str, bool, type(None)))},
               "classes_": [int(c) for c in model.classes_], "n_features_in_": int(model.n_features_in_)}
        out.update(self.diag)
        return out

    @staticmethod
    def expected_resolved(seed):
        """Exact expected resolved LightGBM 4.6.0 parameters for the frozen spec (aliases per
        docs/Parameters.rst v4.6.0; derived seeds per config.cpp + random.h)."""
        e = {"boosting": "gbdt", "objective": "binary", "learning_rate": 0.02, "num_leaves": 256, "max_depth": -1,
             "min_data_in_leaf": 30, "min_sum_hessian_in_leaf": 0.001, "min_gain_to_split": 0.0,
             "bagging_fraction": 0.8, "bagging_freq": 1, "pos_bagging_fraction": 1.0, "neg_bagging_fraction": 1.0,
             "feature_fraction": 0.8, "feature_fraction_bynode": 1.0, "lambda_l1": 1.5, "lambda_l2": 1.5,
             "max_bin": 255, "bin_construct_sample_cnt": 200000, "seed": seed, "deterministic": 1,
             "force_row_wise": 1, "num_threads": 8, "data_sample_strategy": "bagging"}
        e.update(K.lgbm_derived_seeds(seed))
        return e

    def check_contract(self, r, seed):
        ps = r["model_string_parameters"]
        bad = {}
        for k, v in self.expected_resolved(seed).items():
            got = ps.get(k)
            if got is None:
                bad[k] = ("<absent>", v)
                continue
            if isinstance(v, str):
                ok = got == v
            else:
                try:
                    ok = float(got) == float(v)
                except ValueError:
                    ok = False
            if not ok:
                bad[k] = (got, v)
        if r["classes_"] != [0, 1]:
            bad["classes_"] = (r["classes_"], [0, 1])
        if bad:
            raise K.ContractFailure(f"LightGBM resolved configuration differs from frozen spec: {bad}")


# ===================================================================== XGBoost
class _BatchIter:
    """xgboost.DataIter yielding float32 TRAIN batches (zeros remain values: missing=NaN default)."""

    def __new__(cls, X, y, batch):
        import xgboost as xgb

        class It(xgb.DataIter):
            def __init__(s):
                s.i = 0
                super().__init__()

            def next(s, input_data):
                if s.i >= len(X):
                    return False
                sl = slice(s.i, s.i + batch)
                input_data(data=X[sl].astype(np.float32), label=y[sl].astype(np.float32))
                s.i += batch
                return True

            def reset(s):
                s.i = 0
        return It()


def expected_xgb_cuts(col_min, col_max):
    """Cut values XGBoost (3.3.0, hist, max_bin >= #unique) produces for a feature whose TRAIN values are
    {min, max} subset of {0,1}: [-inf] + (max if max > min) + [max + |max| + 1e-5] (float32).
    Verified empirically on CPU by tests/test_adapters.py; GPU equality is checked by the V-4 smoke test."""
    out = [np.float32(-np.inf)]
    if col_max > col_min:
        out.append(np.float32(col_max))
    out.append(np.float32(col_max) + np.float32(abs(col_max)) + np.float32(1e-5))
    return np.asarray(out, dtype=np.float32)


class XGBAdapter:
    name = "xgb"
    score_dtype = np.float32
    FROZEN = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 12, "eta": 0.05,
              "tree_method": "hist", "device": "cuda", "max_bin": 256, "nthread": 8, "verbosity": 1}
    ROUNDS = 3000
    BATCH = 16384

    def __init__(self, overrides=None, container="quantile"):
        if container not in ("quantile", "dmatrix"):
            raise ValueError("container must be 'quantile' (D-2) or 'dmatrix' (literal frozen)")
        self.overrides = dict(overrides or {})
        self.container = container
        self.diag = {"container": container}

    @property
    def device(self):
        return self.overrides.get("device", self.FROZEN["device"])

    @property
    def needs_gpu(self):
        return normalize_device(self.device)[0] == "cuda"

    def params(self, seed):
        p = dict(self.FROZEN, seed=seed)
        p.update({k: v for k, v in self.overrides.items() if k != "rounds"})
        return p

    def fit(self, X, y, seed, monitor=None):
        import xgboost as xgb
        p = self.params(seed)
        if self.container == "quantile":
            dtrain = xgb.QuantileDMatrix(_BatchIter(X, y, self.BATCH), max_bin=p["max_bin"])
        else:
            dtrain = xgb.DMatrix(X.astype(np.float32), label=y.astype(np.float32))
        rounds = self.overrides.get("rounds", self.ROUNDS)
        booster = xgb.train(p, dtrain, num_boost_round=rounds)
        self.diag["cuts_check"] = self.check_cuts(dtrain, X, p)
        return booster

    def check_cuts(self, dtrain, X, p):
        """Cuts must equal (a) the explicit rule for {0,1} features and (b) the cuts a plain DMatrix
        builds on a 2-row proxy holding each feature's TRAIN minimum and maximum (same device/params)."""
        import xgboost as xgb
        indptr, values = dtrain.get_quantile_cut()
        mn, mx = X.min(axis=0), X.max(axis=0)
        if mx.max() > 1:
            raise K.ContractFailure("TRAIN contains a value > 1; cut rule not applicable (STOP)")
        rule_bad = [f for f in range(X.shape[1])
                    if not np.array_equal(values[indptr[f]:indptr[f + 1]].astype(np.float32),
                                          expected_xgb_cuts(int(mn[f]), int(mx[f])))]
        proxy = xgb.DMatrix(np.vstack([mn, mx]).astype(np.float32), label=np.array([0, 1], np.float32))
        xgb.train(dict(p), proxy, num_boost_round=1)
        pi, pv = proxy.get_quantile_cut()
        proxy_equal = np.array_equal(pi, indptr) and np.array_equal(pv.astype(np.float32), values.astype(np.float32))
        res = {"features_violating_rule": len(rule_bad), "proxy_dmatrix_cuts_equal": bool(proxy_equal)}
        if rule_bad or not proxy_equal:
            raise K.ContractFailure(f"XGBoost quantile cuts not exact for the {{0,1}} TRAIN domain: {res}")
        return res

    def save(self, booster, d):
        p1, p2 = os.path.join(d, "model_xgb.ubj"), os.path.join(d, "model_xgb_config.json")
        booster.save_model(p1)
        with open(p2, "w", encoding="utf-8") as fh:
            fh.write(booster.save_config())
        return [p1, p2]

    def load(self, d):
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(os.path.join(d, "model_xgb.ubj"))
        b.set_param({"device": "cpu", "nthread": 8})   # canonical scoring: CPU predictor, fixed threads
        return b

    def predict(self, booster, Xb):
        import xgboost as xgb
        p = booster.predict(xgb.DMatrix(Xb.astype(np.float32)), output_margin=False)
        return np.ascontiguousarray(p, dtype=np.float32)

    def resolved(self, booster, config_json=None):
        c = json.loads(config_json or booster.save_config())
        L = c["learner"]
        gb = L["gradient_booster"]
        t = gb["tree_train_param"]
        out = {"objective": L["objective"]["name"], "eval_metric": [m["name"] for m in L.get("metrics", [])],
               "max_depth": t.get("max_depth"), "eta": t.get("eta"), "max_bin": t.get("max_bin"),
               "subsample": t.get("subsample"), "colsample_bytree": t.get("colsample_bytree"),
               "tree_method": gb["gbtree_train_param"]["tree_method"], "device": L["generic_param"]["device"],
               "seed": L["generic_param"]["seed"], "nthread": L["generic_param"]["nthread"],
               "num_boosted_rounds": int(booster.num_boosted_rounds()),
               "base_score": L["learner_model_param"]["base_score"]}
        out.update(self.diag)
        return out

    def check_contract(self, r, seed):
        exp_rounds = self.overrides.get("rounds", self.ROUNDS)
        checks = {"objective": r["objective"] == "binary:logistic", "eval_metric": r["eval_metric"] == ["logloss"],
                  "max_depth": r["max_depth"] == str(self.params(seed)["max_depth"]),
                  "eta": np.float32(float(r["eta"])) == np.float32(0.05),
                  "max_bin": r["max_bin"] == "256", "subsample": r["subsample"] == "1",
                  "colsample_bytree": r["colsample_bytree"] == "1", "tree_method": r["tree_method"] == "hist",
                  "device": device_matches(self.device, r["device"]), "seed": r["seed"] == str(seed),
                  "nthread": r["nthread"] == "8", "rounds": r["num_boosted_rounds"] == exp_rounds}
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise K.ContractFailure(f"XGBoost resolved configuration differs from frozen spec: {bad} {r}")


# ===================================================================== MLP (PyTorch)
def mlp_batch_sizes(n):
    """Frozen complete-coverage batch sizes (Manifest §5.4): 512-row chunks of the epoch permutation; if the
    remainder is 1 (n > 512) the final 513 rows are split 257 + 256. Sum = n; every size in [2, 512]."""
    if n < 2:
        raise K.ContractFailure("MLP batching requires N >= 2")
    if n <= 512:
        return [n]
    k = -(-n // 512)
    rem = n - 512 * (k - 1)
    if rem >= 2:
        return [512] * (k - 1) + [rem]
    return [512] * (k - 2) + [257, 256]


def mlp_batches(n, generator, torch):
    """Fresh torch.randperm(n, generator) per epoch, cut by mlp_batch_sizes(n)."""
    perm = torch.randperm(n, generator=generator)
    out, s = [], 0
    for b in mlp_batch_sizes(n):
        out.append(perm[s:s + b])
        s += b
    return out


def build_mlp(nf, torch):
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(nf, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, 1), nn.Sigmoid())


def mlp_param_count(nf):
    dims = [(nf, 1024), (1024, 512), (512, 256), (256, 128), (128, 1)]
    return sum(i * o + o for i, o in dims) + 2 * (1024 + 512 + 256 + 128)


class MLPAdapter:
    name = "mlp"
    score_dtype = np.float32
    EPOCHS = 20
    ADAM = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    INFER_BATCH = 4096

    def __init__(self, overrides=None):
        self.overrides = dict(overrides or {})
        self.diag = {}

    @property
    def device(self):
        return self.overrides.get("device", "cuda")

    @property
    def needs_gpu(self):
        return normalize_device(self.device)[0] == "cuda"

    def _torch(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(8)
        return torch

    def fit(self, X, y, seed, monitor=None):
        torch = self._torch()
        dev = torch.device(self.device)
        if dev.type == "cuda":
            if not torch.cuda.is_available():
                raise K.IntegrityStop("CUDA requested for the MLP but unavailable (no silent CPU fallback)")
            dev = torch.device("cuda", EXPECTED_CUDA_ORDINAL if dev.index is None else dev.index)
            self.diag["train_gpu_name"] = torch.cuda.get_device_name(dev)
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = build_mlp(X.shape[1], torch).to(dev)
        self.diag["train_device_resolved"] = str(next(model.parameters()).device)
        opt = torch.optim.Adam(model.parameters(), **self.ADAM)
        crit = torch.nn.BCELoss(reduction="mean")
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        n = len(X)
        yt = torch.from_numpy(y.astype(np.float32))
        epochs = self.overrides.get("epochs", self.EPOCHS)
        losses = []
        for _ in range(epochs):
            model.train()
            batches = mlp_batches(n, g, torch)
            cover = torch.sort(torch.cat(batches)).values
            if not torch.equal(cover, torch.arange(n)) or min(len(b) for b in batches) < 2 or max(len(b) for b in batches) > 512:
                raise K.ContractFailure("MLP batching coverage/size contract violated")
            tot = 0.0
            for idx in batches:
                ii = idx.numpy()
                xb = torch.from_numpy(X[ii]).to(dev).float()
                yb = yt[idx].to(dev)
                opt.zero_grad()
                p = model(xb).squeeze(1)
                if p.shape != yb.shape:
                    raise K.ContractFailure("MLP output/target shape mismatch")
                loss = crit(p, yb)
                if not torch.isfinite(loss):
                    raise K.TechnicalFailure("non-finite MLP training loss")
                loss.backward()
                opt.step()
                tot += float(loss) * len(ii)
            losses.append(tot / n)
        self.diag.update({"epochs": epochs, "epoch_mean_loss": losses, "batches_per_epoch": len(batches),
                          "last_batch_size": int(len(batches[-1]))})
        return model.to("cpu")

    def save(self, model, d):
        torch = self._torch()
        p1, p2 = os.path.join(d, "model_mlp.pt"), os.path.join(d, "model_mlp_repr.txt")
        torch.save(model.state_dict(), p1)
        with open(p2, "w", encoding="utf-8") as fh:
            fh.write(repr(model))
        return [p1, p2]

    def load(self, d, nf=None):
        torch = self._torch()
        sd = torch.load(os.path.join(d, "model_mlp.pt"), map_location="cpu", weights_only=True)
        nf = nf or sd["0.weight"].shape[1]
        m = build_mlp(nf, torch)
        m.load_state_dict(sd)
        m.eval()                                       # dropout off, BatchNorm running statistics (P7)
        return m

    def predict(self, model, Xb):
        torch = self._torch()
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(Xb), self.INFER_BATCH):
                out = model(torch.from_numpy(Xb[s:s + self.INFER_BATCH]).float())
                outs.append(out.squeeze(1).numpy())
        p = np.concatenate(outs) if outs else np.empty(0, np.float32)
        return np.ascontiguousarray(p, dtype=np.float32)

    def resolved(self, model):
        torch = self._torch()
        nf = model[0].in_features
        return {"repr": repr(model), "expected_repr": repr(build_mlp(nf, torch)),
                "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "expected_params": mlp_param_count(nf), "adam": {k: list(v) if isinstance(v, tuple) else v
                                                                for k, v in self.ADAM.items()},
                "loss": "BCELoss(mean)", "device_requested": self.device, **self.diag}

    def check_contract(self, r, seed):
        bad = []
        if r["repr"] != r["expected_repr"]:
            bad.append("architecture")
        if r["trainable_params"] != r["expected_params"]:
            bad.append("parameter count")
        if r.get("epochs") != self.overrides.get("epochs", self.EPOCHS):
            bad.append("epochs")
        if not device_matches(self.device, r.get("train_device_resolved", "")):
            bad.append(f"device (requested {self.device}, resolved {r.get('train_device_resolved')})")
        if bad:
            raise K.ContractFailure(f"MLP resolved configuration differs from frozen spec: {bad}")


ADAPTERS = {"svm": SVMAdapter, "lgbm": LGBMAdapter, "xgb": XGBAdapter, "mlp": MLPAdapter}
