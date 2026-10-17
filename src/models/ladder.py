"""Comparator ladder models (06 issues 5-9, §4.1): feature assembly, R9 weights, learners, nested tuning, B0.

Patients are dicts {"id", "y" (0/1 per epoch), "s" (state per epoch), "evaluable" (states), "F": {name: array}}
where F holds the cached representations of src/features/extract.py plus "b2_count" (the salience count,
computed with the pilot-fixed z*). Every model in a contrast uses the same folds, weights, grid and seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import estimand as est  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastbin  # noqa: E402

fastbin.apply()  # speed only; thresholds bit-identical to scikit-learn 1.9.1 (tests/test_ladder.py)

HGB_GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, min_samples_leaf=msl, l2_regularization=l2)
            for lr in (0.05, 0.1) for ml in (15, 31) for msl in (20, 100) for l2 in (0, 1)]
LR_GRID = [dict(C=float(c)) for c in np.logspace(-3, 2, 8)]
MAX_ITER = 300

# Representation blocks per ladder model (06 issues 4-8). "b0" is the per-channel B1 model.
MODELS = {
    "b0": ["b1_channel"],
    "b1": ["b1"],
    "b2": ["b1", "b2_count", "b2_topo"],
    "b2_adj": ["b1", "b2_count", "b2_topo", "adj"],
    "b2_fcsurr": ["b1", "b2_count", "b2_topo", "fc_surr"],
    "b2_fc": ["b1", "b2_count", "b2_topo", "fc"],
    "b2_fc_delta": ["b1", "b2_count", "b2_topo", "fc", "fc_delta"],  # 4-s sensitivity only: exploratory delta rung
    "pc1_fc_only": ["fc"],
}


def simplicity_key(kind: str, c: dict) -> tuple:
    """Tie-break order for 'simpler configuration'. HGB as EV-1 (11 LOG-2026-10-10-EV1-SCOPE): smaller
    max_leaf_nodes, larger min_samples_leaf, larger l2, smaller learning rate. LR: smaller C."""
    if kind == "lr":
        return (c["C"],)
    return (c["max_leaf_nodes"], -c["min_samples_leaf"], -c["l2_regularization"], c["learning_rate"])


def design_matrix(p: dict, model: str) -> np.ndarray:
    """Epoch-level features (n x d), or for b0 the per-channel B1 rows (n x 19 x 24)."""
    if model == "b0":
        return p["F"]["b1"]
    parts = []
    for b in MODELS[model]:
        a = np.asarray(p["F"][b], dtype=np.float64)
        parts.append(a.reshape(len(a), -1))
    return np.hstack(parts)


def r9_weights(pats: list[dict]) -> np.ndarray:
    """06 R9: 1/n_p per epoch; negatives x pi+_s / pi-_s (patient-weighted state shares of this training set),
    both shares computed from the 1/n_p weights before any rescaling (as EV-1, src/design/ev1_coverage.py)."""
    w = np.concatenate([np.full(len(p["y"]), 1.0 / len(p["y"])) for p in pats])
    y = np.concatenate([p["y"] for p in pats])
    s = np.concatenate([p["s"] for p in pats])
    pos, neg = y == 1, y == 0
    ratio = {}
    for st in ("wake", "sleep"):
        pi_neg = w[neg & (s == st)].sum() / w[neg].sum()
        ratio[st] = (w[pos & (s == st)].sum() / w[pos].sum()) / pi_neg if pi_neg > 0 else 1.0
    for st, r in ratio.items():
        w[neg & (s == st)] *= r
    return w


def learner(kind: str, params: dict, seed: int):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=MAX_ITER, early_stopping=False, random_state=seed, **params)
    return make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=5000,  # L2 (the default)
                                                              random_state=seed, **params))


def fit(pats: list[dict], model: str, kind: str, params: dict, seed: int):
    X = [design_matrix(p, model) for p in pats]
    y = np.concatenate([p["y"] for p in pats])
    w = r9_weights(pats)
    if model == "b0":  # one row per (epoch, channel), each carrying its epoch's label and weight
        n_ch = X[0].shape[1]
        X = np.vstack([x.reshape(-1, x.shape[-1]) for x in X])
        y, w = np.repeat(y, n_ch), np.repeat(w, n_ch)
    else:
        X = np.vstack(X)
    m = learner(kind, params, seed)
    step = m.steps[-1][0] + "__sample_weight" if kind == "lr" else "sample_weight"
    return m.fit(X, y, **{step: w})


def score(m, p: dict, model: str) -> np.ndarray:
    """Epoch scores (decision function); b0: max over channels of the per-channel scores."""
    X = design_matrix(p, model)
    if model == "b0":
        n, n_ch, d = X.shape
        return m.decision_function(X.reshape(-1, d)).reshape(n, n_ch).max(axis=1)
    return m.decision_function(X)


def select(pats: list[dict], inner: dict, model: str, kind: str, seed: int) -> dict:
    """Nested inner CV (06 §4.1): the grid point with the highest mean within-patient same-state AUROC over
    the evaluable patients of the inner validation folds (equal weight per patient); ties -> simpler."""
    grid = HGB_GRID if kind == "hgb" else LR_GRID
    folds = sorted(set(inner.values()))
    best = None
    for c in grid:
        vals = []
        for j in folds:
            tr = [p for p in pats if inner[p["id"]] != j]
            m = fit(tr, model, kind, c, seed)
            vals += [est.patient_auroc(score(m, p, model), p["y"], p["s"], p["evaluable"])
                     for p in pats if inner[p["id"]] == j and p["evaluable"]]
        key = (-float(np.mean(vals)), simplicity_key(kind, c))
        if best is None or key < best[0]:
            best = (key, c)
    return best[1]


def outer_split(train: list[dict], test: list[dict], model: str, kind: str, seed: int,
                inner: dict | None = None, fixed: dict | None = None) -> tuple[dict, dict]:
    """Fit on the outer training set (tuned with inner folds, or with a fixed configuration) and return
    ({test eeg_id: AUROC_p} for evaluable test patients, the chosen configuration)."""
    cfg = fixed if fixed is not None else select(train, inner, model, kind, seed)
    m = fit(train, model, kind, cfg, seed)
    return ({p["id"]: est.patient_auroc(score(m, p, model), p["y"], p["s"], p["evaluable"])
             for p in test if p["evaluable"]}, cfg)
