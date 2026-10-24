"""Primary estimand pieces (06 issue 2, R2): per-patient same-state Mann-Whitney AUROC and Delta.

AUROC_{p,s} is the Mann-Whitney AUROC (ties = 1/2) of patient p's IED vs non-IED epochs in state s, scored
by the outer-fold model that held p out; only states where p is evaluable enter; states are combined with
IED-count weights. Delta_{p,r} = AUROC_{p,r}(B2 + FC) - AUROC_{p,r}(B2); Delta_p = mean over repeats;
Delta-hat = mean over evaluable patients (equal weight). Scores are never pooled across repeats.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def mw_auroc(score: np.ndarray, y: np.ndarray) -> float:
    r = rankdata(score)
    n1 = int(y.sum())
    n0 = y.size - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def patient_auroc(score: np.ndarray, y: np.ndarray, state: np.ndarray, evaluable) -> float:
    """IED-count-weighted mean over the patient's evaluable states of the same-state AUROC."""
    num = den = 0.0
    for s in evaluable:
        k = state == s
        n_ied = int(y[k].sum())
        num += n_ied * mw_auroc(score[k], y[k])
        den += n_ied
    return num / den


def delta_summary(auroc: dict, model: str, ref: str, folds: dict) -> dict:
    """Delta from per-(repeat, patient) AUROCs.

    auroc[(model, repeat, eeg_id)] -> AUROC; folds[(repeat, eeg_id)] -> outer test fold.
    Returns Delta_p per patient, Delta-hat and the 50 fold-repeat means (cells with an evaluable patient).
    """
    keys = sorted({(r, p) for (m, r, p) in auroc if m == model})
    d_rp = {(r, p): auroc[(model, r, p)] - auroc[(ref, r, p)] for r, p in keys}
    pats = sorted({p for _, p in keys})
    reps = sorted({r for r, _ in keys})
    d_p = {p: float(np.mean([d_rp[(r, p)] for r in reps])) for p in pats}
    cells = {}
    for (r, p), d in d_rp.items():
        cells.setdefault((r, folds[(r, p)]), []).append(d)
    return {"delta_p": d_p, "delta_hat": float(np.mean(list(d_p.values()))),
            "fold_means": [float(np.mean(v)) for _, v in sorted(cells.items())], "n_e": len(pats),
            "n_cells": len(cells)}
