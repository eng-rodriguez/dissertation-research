"""Pre-specified inference helpers for the primary estimand (06 issues 3 and 10, adopted 2026-10-10).

Pure functions of already-computed per-patient / per-fold Delta values. Written before protocol lock so
the outcome label is fixed in code before any confirmatory result exists.

  nadeau_bengio_ci   PRIMARY interval (R7): sigma^2 = (1/J + n_test/n_train) * S^2 over the J = 50
                     repeat x outer-fold means; CI = mean +/- t_{n_E - 1} * sigma.
  cluster_bootstrap_ci  SENSITIVITY interval (R8): percentile bootstrap over repeat-averaged Delta_p.
  classify           ordered, mutually exclusive taxonomy (R4, Q1); first matching rule applies.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

SESOI = 0.03
OUTCOMES = (
    "meaningfully superior",
    "positive, below SESOI",
    "positive, meaningful magnitude not established",
    "negative, within SESOI",
    "inferior",
    "equivalent / bounded null",
    "inconclusive",
)
GATE_FAILED = "inconclusive (validity gate failed)"


def nadeau_bengio_ci(delta_hat: float, fold_means, n_patients: int, level: float,
                     test_train_ratio: float = 0.25) -> tuple[float, float]:
    fm = np.asarray(fold_means, float)
    sigma = np.sqrt((1 / fm.size + test_train_ratio) * fm.var(ddof=1))
    q = stats.t.ppf(0.5 + level / 2, n_patients - 1)
    return delta_hat - q * sigma, delta_hat + q * sigma


def cluster_bootstrap_ci(delta_p, level: float, n_resamples: int = 10_000, seed: int = 20261010) -> tuple[float, float]:
    d = np.asarray(delta_p, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, (n_resamples, d.size))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def classify(ci95: tuple[float, float], ci90: tuple[float, float], gates_pass: bool = True, sesoi: float = SESOI) -> str:
    l95, u95 = ci95
    l90, u90 = ci90
    if l95 > sesoi:
        return OUTCOMES[0]
    if l95 > 0 and u90 < sesoi:
        return OUTCOMES[1]
    if l95 > 0:
        return OUTCOMES[2]
    if u95 < 0 and l90 > -sesoi:
        return OUTCOMES[3]
    if u95 < 0:
        return OUTCOMES[4]
    if l90 > -sesoi and u90 < sesoi:
        return OUTCOMES[5] if gates_pass else GATE_FAILED
    return OUTCOMES[6]
