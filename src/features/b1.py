"""B1: per-channel waveform and spectral features on the analysis window (06 issue 5, v1.0).

24 features per channel (19 x 24 = 456), computed on the CAR window. Definitions where 06 leaves the detail
open are the defaults in protocol/2026-10-12-pipeline-implementation-spec.md §3 (approved before EXP-003).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.signal import welch

FS = 500
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "sigma": (11, 16), "beta": (13, 30),
         "low_gamma": (30, 45)}
TOTAL = (1, 45)
NAMES = (["ptp", "max_abs_slope", "line_length", "teager_mean", "kurtosis", "skewness", "rise_s", "fall_s",
          "zero_cross_rate", "hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
         + [f"abs_{b}" for b in BANDS] + [f"rel_{b}" for b in BANDS])


def _half_waves(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rise and fall durations (s) around each row's sample of maximum |x| (mean removed)."""
    y = x - x.mean(axis=1, keepdims=True)
    n = y.shape[1]
    peak = np.argmax(np.abs(y), axis=1)
    sign = np.signbit(y)
    rise, fall = np.empty(len(y)), np.empty(len(y))
    for r, p in enumerate(peak):
        cross = np.flatnonzero(sign[r, 1:] != sign[r, :-1]) + 1  # first sample after a sign change
        before, after = cross[cross <= p], cross[cross > p]
        rise[r] = (p - (before[-1] if before.size else 0)) / FS
        fall[r] = ((after[0] if after.size else n - 1) - p) / FS
    return rise, fall


def _hjorth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx, ddx = np.diff(x, axis=1), np.diff(x, n=2, axis=1)
    v0, v1, v2 = x.var(axis=1), dx.var(axis=1), ddx.var(axis=1)
    mob = np.sqrt(np.divide(v1, v0, out=np.zeros_like(v0), where=v0 > 0))
    mob_d = np.sqrt(np.divide(v2, v1, out=np.zeros_like(v1), where=v1 > 0))
    comp = np.divide(mob_d, mob, out=np.zeros_like(mob), where=mob > 0)
    return v0, mob, comp


def band_powers(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute and relative band power per row: Welch (Hann, 1-s segments, 50% overlap)."""
    f, p = welch(x, fs=FS, window="hann", nperseg=FS, noverlap=FS // 2, axis=-1)
    df = f[1] - f[0]
    ab = np.stack([p[:, (f >= lo) & (f < hi)].sum(axis=1) * df for lo, hi in BANDS.values()], axis=1)
    tot = p[:, (f >= TOTAL[0]) & (f < TOTAL[1])].sum(axis=1, keepdims=True) * df
    rel = np.divide(ab, tot, out=np.zeros_like(ab), where=tot > 0)
    return ab, rel


def b1(window: np.ndarray) -> np.ndarray:
    """Features (channels x 24) for one CAR window (channels x samples)."""
    x = np.asarray(window, dtype=np.float64)
    dx = np.diff(x, axis=1)
    psi = x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:]
    y = x - x.mean(axis=1, keepdims=True)
    zc = (np.signbit(y[:, 1:]) != np.signbit(y[:, :-1])).sum(axis=1) / (x.shape[1] / FS)
    flat = x.std(axis=1) == 0
    with np.errstate(all="ignore"):
        kurt = np.where(flat, 0.0, stats.kurtosis(x, axis=1, fisher=True, bias=False))
        skew = np.where(flat, 0.0, stats.skew(x, axis=1, bias=False))
    rise, fall = _half_waves(x)
    act, mob, comp = _hjorth(x)
    ab, rel = band_powers(x)
    cols = [np.ptp(x, axis=1), np.abs(dx).max(axis=1) * FS, np.abs(dx).sum(axis=1), psi.mean(axis=1),
            kurt, skew, rise, fall, zc, act, mob, comp]
    return np.column_stack(cols + [ab, rel])
