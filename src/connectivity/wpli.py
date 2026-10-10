"""wPLI functional-connectivity features, exactly as 06 §4.3 (v1.0, RDR-013) specifies.

Steps (06 §4.3): CAR over the 19 scalp rows; per band, Butterworth order-4 band-pass applied zero-phase
with sosfiltfilt to the whole continuous recording; Hilbert transform of the whole band-passed recording;
then per window and channel pair i < j, X_ij(t) = Im(z_i(t) conj z_j(t)) and
wPLI_ij = |sum_t X_ij| / sum_t |X_ij|, set to 0 when the denominator is 0 (and counted).
Edges are in stored channel order, upper triangle, row-major (171 for 19 channels).
Recording-edge epochs (first and last full 4-s epoch) are excluded by the caller, not here.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

FS = 500
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "low_gamma": (30, 45)}
PRIMARY_BANDS = ("theta", "alpha", "beta")


def car(x: np.ndarray) -> np.ndarray:
    """Common average reference over channels (rows)."""
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True)


def band_sos(band: str, fs: int = FS) -> np.ndarray:
    return butter(4, BANDS[band], btype="bandpass", fs=fs, output="sos")


def bandpass(x: np.ndarray, band: str, fs: int = FS) -> np.ndarray:
    """Zero-phase band-pass of each row over its whole length (sosfiltfilt, default odd padding)."""
    return sosfiltfilt(band_sos(band, fs), x, axis=-1)


def analytic(x_band: np.ndarray) -> np.ndarray:
    return hilbert(x_band, axis=-1)


def edge_index(n_ch: int = 19) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_ch, k=1)


def _cross_imag(z: np.ndarray) -> np.ndarray:
    i, j = edge_index(z.shape[0])
    return (z[i] * np.conj(z[j])).imag  # edges x samples


def wpli(z: np.ndarray) -> tuple[np.ndarray, int]:
    """wPLI per edge for one window of analytic signals (channels x samples). Returns (values, n zero-denominator edges)."""
    x = _cross_imag(z)
    num = np.abs(x.sum(axis=1))
    den = np.abs(x).sum(axis=1)
    zero = den == 0
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=~zero)
    return out, int(zero.sum())


def signed_imag(z: np.ndarray) -> np.ndarray:
    """sum_t Im(z_i conj z_j) per edge: > 0 when channel j lags channel i. Diagnostic only (PC-3 lag direction)."""
    return _cross_imag(z).sum(axis=1)


def real_fc(x_cont: np.ndarray, window_starts, n_samples: int, bands=PRIMARY_BANDS, fs: int = FS) -> dict:
    """Real-signal FC for one recording: dict band -> (n_windows x n_edges) wPLI and zero-denominator count.

    x_cont: continuous recording, channels x samples (the 19 scalp rows). window_starts: sample indices.
    """
    xc = car(x_cont)
    out = {}
    for b in bands:
        z = analytic(bandpass(xc, b, fs))
        vals, zeros = zip(*(wpli(z[:, s:s + n_samples]) for s in window_starts))
        out[b] = {"wpli": np.vstack(vals), "n_zero_denominator": int(sum(zeros))}
    return out
