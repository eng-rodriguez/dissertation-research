"""FC_surr (06 issue 8, §4.3 step 8): per-channel IAAFT surrogates of each band-passed window.

For each window, band and realisation, each channel of the band-passed real-valued window (cut from the
continuously filtered CAR signal, before the Hilbert transform) is replaced by an independent IAAFT
surrogate (at most 200 iterations; stop when the change in relative spectral error is < 1e-8). The
surrogate's analytic signal is computed on the window itself, wPLI is applied as for the real signal,
and wPLI is averaged per edge across the k realisations.

Seed per channel surrogate: the integer from the first 8 bytes of
SHA-256("fcsurr:20261010:<eeg_id>:<epoch start sample>:<band>:<realisation>:<channel>"), read big-endian
(the same integer as int(hexdigest[:16], 16)). Implementation choice: the protocol does not name the byte order.
"""

from __future__ import annotations

import hashlib

import numpy as np

import wpli as wp

GLOBAL_SEED = 20261010
MAX_ITER = 200
TOL = 1e-8


def surrogate_seed(eeg_id: str, start: int, band: str, realisation: int, channel: int) -> int:
    key = f"fcsurr:{GLOBAL_SEED}:{eeg_id}:{start}:{band}:{realisation}:{channel}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def iaaft_rows(x: np.ndarray, seeds, max_iter: int = MAX_ITER, tol: float = TOL) -> tuple[np.ndarray, list[int]]:
    """Independent IAAFT surrogate of each row of x, row r seeded by seeds[r].

    Rows are processed together but each keeps its own stopping point, so every row equals what a
    one-row run with the same seed gives. Returns (surrogates, iterations used per row).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    amp = np.abs(np.fft.rfft(x, axis=-1))
    sorted_x = np.sort(x, axis=-1)
    s = np.vstack([np.random.default_rng(sd).permutation(row) for sd, row in zip(seeds, x)])
    prev = np.full(x.shape[0], np.inf)
    iters = np.zeros(x.shape[0], dtype=int)
    active = np.ones(x.shape[0], dtype=bool)
    amp2 = np.mean(amp ** 2, axis=-1)
    for it in range(1, max_iter + 1):
        a = np.flatnonzero(active)
        phase = np.angle(np.fft.rfft(s[a], axis=-1))
        y = np.fft.irfft(amp[a] * np.exp(1j * phase), n=n, axis=-1)
        ranks = np.argsort(np.argsort(y, axis=-1), axis=-1)
        s[a] = np.take_along_axis(sorted_x[a], ranks, axis=-1)
        err = np.mean((np.abs(np.fft.rfft(s[a], axis=-1)) - amp[a]) ** 2, axis=-1) / amp2[a]
        iters[a] = it
        done = np.abs(prev[a] - err) < tol
        prev[a] = err
        active[a[done]] = False
        if not active.any():
            break
    return s, iters.tolist()


def surrogate_wpli(band_window: np.ndarray, eeg_id: str, start: int, band: str, k: int) -> np.ndarray:
    """wPLI of k surrogate realisations of one band-passed window (channels x samples): k x n_edges."""
    out = []
    for r in range(k):
        seeds = [surrogate_seed(eeg_id, start, band, r, c) for c in range(band_window.shape[0])]
        s, _ = iaaft_rows(band_window, seeds)
        out.append(wp.wpli(wp.analytic(s))[0])
    return np.vstack(out)


def surrogate_fc(x_cont: np.ndarray, eeg_id: str, window_starts, n_samples: int, k: int,
                 bands=wp.PRIMARY_BANDS, fs: int = wp.FS, epoch_starts=None, keep_realisations: bool = False) -> dict:
    """FC_surr for one recording: dict band -> (n_windows x n_edges) mean wPLI over k realisations.

    epoch_starts: the 4-s epoch start sample of each window, used in the seed (defaults to window_starts).
    With keep_realisations, also returns the per-realisation array (n_windows x k x n_edges) under "realisations".
    """
    xc = wp.car(x_cont)
    seed_starts = list(window_starts if epoch_starts is None else epoch_starts)
    out = {}
    for b in bands:
        xb = wp.bandpass(xc, b, fs)
        reals = np.stack([surrogate_wpli(xb[:, s:s + n_samples], eeg_id, e, b, k)
                          for s, e in zip(window_starts, seed_starts)])
        out[b] = {"wpli": reals.mean(axis=1)}
        if keep_realisations:
            out[b]["realisations"] = reals
    return out
