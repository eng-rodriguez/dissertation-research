"""Label-blind salience-selected 2-s window (06 issue 1, v1.0) and the kept-epoch list for one recording.

Rule (issue 1): on the CAR signal, per channel, the Teager-Kaiser energy psi[n] = x[n]^2 - x[n-1] x[n+1],
smoothed with a 20-ms centred moving average, is z-scored with that channel's recording-wide median and
MAD (scaled by 1.4826, implementation detail in protocol/2026-10-12-pipeline-implementation-spec.md).
Salience s(t) = max over the 19 channels. Within each 4-s epoch: t_raw = argmax s over [0, 4) s,
t_center = clip(t_raw, 1.0 s, 3.0 s), window = [t_center - 1.0 s, t_center + 1.0 s): 1,000 samples, always
inside the epoch and always containing t_raw. Identical for IED and non-IED epochs and every representation.
The 4-s sensitivity analysis uses the whole epoch instead.

Epoch exclusions are the locked ones in src/data/vepiset_evaluable_recount.classify (06 §4.2, R3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "connectivity"))
import vepiset_audit as va  # noqa: E402
import vepiset_evaluable_recount as rc  # noqa: E402
import wpli as wp  # noqa: E402

FS = 500
EPOCH = va.EPOCH_SAMPLES          # 2,000 samples = 4 s
HALF = FS                         # 1.0 s either side of the centre
WIN = 2 * HALF                    # 1,000 samples
SMOOTH = int(0.020 * FS)          # 20 ms = 10 samples
MAD_SCALE = 1.4826
N_SCALP = 19


def load_scalp(mat_path: Path) -> np.ndarray:
    """MAT eeg_data rows 1-19 (stored order Fp1 ... Pz) as float64, channels x samples. Never other rows."""
    import scipy.io

    x = scipy.io.loadmat(mat_path, variable_names=["eeg_data"])["eeg_data"]
    return np.asarray(x[:N_SCALP], dtype=np.float64)


def tkeo(x: np.ndarray) -> np.ndarray:
    """Teager-Kaiser energy per row; the two end samples copy their neighbours."""
    psi = np.empty_like(x)
    psi[:, 1:-1] = x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:]
    psi[:, 0], psi[:, -1] = psi[:, 1], psi[:, -2]
    return psi


def salience_z(x_car: np.ndarray) -> np.ndarray:
    """Per-channel robust z of the smoothed TKEO (channels x samples), z-scored over the whole recording."""
    e = uniform_filter1d(tkeo(x_car), size=SMOOTH, axis=-1, mode="nearest")
    med = np.median(e, axis=-1, keepdims=True)
    mad = MAD_SCALE * np.median(np.abs(e - med), axis=-1, keepdims=True)
    return (e - med) / np.where(mad > 0, mad, 1.0)


def window_start(s_max: np.ndarray, epoch_start: int) -> tuple[int, int]:
    """(window start sample, t_raw offset in samples) for the epoch starting at epoch_start."""
    t_raw = int(np.argmax(s_max[epoch_start:epoch_start + EPOCH]))
    t_center = min(max(t_raw, HALF), EPOCH - HALF)
    return epoch_start + t_center - HALF, t_raw


def kept_epochs(epochs: list[dict], qc_flags: list[bool], lift: frozenset = frozenset()) -> list[dict]:
    """Epochs that pass every locked exclusion, in start order (classify expects start order). With lift
    (excluded-epochs sensitivity only), epochs excluded only by a lifted rule are kept too, each carrying its
    locked exclusion reason under "locked_reason" ("" for epochs the primary analysis keeps)."""
    eps = sorted(epochs, key=lambda e: e["start"])
    if not lift:
        return [e for e, r in zip(eps, rc.classify(eps, qc_flags)) if r is None]
    locked = rc.classify(eps, qc_flags)
    return [{**e, "locked_reason": r0 or ""} for e, r0, r in zip(eps, locked, rc.classify(eps, qc_flags, lift))
            if r is None]


def windows(x_scalp: np.ndarray, epochs: list[dict], mode: str = "salience2s") -> dict:
    """Window starts and per-channel peak salience z inside each window for the kept epochs.

    mode "salience2s" (primary) or "epoch4s" (sensitivity: whole epoch, no selection).
    Returns {"starts", "n_samples", "t_raw", "peak_z" (n x 19), "x_car"}.
    """
    xc = wp.car(x_scalp)
    z = salience_z(xc)
    s_max = z.max(axis=0)
    starts, t_raw, peak = [], [], []
    for e in epochs:
        if mode == "salience2s":
            w0, tr = window_start(s_max, e["start"])
            n = WIN
        elif mode == "epoch4s":
            w0, tr, n = e["start"], int(np.argmax(s_max[e["start"]:e["start"] + EPOCH])), EPOCH
        else:
            raise ValueError(mode)
        starts.append(w0)
        t_raw.append(tr)
        peak.append(z[:, w0:w0 + n].max(axis=1))
    return {"starts": np.array(starts, dtype=int), "n_samples": WIN if mode == "salience2s" else EPOCH,
            "t_raw": np.array(t_raw, dtype=int), "peak_z": np.array(peak).reshape(len(epochs), N_SCALP),
            "x_car": xc}
