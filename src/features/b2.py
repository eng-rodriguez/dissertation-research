"""B2 = B1 + spatial-extent and topography summaries from per-channel amplitude and power only (06 issue 6).

18 features. Definitions where 06 leaves the detail open are the defaults in
protocol/2026-10-12-pipeline-implementation-spec.md §3 (approved before EXP-003); the salience z threshold
is the one scientific default. No inter-channel timing or coupling information enters B2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1 as fb1  # noqa: E402
from vepiset_channel_qc import NAMES as CHANNELS, POS  # noqa: E402

# z* is set once, label-blind, from the pilot before any model output (spec §3); None until then.
SALIENCE_Z: float | None = None
XY = np.array([POS[c] for c in CHANNELS])
LEFT, RIGHT = XY[:, 0] < 0, XY[:, 0] > 0
ANTERIOR, POSTERIOR = XY[:, 1] > 0, XY[:, 1] < 0
I_PTP, I_TE = fb1.NAMES.index("ptp"), fb1.NAMES.index("teager_mean")
I_ABS = [fb1.NAMES.index(f"abs_{b}") for b in fb1.BANDS]
NAMES = (["n_channels_salience_gt_z", "ptp_max_over_median", "teager_max_over_median", "centroid_x",
          "centroid_y", "centroid_spread"]
         + [f"lr_asym_{b}" for b in fb1.BANDS] + [f"ap_asym_{b}" for b in fb1.BANDS])


def _ratio(v: np.ndarray) -> float:
    med = np.median(v)
    return float(v.max() / med) if med > 0 else 0.0


def _asym(a: np.ndarray, b: np.ndarray) -> float:
    return float((a - b) / (a + b)) if a + b > 0 else 0.0


def salience_count(peak_z: np.ndarray, z_star: float | None = None) -> np.ndarray:
    """Number of channels whose window-peak salience z exceeds z*; peak_z is (..., 19)."""
    z_star = SALIENCE_Z if z_star is None else z_star
    if z_star is None:
        raise ValueError("salience threshold z* not fixed yet (set it from the pilot, label-blind, spec §3)")
    return (np.asarray(peak_z) > z_star).sum(axis=-1).astype(float)


def topography(f1: np.ndarray, xy: np.ndarray = XY) -> np.ndarray:
    """The 17 B2 additions that need no threshold, for one window's B1 matrix (channels x 24). xy: channel
    positions (10-20 electrodes; derivation midpoints for the bipolar sensitivity montage)."""
    left, right, anterior, posterior = xy[:, 0] < 0, xy[:, 0] > 0, xy[:, 1] > 0, xy[:, 1] < 0
    ptp = f1[:, I_PTP]
    w = ptp / ptp.sum() if ptp.sum() > 0 else np.full(len(ptp), 1 / len(ptp))
    c = w @ xy
    spread = float(np.sqrt(w @ ((xy - c) ** 2).sum(axis=1)))
    powers = f1[:, I_ABS]
    lr = [_asym(powers[left, k].sum(), powers[right, k].sum()) for k in range(len(I_ABS))]
    ap = [_asym(powers[anterior, k].sum(), powers[posterior, k].sum()) for k in range(len(I_ABS))]
    return np.array([_ratio(ptp), _ratio(f1[:, I_TE]), c[0], c[1], spread] + lr + ap)


def b2_extra(f1: np.ndarray, peak_z: np.ndarray, z_star: float | None = None) -> np.ndarray:
    """The 18 B2 additions (in NAMES order) for one window."""
    return np.concatenate([salience_count(peak_z, z_star)[None], topography(f1)])
