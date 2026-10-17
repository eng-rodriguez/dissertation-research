"""Spherical-spline interpolation of bad scalp channels: the issue-12 sensitivity analysis only (06 v1.1).

Never used in the primary analysis (06 issue 12: interpolation manufactures spatial dependence). Perrin et al.
(1989) [VERIFY] surface spline on the unit sphere, implementation defaults fixed with EXP-012 before its run:
  positions   the 2-D 10-20 positions of vepiset_channel_qc.POS read as an azimuthal equidistant projection
              from the vertex (radius 1 = 90 degrees, so T3/T4 lie on the equator), mapped to the unit sphere
  kernel      g(x) = 1/(4 pi) sum_{n=1}^{50} (2n + 1) / (n (n + 1))^4 P_n(x)   (stiffness m = 4, 50 terms)
  fit         [G + 1e-5 I, 1; 1', 0] [c; c0] = [v; 0] over the good channels, per sample
The interpolation is a fixed linear map of the good rows, so a reference shared by all stored rows passes
through unchanged (the constant term absorbs it) and the result is the same before or after re-referencing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from vepiset_channel_qc import NAMES, POS  # noqa: E402

STIFFNESS = 4
N_TERMS = 50
LAMBDA = 1e-5


def positions3d(names=NAMES) -> np.ndarray:
    xy = np.array([POS[n] for n in names], dtype=float)
    r = np.hypot(xy[:, 0], xy[:, 1])
    theta, phi = r * np.pi / 2, np.arctan2(xy[:, 1], xy[:, 0])
    return np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def _g(cosang: np.ndarray) -> np.ndarray:
    n = np.arange(1, N_TERMS + 1)
    coef = (2 * n + 1) / (n * (n + 1)) ** STIFFNESS / (4 * np.pi)
    x = np.clip(cosang, -1, 1)
    return sum(c * eval_legendre(k, x) for k, c in zip(n, coef))


def matrix(bad: list[str], names=NAMES) -> np.ndarray:
    """(n_bad x n_good) map from the good rows (in stored order) to the bad rows."""
    pos = positions3d(names)
    bi = [names.index(b) for b in bad]
    gi = [i for i in range(len(names)) if i not in bi]
    pg, pb = pos[gi], pos[bi]
    G = _g(pg @ pg.T) + LAMBDA * np.eye(len(gi))
    A = np.block([[G, np.ones((len(gi), 1))], [np.ones((1, len(gi))), np.zeros((1, 1))]])
    Gb = np.hstack([_g(pb @ pg.T), np.ones((len(bi), 1))])
    return Gb @ np.linalg.inv(A)[:, :len(gi)]


def interpolate(x_scalp: np.ndarray, bad: list[str], names=NAMES) -> np.ndarray:
    """Copy of the 19 stored rows with the bad rows replaced by their spline interpolation."""
    x = np.array(x_scalp, dtype=np.float64, copy=True)
    if not bad:
        return x
    bi = [names.index(b) for b in bad]
    gi = [i for i in range(len(names)) if i not in bi]
    x[bi] = matrix(bad, names) @ x[gi]
    return x
