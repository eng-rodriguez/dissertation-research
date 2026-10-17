"""B2 + ADJ: fixed anatomical-adjacency comparator (06 issue 7).

Fixed graph over the 2-D 10-20 positions, Gaussian weights w_ij = exp(-d_ij^2 / sigma^2) with sigma = the
median nearest-neighbour distance; no data-driven edges. For every B1 feature, per channel: the
neighbour-weighted mean (self excluded, weights normalised to 1) and the channel-minus-neighbour contrast.
No phase, correlation or coupling information.
"""

from __future__ import annotations

import numpy as np

from b2 import XY

def weights(xy: np.ndarray) -> np.ndarray:
    """Row-normalised Gaussian adjacency over the given 2-D positions (self excluded)."""
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    sigma = float(np.median(np.where(np.eye(len(xy), dtype=bool), np.inf, d).min(axis=1)))
    w = np.exp(-(d ** 2) / sigma ** 2)
    np.fill_diagonal(w, 0.0)
    return w / w.sum(axis=1, keepdims=True)


W = weights(XY)


def adj(f1: np.ndarray, w: np.ndarray = W) -> np.ndarray:
    """ADJ features for one window from its B1 matrix (channels x 24): means then contrasts, flattened."""
    nb = w @ f1
    return np.concatenate([nb.ravel(), (f1 - nb).ravel()])
