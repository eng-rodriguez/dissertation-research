"""B2 + ADJ: fixed anatomical-adjacency comparator (06 issue 7).

Fixed graph over the 2-D 10-20 positions, Gaussian weights w_ij = exp(-d_ij^2 / sigma^2) with sigma = the
median nearest-neighbour distance; no data-driven edges. For every B1 feature, per channel: the
neighbour-weighted mean (self excluded, weights normalised to 1) and the channel-minus-neighbour contrast.
No phase, correlation or coupling information.
"""

from __future__ import annotations

import numpy as np

from b2 import XY

_D = np.linalg.norm(XY[:, None] - XY[None], axis=-1)
SIGMA = float(np.median(np.where(np.eye(len(XY), dtype=bool), np.inf, _D).min(axis=1)))
W = np.exp(-(_D ** 2) / SIGMA ** 2)
np.fill_diagonal(W, 0.0)
W /= W.sum(axis=1, keepdims=True)


def adj(f1: np.ndarray) -> np.ndarray:
    """ADJ features for one window from its B1 matrix (19 x 24): 19 x 24 means then 19 x 24 contrasts, flattened."""
    nb = W @ f1
    return np.concatenate([nb.ravel(), (f1 - nb).ravel()])
