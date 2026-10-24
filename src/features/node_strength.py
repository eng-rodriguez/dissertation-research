"""Secondary FC features (06 v1.1 §4.3 item 7): node strength s_i = mean over j != i of wPLI_ij per primary band
(19 per band) and the global mean wPLI per band, derived from the cached 513 edges (171 per band, theta/alpha/beta,
upper triangle row-major in the stored channel order)."""

from __future__ import annotations

import numpy as np

N_CH, N_BANDS = 19, 3
N_EDGES = N_CH * (N_CH - 1) // 2
IU = np.triu_indices(N_CH, k=1)


def node_strength(fc: np.ndarray) -> np.ndarray:
    """(n, 513) edges -> (n, 60): per band 19 node strengths then the global mean; bands in cache order."""
    n = fc.shape[0]
    out = []
    for b in range(N_BANDS):
        e = fc[:, b * N_EDGES:(b + 1) * N_EDGES]
        m = np.zeros((n, N_CH, N_CH))
        m[:, IU[0], IU[1]] = e
        m = m + m.transpose(0, 2, 1)
        out += [m.sum(axis=2) / (N_CH - 1), e.mean(axis=1, keepdims=True)]
    return np.hstack(out)
