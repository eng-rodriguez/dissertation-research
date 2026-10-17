"""Longitudinal bipolar montage (double banana, 18 derivations): the issue-11 sensitivity analysis (06 v1.1).

Derived from the 19 stored scalp rows (stored order Fp1 ... Pz). Every row carries the same reference, so each
difference removes it exactly; CAR is not applied on top. Derivation positions are the midpoints of the two
electrodes' 2-D 10-20 positions (06 issue 11: adjacency graph on derivation midpoints).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from vepiset_channel_qc import NAMES, POS  # noqa: E402

# Left temporal, right temporal, left parasagittal, right parasagittal, midline.
PAIRS = [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
         ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
         ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
         ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
         ("Fz", "Cz"), ("Cz", "Pz")]
DERIVATIONS = [f"{a}-{b}" for a, b in PAIRS]
_A = np.array([NAMES.index(a) for a, _ in PAIRS])
_B = np.array([NAMES.index(b) for _, b in PAIRS])
XY = np.array([(np.array(POS[a]) + np.array(POS[b])) / 2 for a, b in PAIRS])


def bipolar(x_scalp: np.ndarray) -> np.ndarray:
    """18 x samples double-banana derivations from the 19 stored scalp rows (19 x samples)."""
    x = np.asarray(x_scalp, dtype=np.float64)
    return x[_A] - x[_B]
