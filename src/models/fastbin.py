"""Speed-only patch for scikit-learn 1.9.1's weighted HGB binning (no change in results).

With sample weights, sklearn 1.9.1's `_find_binning_thresholds` calls `_weighted_percentile` once per
percentile (254 calls per feature), and each call re-sorts the column. On ladder-sized inputs (about 500-1,400
features) that dominated every fit (about 0.3 s per feature in the cloud container). The patched function
below is sklearn's own code with that list comprehension replaced by a vectorised re-implementation of
`_weighted_percentile(..., average=True)` that performs the identical operations on the same sorted order.
`tests/test_ladder.py` checks the thresholds are bit-identical to the original on tied, weighted data.
The patch refuses to apply to any scikit-learn version other than the pinned one.
"""

from __future__ import annotations

import numpy as np
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble._hist_gradient_boosting import binning
from sklearn.ensemble._hist_gradient_boosting.common import ALMOST_INF, X_DTYPE

PINNED = "1.9.1"
ORIGINAL = binning._find_binning_thresholds


def weighted_percentiles_averaged(array: np.ndarray, sample_weight: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """`_weighted_percentile(array, sample_weight, ranks, average=True)` for a 1-D float64 array with strictly
    positive weights and 0 < ranks < 100, vectorised over ranks with the same operations in the same order:
    the same unstable argsort, sequential cumulative sum, left searchsorted, clip and eps rule."""
    sorted_idx = np.argsort(array, axis=0)
    cdf = np.cumsum(sample_weight[sorted_idx])
    adjusted = ranks / 100 * cdf[-1]
    max_idx = array.shape[0] - 1
    idx = np.clip(np.searchsorted(cdf, adjusted), 0, max_idx)
    below = array[sorted_idx[idx]]
    above = array[sorted_idx[np.clip(idx + 1, 0, max_idx)]]
    is_fraction_above = (cdf[idx] - adjusted) > np.finfo(np.float64).eps
    return np.where(is_fraction_above, below, (below + above) / 2)


def find_binning_thresholds(col_data, max_bins, sample_weight=None):
    missing_mask = np.isnan(col_data)
    any_missing = missing_mask.any()
    if any_missing:
        col_data = col_data[~missing_mask]
    if sample_weight is not None:
        if any_missing:
            sample_weight = sample_weight[~missing_mask]
        nnz_sw = sample_weight != 0
        col_data = col_data[nnz_sw]
        sample_weight = sample_weight[nnz_sw]
    sort_idx = np.argsort(col_data)
    col_data = col_data[sort_idx]
    if sample_weight is not None:
        sample_weight = sample_weight[sort_idx]
    distinct_values = np.unique(col_data).astype(X_DTYPE)
    if len(distinct_values) == 1:
        return np.asarray([])
    if len(distinct_values) <= max_bins:
        bin_thresholds = sliding_window_view(distinct_values, 2).mean(axis=1)
    elif sample_weight is None:
        percentiles = np.linspace(0, 100, num=max_bins + 1)[1:-1]
        bin_thresholds = np.percentile(col_data, percentiles, method="averaged_inverted_cdf")
    else:
        percentiles = np.linspace(0, 100, num=max_bins + 1)[1:-1]
        bin_thresholds = weighted_percentiles_averaged(col_data, sample_weight, percentiles)
    unique_bin_values = np.unique(bin_thresholds)
    if unique_bin_values.shape[0] != bin_thresholds.shape[0]:
        bin_thresholds = unique_bin_values
    np.clip(bin_thresholds, a_min=None, a_max=ALMOST_INF, out=bin_thresholds)
    return bin_thresholds


def apply() -> None:
    if sklearn.__version__ != PINNED:
        raise RuntimeError(f"fastbin is written for scikit-learn {PINNED}, found {sklearn.__version__}")
    binning._find_binning_thresholds = find_binning_thresholds
