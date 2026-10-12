"""Salience window, B1/B2/ADJ and per-recording extraction (06 issues 1, 5-8)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("features", "preprocessing"):
    sys.path.insert(0, str(ROOT / "src" / p))
import adj as fadj  # noqa: E402
import b1 as fb1  # noqa: E402
import b2 as fb2  # noqa: E402
import extract as ex  # noqa: E402
import windows as wn  # noqa: E402


def test_window_is_clamped_inside_epoch_and_contains_peak():
    s = np.zeros(10 * wn.EPOCH)
    for t_raw, expect in ((100, 0), (1000, 500), (1999, 1000)):
        s[:] = 0
        s[4 * wn.EPOCH + t_raw] = 1
        w0, tr = wn.window_start(s, 4 * wn.EPOCH)
        assert tr == t_raw and w0 - 4 * wn.EPOCH == expect
        assert w0 <= 4 * wn.EPOCH + t_raw < w0 + wn.WIN and w0 + wn.WIN <= 5 * wn.EPOCH


def test_salience_is_label_blind_and_finds_injected_transient():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((19, 12 * wn.EPOCH)) * 10
    x[:4, 5 * wn.EPOCH + 1700:5 * wn.EPOCH + 1710] += 300
    eps = [{"start": i * wn.EPOCH, "end": (i + 1) * wn.EPOCH, "label": 0, "state": "sleep", "has_ied_event": 0}
           for i in range(12)]
    w = wn.windows(x, eps)
    assert abs(w["t_raw"][5] - 1700) < 15 and w["starts"][5] == 5 * wn.EPOCH + 1000


def test_kept_epochs_apply_locked_exclusions():
    eps = [{"start": i * 2000, "end": (i + 1) * 2000, "label": int(i == 5), "state": "sleep",
            "has_ied_event": int(i == 5)} for i in range(12)]
    assert [e["start"] // 2000 for e in wn.kept_epochs(eps, [False] * 12)] == [1, 2, 3, 5, 7, 8, 9, 10]


def test_b1_known_values():
    t = np.arange(1000) / 500
    x = np.vstack([np.sin(2 * np.pi * 10 * t)] * 19)
    f = fb1.b1(x)
    n = fb1.NAMES
    assert f.shape == (19, 24) and np.isfinite(f).all()
    assert f[0, n.index("ptp")] == pytest.approx(2, abs=0.01)
    assert f[0, n.index("zero_cross_rate")] == pytest.approx(20, abs=1)
    assert f[0, n.index("rel_alpha")] > 0.9
    assert f[0, n.index("rel_beta")] < 0.05


def test_b2_and_adj_shapes_and_threshold_gate():
    f1 = np.abs(np.random.default_rng(1).standard_normal((19, 24)))
    assert fb2.topography(f1).shape == (17,) and fadj.adj(f1).shape == (912,)
    with pytest.raises(ValueError):
        fb2.salience_count(np.zeros(19))
    assert fb2.salience_count(np.arange(19.0), 10.0) == 8
    assert np.allclose(fadj.W.sum(axis=1), 1) and np.all(np.diag(fadj.W) == 0)


def test_lateralised_power_gives_left_asymmetry():
    f1 = np.ones((19, 24))
    f1[fb2.LEFT, :] *= 3
    lr = fb2.topography(f1)[5:11]
    assert np.allclose(lr, 0.5)


def test_extract_shapes_no_surrogates():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((19, 8 * 2000))
    eps = [{"start": i * 2000, "end": (i + 1) * 2000, "label": 0, "state": "wake", "has_ied_event": 0}
           for i in range(1, 7)]
    f = ex.features(x, "SYN", eps, with_surrogates=False)
    assert f["b1"].shape == (6, 19, 24) and f["b2_topo"].shape == (6, 17)
    assert f["adj"].shape == (6, 912) and f["fc"].shape == (6, 513) and f["peak_z"].shape == (6, 19)
