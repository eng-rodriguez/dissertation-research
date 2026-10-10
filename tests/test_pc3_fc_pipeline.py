"""PC-3 (06 issue 4): the FC pipeline recovers an injected phase lag, IAAFT surrogates destroy it, and the
real and surrogate pipelines agree when there is no coupling. Criteria: experiments/EXP-001-pc3/config.json."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "connectivity"))
import pc3  # noqa: E402
import surrogates as sg  # noqa: E402
import wpli as wp  # noqa: E402

CFG = json.loads((ROOT / "experiments" / "EXP-001-pc3" / "config.json").read_text())
CRIT = CFG["criteria"]


def test_edge_order_is_upper_triangle_row_major():
    i, j = wp.edge_index(19)
    assert len(i) == 171 and (i[0], j[0]) == (0, 1) and (i[17], j[17]) == (0, 18) and (i[18], j[18]) == (1, 2)
    assert (i[-1], j[-1]) == (17, 18)


def test_wpli_formula_known_values():
    rng = np.random.default_rng(0)
    z0 = np.exp(1j * np.cumsum(rng.uniform(0.05, 0.2, 1000))) * rng.uniform(0.5, 2, 1000)
    lagged = np.vstack([z0, z0 * np.exp(-1j * np.pi / 3), z0, np.zeros(1000)])
    w, n_zero = wp.wpli(lagged)
    i, j = wp.edge_index(4)
    got = dict(zip(zip(i, j), w))
    assert got[(0, 1)] == pytest.approx(1.0)
    assert got[(0, 2)] == 0.0 and got[(0, 3)] == 0.0  # zero lag and zero signal: denominator 0 -> 0
    assert n_zero == 4  # (0,2), (0,3), (1,3), (2,3)
    assert wp.signed_imag(lagged)[0] > 0  # channel 1 lags channel 0


def test_surrogate_seed_rule():
    s = sg.surrogate_seed("DA00100A", 2000, "theta", 0, 3)
    assert s == sg.surrogate_seed("DA00100A", 2000, "theta", 0, 3)
    assert s != sg.surrogate_seed("DA00100A", 2000, "theta", 0, 4)
    assert 0 <= s < 2 ** 64


def test_iaaft_preserves_amplitudes_and_is_reproducible():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(3, 1000)), axis=1)
    s1, it1 = sg.iaaft_rows(x, [11, 12, 13])
    s2, _ = sg.iaaft_rows(x, [11, 12, 13])
    assert np.array_equal(s1, s2)
    assert np.allclose(np.sort(s1, axis=1), np.sort(x, axis=1))
    rel = np.abs(np.abs(np.fft.rfft(s1, axis=1)) - np.abs(np.fft.rfft(x, axis=1))).mean() / np.abs(np.fft.rfft(x, axis=1)).mean()
    assert rel < 0.1 and max(it1) <= 200
    # each row depends only on its own seed: row 0 matches a one-row run with the same seed
    s0, _ = sg.iaaft_rows(x[:1], [11])
    assert np.array_equal(s0[0], s1[0])


@pytest.fixture(scope="module")
def report():
    return pc3.run(CFG)


@pytest.mark.parametrize("band", CFG["synthetic_design"]["bands"])
def test_pc3_lag_recovery(report, band):
    r, c = report[band]["lag"], CRIT["lag_recovery"]
    assert min(r["mean_wpli_coupled"]) >= c["min_mean_wpli_coupled_edge"]
    assert r["coupled_are_top_edges"] is c["coupled_edges_are_the_top_edges"]
    assert min(r["share_correct_direction"]) >= c["min_share_windows_correct_lag_direction"]


@pytest.mark.parametrize("band", CFG["synthetic_design"]["bands"])
def test_pc3_surrogates_destroy_coupling(report, band):
    r, c = report[band]["surrogate"], CRIT["surrogate_destruction"]
    lo, hi = r["uncoupled_mean_range"]
    assert all(lo <= m <= hi for m in r["mean_wpli_coupled"])
    assert abs(r["diff_mean_coupled_vs_uncoupled"]) <= c["max_abs_diff_mean_surrogate_coupled_vs_uncoupled"]


@pytest.mark.parametrize("band", CFG["synthetic_design"]["bands"])
def test_pc3_null_coupling_real_matches_surrogate(report, band):
    r, c = report[band]["null"], CRIT["null_coupling_match"]
    assert max(abs(b) for b in r["offset_ci95"]) <= c["max_abs_offset_ci_bound"]
    assert r["ks_distance"] <= c["max_ks_distance_real_vs_single_realisation"]
