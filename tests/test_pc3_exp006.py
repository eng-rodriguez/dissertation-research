"""PC-3 gate, EXP-006 (06 issue 4): re-test on fresh seeds; surrogate destruction judged against the
same-channel surrogate reference (two independent surrogates of one channel: no coupling by construction).
Lag recovery, null match and lag direction at chance as in EXP-005. Criteria: experiments/EXP-006-pc3/config.json."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "connectivity"))
import pc3  # noqa: E402

CFG = json.loads((ROOT / "experiments" / "EXP-006-pc3" / "config.json").read_text())
CRIT = CFG["criteria"]
BANDS = CFG["synthetic_design"]["bands"]


@pytest.fixture(scope="module")
def report():
    return pc3.run(CFG)


@pytest.mark.parametrize("band", BANDS)
def test_exp006_lag_recovery(report, band):
    r, c = report[band]["lag"], CRIT["lag_recovery"]
    assert min(r["mean_wpli_coupled"]) >= c["min_mean_wpli_coupled_edge"]
    assert r["coupled_are_top_edges"] is c["coupled_edges_are_the_top_edges"]
    assert min(r["share_correct_direction"]) >= c["min_share_windows_correct_lag_direction"]


@pytest.mark.parametrize("band", BANDS)
def test_exp006_surrogates_destroy_coupling(report, band):
    r, c = report[band]["surrogate"], CRIT["surrogate_destruction"]
    lo, hi = c["direction_share_range"]
    assert all(lo <= s <= hi for s in r["share_injected_direction"])
    assert max(v for pair in r["excess_over_same_channel_reference"] for v in pair) \
        <= c["max_excess_over_same_channel_reference"]


@pytest.mark.parametrize("band", BANDS)
def test_exp006_null_coupling_real_matches_surrogate(report, band):
    r, c = report[band]["null"], CRIT["null_coupling_match"]
    assert max(abs(b) for b in r["offset_ci95"]) <= c["max_abs_offset_ci_bound"]
    assert r["ks_distance"] <= c["max_ks_distance_real_vs_single_realisation"]
