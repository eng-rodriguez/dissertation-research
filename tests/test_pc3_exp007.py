"""PC-3 gate, EXP-007 (06 issue 4): EXP-006 criteria with a Monte Carlo precision rule, run over many synthetic
blocks (about an hour), so this test checks the committed report rather than re-running it:
    python src/connectivity/pc3.py --config experiments/EXP-007-pc3/config.json
Criteria and result rule: experiments/EXP-007-pc3/config.json."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "experiments" / "EXP-007-pc3" / "config.json"
CFG = json.loads(CFG_PATH.read_text())
CRIT = CFG["criteria"]
BANDS = CFG["synthetic_design"]["bands"]
REPORT = ROOT / "artifacts" / "protocol" / "pc3" / "exp007_report.json"


@pytest.fixture(scope="module")
def out():
    assert REPORT.exists(), "EXP-007 report missing: run pc3.py with the EXP-007 config"
    o = json.loads(REPORT.read_text())
    assert o["config_sha256"] == hashlib.sha256(CFG_PATH.read_bytes()).hexdigest(), "report made from another config"
    return o


@pytest.mark.parametrize("band", BANDS)
def test_exp007_lag_recovery(out, band):
    r, c = out["report"][band]["lag"], CRIT["lag_recovery"]
    assert min(r["mean_wpli_coupled"]) >= c["min_mean_wpli_coupled_edge"]
    assert r["coupled_are_top_edges"] is c["coupled_edges_are_the_top_edges"]
    assert min(r["share_correct_direction"]) >= c["min_share_windows_correct_lag_direction"]


@pytest.mark.parametrize("band", BANDS)
def test_exp007_surrogates_destroy_coupling(out, band):
    r, c = out["report"][band]["surrogate"], CRIT["surrogate_destruction"]
    lo, hi = c["direction_share_range"]
    assert all(lo <= s <= hi for s in r["share_injected_direction"])
    assert max(v for pair in r["excess_over_same_channel_reference"] for v in pair) \
        <= c["max_excess_over_same_channel_reference"]
    assert max(v for pair in r["excess_se"] for v in pair) <= c["precision"]["max_se_per_excess"]


@pytest.mark.parametrize("band", BANDS)
def test_exp007_null_coupling_real_matches_surrogate(out, band):
    r, c = out["report"][band]["null"], CRIT["null_coupling_match"]
    assert max(abs(b) for b in r["offset_ci95"]) <= c["max_abs_offset_ci_bound"]
    assert r["ks_distance"] <= c["max_ks_distance_real_vs_single_realisation"]


def test_exp007_result_is_pass(out):
    assert out["result"] == "PASS"
