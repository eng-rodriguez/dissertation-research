"""EXP-017 node-strength rung: the feature, its loading, and the run + summary on a synthetic cache."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("pipeline", "features"):
    sys.path.insert(0, str(ROOT / "src" / p))
sys.path.insert(0, str(ROOT / "tests"))
import confirmatory as cf  # noqa: E402
import exp017_node_strength as e17  # noqa: E402
import node_strength as fns  # noqa: E402
from test_confirmatory import FAST, fake_cache  # noqa: E402


def test_node_strength_matches_a_direct_computation():
    rng = np.random.default_rng(0)
    fc = rng.random((4, 513))
    ns = fns.node_strength(fc)
    assert ns.shape == (4, 60)
    iu = np.triu_indices(19, k=1)
    for b in range(3):
        e = fc[1, b * 171:(b + 1) * 171]
        m = np.zeros((19, 19))
        m[iu] = e
        m = m + m.T
        assert np.allclose(ns[1, b * 20:b * 20 + 19], [m[i].sum() / 18 for i in range(19)])
        assert np.isclose(ns[1, b * 20 + 19], e.mean())


def test_frozen_config_names_the_locked_protocol():
    c, sha = e17.load_config()
    assert sha == e17.FROZEN_SHA256
    e17.check_frozen(sha)
    assert c["protocol"]["sha256"] == hashlib.sha256((ROOT / c["protocol"]["document"]).read_bytes()).hexdigest()
    assert set(c["models_run"]) | set(c["reference_models"]) <= set(cf.L.MODELS)
    with pytest.raises(SystemExit):
        e17.check_frozen("0" * 64)


@pytest.fixture
def small_design(monkeypatch):
    d = cf.design()
    two = dict(list(d["splits"].items())[:2])
    monkeypatch.setattr(cf, "design", lambda manifest=None, sets=("primary",): {"recs": d["recs"], "splits": two})
    monkeypatch.setattr(cf.L, "HGB_GRID", FAST)


def test_run_and_summary_pair_with_the_reference_units(tmp_path, small_design):
    cache, ref, out = tmp_path / "cache", tmp_path / "ref", tmp_path / "out"
    cache.mkdir()
    fake_cache(cache, gain=0.3)
    cf._CACHE.clear()
    cf.run(cache, ref, ref / "scores", ["b2", "b2_fc"], "hgb", 5.0, 1)
    cf.run(cache, out, out / "scores", ["b2_ns"], "hgb", 5.0, 1)
    p = cf.patient(cache, sorted(cf.design()["recs"])[0], cf.blocks_for("b2_ns"), [], 5.0)
    assert p["F"]["fc_ns"].shape[1] == 60
    c, _ = e17.load_config()
    c.update(out=str(out), reference_units=str(ref / "units" / "hgb.jsonl"))
    res = e17.summary(c, "test", with_states=False)
    assert set(res["contrasts"]) == {"b2_ns_vs_b2", "b2_fc_vs_b2_ns"}
    assert res["contrasts"]["b2_ns_vs_b2"]["n_e"] == 35
    assert res["mean_auroc"]["b2_ns"] > res["mean_auroc"]["b2"]  # planted FC signal survives the summary
    json.dumps(res)
