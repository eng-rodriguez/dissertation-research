"""EXP-015 epoch-level-split inflation on a synthetic cache with the real frozen design (reduced grid and repeats)."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tests"))
import confirmatory as cf  # noqa: E402
import rq2_epoch_split as es  # noqa: E402
from test_confirmatory import FAST, fake_cache  # noqa: E402


@pytest.fixture
def small_design(monkeypatch):
    d = cf.design()
    two = dict(list(d["splits"].items())[:2])
    monkeypatch.setattr(cf, "design", lambda manifest=None, sets=("primary",): {"recs": d["recs"], "splits": two})
    monkeypatch.setattr(cf.L, "HGB_GRID", FAST)
    return d


def signature_cache(path: Path) -> None:
    """The EXP-004 test cache with B1 zeroed and B2 topography replaced by a fixed per-recording code: it identifies the
    recording and carries no label information across recordings, so only a split that shares recordings can
    exploit it (through each recording's IED rate)."""
    fake_cache(path, gain=0.0)
    rng = np.random.default_rng(1)
    for f in sorted(path.glob("*.npz")):
        with np.load(f) as z:
            arrs = dict(z)
        arrs["b1"] = np.zeros_like(arrs["b1"])  # no epoch-level signal: the code is the only usable information
        arrs["b2_topo"] = np.repeat(rng.uniform(-5, 5, (1, arrs["b2_topo"].shape[1])), len(arrs["label"]), axis=0)
        np.savez_compressed(f, **arrs)


def test_config_names_the_locked_protocol_and_is_not_frozen_yet():
    c, _ = es.load_config()
    assert c["protocol"]["sha256"] == hashlib.sha256((ROOT / c["protocol"]["document"]).read_bytes()).hexdigest()
    assert es.FROZEN_SHA256 is None
    with pytest.raises(SystemExit):
        es.check_frozen(es.load_config()[1])


def test_epoch_folds_partition_and_stratify():
    y = np.array([1] * 50 + [0] * 450)
    f = es.epoch_folds(y, 7)
    assert sorted(set(f)) == list(range(5))
    assert all((y[f == k] == 1).sum() == 10 for k in range(5))
    assert (es.epoch_folds(y, 7) == f).all()


def test_subset_recounts_evaluability():
    p = {"id": "x", "y": np.array([1] * 6 + [0] * 6), "s": np.array(["sleep"] * 12), "F": {"b1": np.zeros((12, 2))}}
    assert es.subset(p, np.ones(12, bool))["evaluable"] == ["sleep"]
    assert es.subset(p, np.arange(12) % 2 == 0)["evaluable"] == []
    assert es.subset(p, np.zeros(12, bool)) is None


def test_epoch_split_inflates_pooled_auroc_on_a_recording_signature(tmp_path, small_design):
    cache, ref, out = tmp_path / "cache", tmp_path / "ref", tmp_path / "out"
    cache.mkdir()
    signature_cache(cache)
    cf.run(cache, ref, ref / "scores", ["b2"], "hgb", 5.0, 1)
    es.run(cache, out, out / "scores", 5.0, 1)
    es.run(cache, out, out / "scores", 5.0, 1)  # resume: nothing to do
    assert len((out / "units" / "hgb.jsonl").read_text().splitlines()) == 2 * 5
    c, _ = es.load_config()
    c.update(cache=str(cache), scores=str(out / "scores"), out=str(out), reference_scores=str(ref / "scores"),
             reference_units=str(ref / "units" / "hgb.jsonl"), n_bootstrap=200)
    res = es.summary(c, "test")
    assert res["exp004_scores_reproduce_units"] and res["n_evaluable"] == 35
    pa = res["pooled_auroc"]
    lo, hi = pa["bootstrap95_recordings"]
    assert pa["inflation"] > 0.03 and lo < pa["inflation"] < hi
    wp = res["within_patient_auroc"]
    assert len(wp["bootstrap95_patients"]) == 2 and wp["inflation_repeat_range"][0] <= wp["inflation"]
    assert abs(pa["patient_grouped"] - 0.5) < 0.1
    json.dumps(res)
