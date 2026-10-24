"""EXP-009 deployment metrics on a synthetic cache with the real frozen design (one repeat, LR for speed)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tests"))
import confirmatory as cf  # noqa: E402
import deployment as dp  # noqa: E402
from test_confirmatory import fake_cache  # noqa: E402


def test_threshold_respects_fp_budget_and_ties():
    neg = np.array([5.0, 4.0, 4.0, 3.0, 2.0, 1.0] * 10)  # 60 epochs of 4 s = 4 minutes
    for t in (0.5, 1, 2, 100):
        th = dp.threshold_for(neg, t, 4)
        assert (neg >= th).sum() <= np.floor(t * 4)
    assert dp.threshold_for(neg, 0, 4) > 5.0


def test_run_reproduces_exp004_scores_and_summarises(tmp_path, monkeypatch):
    d = cf.design()
    one = dict(list(d["splits"].items())[:1])
    monkeypatch.setattr(cf, "design", lambda manifest=None, sets=("primary",): {"recs": d["recs"], "splits": one})
    cache, saved, probs, out, units = (tmp_path / n for n in ("cache", "saved", "probs", "out", "units"))
    for x in (cache, saved, units):
        x.mkdir()
    fake_cache(cache, gain=0.5)
    cfg = {"C": 1.0}
    rows = []
    for m in ("b2", "b2_fc"):
        for (rep, seed), folds in one.items():
            for k in range(5):
                train = [cf.patient(cache, i, cf.blocks_for(m), d["recs"][i]["evaluable_states"], 5.0)
                         for i in sorted(folds) if folds[i] != k]
                model = cf.L.fit(train, m, "lr", cfg, seed + k)
                sc = {i: cf.L.score(model, cf.patient(cache, i, cf.blocks_for(m), d["recs"][i]["evaluable_states"],
                                                      5.0), m) for i in sorted(folds) if folds[i] == k}
                np.savez_compressed(saved / f"lr_{m}_r{rep}_k{k}.npz", **sc)
                rows.append(json.dumps({"model": m, "repeat": rep, "outer_fold": k, "config": cfg}))
    (units / "lr.jsonl").write_text("\n".join(rows) + "\n")
    dp.run(cache, saved, probs, "lr", 5.0, 1, out=out, exp004_units=units)
    res = dp.summary("lr", out=out, prob_dir=probs, cache=cache)
    assert res["n_units"] == 10 and res["max_refit_diff"] <= 1e-9
    s1 = res["sensitivity"]["1"]
    assert 0 <= s1["mean_ref"] <= 1 and s1["n"] > 0 and "0.90" in s1["intervals"]
    assert res["sensitivity"]["2"]["mean_model"] >= res["sensitivity"]["0.5"]["mean_model"]
    assert set(res["fp_per_min"]["1"]["all"]) == {"b2", "b2_fc"}
    assert set(res["calibration"]) == {"b2", "b2_fc"} and len(res["calibration"]["b2"]["reliability"]) == 10
    np.savez_compressed(saved / "lr_b2_r1_k0.npz", **{i: v + 1 for i, v in np.load(saved / "lr_b2_r1_k0.npz").items()})
    (out / "units" / "lr.jsonl").unlink()
    with pytest.raises(RuntimeError, match="refit differs"):
        dp.run(cache, saved, probs, "lr", 5.0, 1, out=out, exp004_units=units)


def test_frozen_exp009_config_is_unchanged():
    """EXP-009 was frozen 2026-10-17 (run plan A, approved by Josue) before any deployment metric existed."""
    import hashlib

    assert hashlib.sha256(dp.CONFIG.read_bytes()).hexdigest() == \
        "227449a58b2982eb028af77c1da271906e6c0be693f9cc20a70165d8067458a8"
