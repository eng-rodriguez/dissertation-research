"""EXP-004 descriptive analyses on a synthetic cache and synthetic score files (no fitting)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tests"))
import confirmatory as cf  # noqa: E402
import exp004_descriptive as dx  # noqa: E402
from test_confirmatory import fake_cache  # noqa: E402


def test_descriptive_reproduces_primary_and_reports_all_parts(tmp_path, monkeypatch):
    d = cf.design()
    two = dict(list(d["splits"].items())[:2])
    monkeypatch.setattr(cf, "design", lambda manifest=None: {"recs": d["recs"], "splits": two})
    cache, scores = tmp_path / "cache", tmp_path / "scores"
    cache.mkdir()
    scores.mkdir()
    fake_cache(cache, gain=0.3)  # FC (wPLI-like) columns shifted up in IED windows
    rng = np.random.default_rng(1)
    for m in cf.L.MODELS:
        gain = 1.2 if m == "b2_fc" else 1.0
        for (rep, _), fs in two.items():
            for k in range(5):
                arrs = {}
                for i, f in fs.items():
                    if f == k:
                        with np.load(cache / f"{i}.npz") as z:
                            arrs[i] = gain * z["label"] + rng.standard_normal(len(z["label"]))
                np.savez_compressed(scores / f"hgb_{m}_r{rep}_k{k}.npz", **arrs)
    L = dx.load(cache, scores, "hgb", ["b2", "b2_fc"])
    base = {i: r["evaluable_states"] for i, r in L["recs"].items() if r.get("evaluable_states") and i in L["data"]}
    want = dx.est.delta_summary(dx.aurocs(L, cf.PRIMARY, base), *cf.PRIMARY, L["folds"])["delta_hat"]
    res = dx.run(cache, scores, "hgb", {"contrasts": {"b2_fc_vs_b2": {"delta_hat": want}}})
    assert res["check"]["matches_summary"]
    assert set(res["per_state"]) == {"wake", "sleep", "note"}
    assert res["per_state"]["sleep"]["b2_fc_vs_b2"]["n_e"] > res["per_state"]["wake"]["b2_fc_vs_b2"]["n_e"]
    assert res["thresholds"][">=3"]["n_e"] >= res["thresholds"][">=10"]["n_e"]
    assert abs(res["epoch_weighted"]["delta_equal_weight"] - want) < 1e-12
    assert res["pooled"]["sleep"]["d_auroc"]["mean"] > 0
    assert all(res["pc2"][b]["median_diff"] > 0 for b in ("theta", "alpha", "beta"))
    ps = [res["pc2"][b]["p_wilcoxon_holm"] for b in ("theta", "alpha", "beta")]
    assert all(0 <= p <= 1 for p in ps)
