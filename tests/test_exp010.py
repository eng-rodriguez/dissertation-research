"""EXP-010 (native 4-s sensitivity incl. delta FC): extraction blocks, frozen config, summary and reading."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("features", "preprocessing", "pipeline"):
    sys.path.insert(0, str(ROOT / "src" / p))
import confirmatory as cf  # noqa: E402
import exp010_epoch4s as e10  # noqa: E402
import extract as ex  # noqa: E402


def test_epoch4s_adds_delta_block_and_leaves_fc_unchanged():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((19, 8 * 2000))
    eps = [{"start": i * 2000, "end": (i + 1) * 2000, "label": 0, "state": "wake", "has_ied_event": 0}
           for i in range(1, 7)]
    f4 = ex.features(x, "SYN", eps, mode="epoch4s", with_surrogates=False)
    f2 = ex.features(x, "SYN", eps, mode="salience2s", with_surrogates=False)
    assert f4["fc"].shape == (6, 513) and f4["fc_delta"].shape == (6, 171)
    assert "fc_delta" not in f2  # the primary cache is unchanged
    assert (f4["window_start"] == np.array([e["start"] for e in eps])).all()
    direct = ex.wp.real_fc(x, f4["window_start"], 2000, bands=("delta", "theta"))
    assert np.array_equal(f4["fc_delta"], direct["delta"]["wpli"])
    assert np.array_equal(f4["fc"][:, :171], direct["theta"]["wpli"])


def test_frozen_exp010_config_is_unchanged():
    sha = hashlib.sha256(e10.CONFIG.read_bytes()).hexdigest()
    assert e10.FROZEN_SHA256 is not None and sha == e10.FROZEN_SHA256
    c = json.loads(e10.CONFIG.read_bytes())
    assert c["unchanged_from_exp004"]["cohort_and_folds"]["manifest_sha256"] == \
        hashlib.sha256((ROOT / "artifacts/protocol/f3/fold_manifest.csv").read_bytes()).hexdigest()
    assert c["unchanged_from_exp004"]["z_star"]["sha256"] == \
        hashlib.sha256((ROOT / "artifacts/pilot/zstar.json").read_bytes()).hexdigest()
    assert c["protocol"]["sha256"] == hashlib.sha256((ROOT / c["protocol"]["document"]).read_bytes()).hexdigest()
    assert set(c["models"]) <= set(cf.L.MODELS) and "b2_fc_delta" in c["models"]


def fake_units(out: Path, models, shift: float) -> None:
    d = cf.design()
    rng = np.random.default_rng(1)
    base = {i: 0.8 + 0.1 * rng.random() for i in d["recs"]}
    lines = []
    for m in models:
        for (rep, seed), folds in d["splits"].items():
            for k in range(5):
                au = {}
                for i, f in folds.items():
                    if f == k and d["recs"][i].get("evaluable_states"):
                        noise = 0.002 * rng.standard_normal()
                        au[i] = base[i] + (shift if m in ("b2_fc", "b2_fc_delta") else 0.0) + noise
                lines.append(json.dumps({"model": m, "learner": "hgb", "repeat": rep, "seed": seed,
                                         "outer_fold": k, "config": {}, "auroc": au}))
    (out / "units").mkdir(parents=True)
    (out / "units" / "hgb.jsonl").write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("shift,want", [(0.0, "consistent"), (0.08, "discordant (positive beyond SESOI)")])
def test_summary_reading_and_concordance(tmp_path, monkeypatch, shift, want):
    c, sha = e10.load_config()
    c = {**c, "out": str(tmp_path)}
    fake_units(tmp_path, c["models"], shift)
    res = e10.summary(c, sha)
    m = res["main"]["b2_fc_vs_b2"]
    assert m["n_e"] == 35 and m["concordance_with_result001"] == want
    assert res["comparison"]["concordance"] == want
    assert "label" not in m and "heterogeneity" in m
    assert set(res["delta_band"]) == {"b2_fc_delta_vs_b2_fc", "b2_fc_delta_vs_b2"}
    e10.write_figures(c, res)
    assert (tmp_path / "figures" / "forest_b2_fc_vs_b2.svg").exists()


def test_run_refuses_unless_frozen(monkeypatch):
    monkeypatch.setattr(e10, "FROZEN_SHA256", "0" * 64)
    with pytest.raises(SystemExit):
        e10.check_frozen(e10.load_config()[1])
