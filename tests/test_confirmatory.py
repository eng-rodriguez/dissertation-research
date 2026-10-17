"""EXP-004 machinery on a synthetic cache with the real frozen design (reduced grid and repeats for speed)."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
import confirmatory as cf  # noqa: E402

FAST = [dict(learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=0)]


def fake_cache(path: Path, gain: float) -> None:
    rng = np.random.default_rng(0)
    for r in csv.DictReader(cf.RECOUNT_CSV.open()):
        if r["analysis_set"] != "primary":
            continue
        y, s = [], []
        for st in ("wake", "sleep"):
            n_i, n_n = int(r[f"n_ied_{st}"]), int(r[f"n_nonied_{st}"])
            n_i, n_n = (min(n_i, 6), min(n_n, 10)) if n_i >= 5 else (min(n_i, 2), min(n_n, 4))
            y += [1] * n_i + [0] * n_n
            s += [st] * (n_i + n_n)
        y, n = np.array(y), len(y)
        b1 = rng.standard_normal((n, 19, 24)) + 0.03 * y[:, None, None]
        fc = rng.standard_normal((n, 513)) + gain * y[:, None]
        np.savez_compressed(path / f"{r['eeg_id']}.npz", label=y, state=np.array(s), b1=b1,
                            b2_topo=rng.standard_normal((n, 17)), fc=fc, peak_z=rng.gamma(2, 3, (n, 19)))


@pytest.fixture
def small_design(monkeypatch):
    d = cf.design()
    two = dict(list(d["splits"].items())[:2])
    monkeypatch.setattr(cf, "design", lambda manifest=None, sets=("primary",): {"recs": d["recs"], "splits": two})
    monkeypatch.setattr(cf.L, "HGB_GRID", FAST)
    return d


def test_design_matches_frozen_counts():
    d = cf.design()
    assert len(d["splits"]) == 10 and all(len(f) == 69 for f in d["splits"].values())
    assert sum(r["group"] == "E" for i, r in d["recs"].items() if i in next(iter(d["splits"].values()))) == 35


def test_run_resume_and_summary(tmp_path, small_design):
    cache = tmp_path / "cache"
    cache.mkdir()
    fake_cache(cache, gain=0.08)
    out, scores = tmp_path / "out", tmp_path / "scores"
    cf.run(cache, out, scores, ["b2", "b2_fc", "pc1_fc_only"], "hgb", 5.0, 1)
    n_lines = len((out / "units" / "hgb.jsonl").read_text().splitlines())
    assert n_lines == 3 * 2 * 5
    cf.run(cache, out, scores, ["b2", "b2_fc", "pc1_fc_only"], "hgb", 5.0, 1)  # resume: nothing to do
    assert len((out / "units" / "hgb.jsonl").read_text().splitlines()) == n_lines
    res = cf.summarise(out, "hgb", "m1", pc3_pass=True)
    prim = res["contrasts"]["b2_fc_vs_b2"]
    assert prim["n_e"] == 35 and prim["n_cells"] == 10 and prim["role"].startswith("PRIMARY")
    assert prim["delta_hat"] > 0 and res["pc1"]["pass"]
    nb = cf.summarise(out, "hgb", "nb", pc3_pass=True)["contrasts"]["b2_fc_vs_b2"]["intervals"]["0.95"]
    m1 = prim["intervals"]["0.95"]
    assert m1[1] - m1[0] > nb[1] - nb[0]  # M1 adds patient-sampling variance
