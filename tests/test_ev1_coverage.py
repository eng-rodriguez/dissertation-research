"""Fast checks of the EV-1 (EXP-002) simulation code: metric, weights, design inputs, determinism.

These test the implementation only; they compute no coverage and say nothing about EV-1's verdict.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "design"))
import ev1_coverage as ev  # noqa: E402

CFG = json.loads((ROOT / "experiments" / "EXP-002-ev1" / "config.json").read_text())


def small_cfg(gamma=0.5):
    c = copy.deepcopy(CFG)
    c["design"]["max_iter"] = 5
    for s in c["scenarios"].values():
        if s["gamma"] is None:
            s["gamma"] = gamma
    return c


def test_mann_whitney_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.integers(0, 5, 200).astype(float)  # heavy ties
    assert ev.mw_auroc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_fixed_config_is_simplest_grid_point():
    assert len(ev.GRID) == 16
    assert min(ev.GRID, key=ev.simplicity_key) == CFG["design"]["fixed_config"]


def test_design_matches_post_f6_primary_set():
    d = ev.load_design(CFG)
    groups = [p["group"] for p in d["profiles"].values()]
    assert (groups.count("E"), groups.count("N"), groups.count("F")) == (35, 7, 27)
    assert len(d["splits"]) == 10 and all(len(set(f.values())) == 5 for _, _, f in d["splits"])
    for p in d["profiles"].values():
        assert bool(p["evaluable"]) == (p["group"] == "E")
        for s in p["evaluable"]:
            n_ied, n_non = p["counts"][s]
            assert n_ied >= 5 and n_non >= 5
        for n_ied, n_non in p["counts"].values():
            assert n_ied <= CFG["dgp"]["ied_cap_per_state"] and n_non <= CFG["dgp"]["nonied_cap_per_state"]


def test_inner_folds_valid_for_every_outer_split():
    d, recs = ev.load_design(CFG), ev.inner_recs(CFG)
    ids = sorted(d["profiles"])
    for _, rseed, folds in d["splits"]:
        for k in range(5):
            tr = [r for r in ids if folds[r] != k]
            _, n, bad = ev.inf.derive(tr, recs, rseed, k, [r for r in ids if folds[r] == k])
            assert not bad and n == 4


def test_weights_follow_r9():
    d = ev.load_design(CFG)
    rng = np.random.default_rng(3)
    pats = [ev.make_patient(rng, p, CFG["dgp"], CFG["scenarios"]["null_homog"]) for p in d["profiles"].values()]
    w = ev.train_weights(pats)
    y = np.concatenate([p["y"] for p in pats])
    s = np.concatenate([p["s"] for p in pats])
    for k in range(2):  # negatives' state shares equal positives' after reweighting
        assert w[(y == 0) & (s == k)].sum() / w[y == 0].sum() == pytest.approx(
            w[(y == 1) & (s == k)].sum() / w[y == 1].sum(), rel=1e-12)
    pos_part = np.concatenate([np.where(p["y"] == 1, 1.0 / p["y"].size, 0) for p in pats])
    assert np.allclose(w[y == 1], pos_part[y == 1])  # positives keep 1/n_p


def test_patients_follow_profiles():
    d = ev.load_design(CFG)
    rng = np.random.default_rng(4)
    for prof in d["profiles"].values():
        p = ev.make_patient(rng, prof, CFG["dgp"], CFG["scenarios"]["null_highvar"])
        assert p["X"].shape[1] == CFG["dgp"]["d_base"] + 96
        for k, s in enumerate(ev.STATES):
            m = p["s"] == k
            assert (int(p["y"][m].sum()), int((1 - p["y"][m]).sum())) == prof["counts"][s]


def test_dataset_end_to_end_is_deterministic(monkeypatch):
    cfg = small_cfg()
    full = ev.load_design(cfg)
    monkeypatch.setattr(ev, "load_design", lambda c: dict(full, splits=full["splits"][:1]))
    a, b = ev.dataset((cfg, "pos_sesoi", 7, "fixed")), ev.dataset((cfg, "pos_sesoi", 7, "fixed"))
    assert a == b and a["n_e"] == 35
    lo, hi = a["nb_0.95"]
    assert lo < a["estimate"] < hi and a["nb_0.90"][0] > lo and a["nb_0.90"][1] < hi
    assert ev.dataset((cfg, "pos_sesoi", 8, "fixed"))["estimate"] != a["estimate"]


def test_tuned_mode_runs(monkeypatch):
    cfg = small_cfg()
    full = ev.load_design(cfg)
    monkeypatch.setattr(ev, "load_design", lambda c: dict(full, splits=full["splits"][:1]))
    monkeypatch.setattr(ev, "GRID", ev.GRID[:2])
    r = ev.dataset((cfg, "null_hetero", 0, "tuned"))
    assert r["mode"] == "tuned" and np.isfinite(r["estimate"])


def test_theta_draw_is_deterministic():
    cfg = small_cfg()
    cfg["truth"]["test_patients_per_set"] = 20
    assert ev.theta_draw((cfg, "null_homog", 3, None)) == ev.theta_draw((cfg, "null_homog", 3, None))
