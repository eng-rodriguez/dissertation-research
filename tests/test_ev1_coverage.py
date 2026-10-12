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
    monkeypatch.setattr(ev, "load_design", lambda c, scen=None: dict(full, splits=full["splits"][:1]))
    a, b = ev.dataset((cfg, "pos_sesoi", 7, "fixed")), ev.dataset((cfg, "pos_sesoi", 7, "fixed"))
    assert a == b and a["n_e"] == 35
    lo, hi = a["nb_0.95"]
    assert lo < a["estimate"] < hi and a["nb_0.90"][0] > lo and a["nb_0.90"][1] < hi
    assert ev.dataset((cfg, "pos_sesoi", 8, "fixed"))["estimate"] != a["estimate"]


def test_tuned_mode_runs(monkeypatch):
    cfg = small_cfg()
    full = ev.load_design(cfg)
    monkeypatch.setattr(ev, "load_design", lambda c, scen=None: dict(full, splits=full["splits"][:1]))
    monkeypatch.setattr(ev, "GRID", ev.GRID[:2])
    r = ev.dataset((cfg, "null_hetero", 0, "tuned"))
    assert r["mode"] == "tuned" and np.isfinite(r["estimate"])


def test_theta_draw_is_deterministic():
    cfg = small_cfg()
    cfg["truth"]["test_patients_per_set"] = 20
    assert ev.theta_draw((cfg, "null_homog", 3, None)) == ev.theta_draw((cfg, "null_homog", 3, None))


def test_resume_drops_truncated_line_and_counts_repeats_once(tmp_path):
    p = tmp_path / "s.jsonl"
    r0, r1 = {"i": 0, "estimate": 0.1}, {"i": 1, "estimate": 0.2}
    p.write_text(json.dumps(r0) + "\n" + json.dumps(r1) + "\n" + json.dumps(r0) + "\n" + '{"i": 2, "esti')
    assert ev.read_results(p) == {0: r0, 1: r1}
    assert p.read_text().endswith('{"i": 2, "esti')  # read-only by default
    ev.read_results(p, repair=True)
    assert p.read_text().endswith(json.dumps(r0) + "\n")
    p.write_text(json.dumps(r0) + "\n" + json.dumps({"i": 0, "estimate": 0.3}) + "\n")
    with pytest.raises(RuntimeError):
        ev.read_results(p)


def test_m1_adds_patient_sampling_variance_to_nb():
    sys.path.insert(0, str(ROOT / "src" / "evaluation"))
    import taxonomy as tx
    rng = np.random.default_rng(5)
    fm, dp = rng.normal(0.01, 0.02, 50), rng.normal(0.01, 0.05, 35)
    lo, hi = tx.m1_ci(0.01, fm, dp, 0.90)
    se = np.sqrt((1 / 50 + 0.25) * fm.var(ddof=1) + dp.var(ddof=1) / 35)
    assert hi - 0.01 == pytest.approx(se * __import__("scipy").stats.t.ppf(0.95, 34), rel=1e-12)
    nlo, nhi = tx.nadeau_bengio_ci(0.01, fm, 35, 0.90)
    assert lo < nlo and hi > nhi


def test_demotion_gives_boundary_n_e_and_keeps_every_test_fold_evaluable():
    cfg = copy.deepcopy(CFG)
    cfg["scenarios"]["boundary"] = {"d_F": 24, "gamma": 2.5, "tau": 0.4, "demote_e_to_n": 2}
    base, d = ev.load_design(cfg), ev.load_design(cfg, "boundary")
    e0 = {i for i, p in base["profiles"].items() if p["group"] == "E"}
    e1 = {i for i, p in d["profiles"].items() if p["group"] == "E"}
    assert len(e1) == 33 and e1 < e0
    n_ied = {i: sum(base["profiles"][i]["counts"][s][0] for s in base["profiles"][i]["evaluable"]) for i in e0}
    assert max(n_ied[i] for i in e0 - e1) <= min(n_ied[i] for i in e1)
    for i in e0 - e1:
        assert d["profiles"][i]["counts"] == base["profiles"][i]["counts"] and not d["profiles"][i]["evaluable"]
    assert all(sum(f[i] == k for i in e1) >= 1 for _, _, f in d["splits"] for k in range(5))


def test_exp002_dataset_reproduces_after_m1_changes():
    """The committed EXP-002 output for null_homog dataset 0 (Josue's Mac) is reproduced: same estimate and
    fold variance, NB intervals equal up to floating-point rounding across platforms (about 1e-17)."""
    path = ROOT / "artifacts" / "protocol" / "ev1" / "fixed" / "null_homog.jsonl"
    if not path.exists():
        pytest.skip("EXP-002 outputs not present")
    ref = next(r for r in map(json.loads, path.open()) if r["i"] == 0)
    got = ev.dataset((CFG, "null_homog", 0, "fixed"))
    assert set(got) == set(ref)
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-12, abs=1e-15) if not isinstance(v, str) else got[k] == v
