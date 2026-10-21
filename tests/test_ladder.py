"""Comparator ladder: exact fast binning, R9 weights, nested selection, B0, and the estimand (06 §4.1, R2, R9)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in ("models", "evaluation", "design"):
    sys.path.insert(0, str(ROOT / "src" / p))
import estimand as est  # noqa: E402
import ev1_coverage as ev  # noqa: E402
import fastbin as fb  # noqa: E402
import ladder as L  # noqa: E402


def synthetic_patients(n_pat=10, n=60, seed=0):
    rng = np.random.default_rng(seed)
    pats = []
    for i in range(n_pat):
        y = (rng.random(n) < 0.35).astype(int)
        s = np.where(rng.random(n) < 0.6, "sleep", "wake")
        b1 = rng.standard_normal((n, 19, 24)) + 0.4 * y[:, None, None]
        F = {"b1": b1, "b2_count": rng.integers(0, 19, n).astype(float)[:, None],
             "b2_topo": rng.standard_normal((n, 17)), "adj": rng.standard_normal((n, 912)),
             "fc": rng.random((n, 513)) + 0.05 * y[:, None], "fc_surr": rng.random((n, 513))}
        evs = [st for st in ("wake", "sleep") if ((y == 1) & (s == st)).sum() >= 5 and ((y == 0) & (s == st)).sum() >= 5]
        pats.append({"id": f"P{i}", "y": y, "s": s, "evaluable": evs, "F": F})
    return pats


def test_fast_binning_is_bit_identical_to_sklearn():
    rng = np.random.default_rng(3)
    for trial in range(60):
        n = int(rng.integers(260, 4000))
        x = rng.standard_normal(n) * rng.choice([1, 10, 1e-3])
        if trial % 3 == 1:
            x = np.round(x, 1)
        if trial % 3 == 2:
            x = rng.integers(0, 300, n).astype(float)
        w = rng.random(n) * rng.choice([1, 1e-3, 1e3])
        w[rng.random(n) < 0.05] = 0
        a, b = fb.ORIGINAL(x.copy(), 255, w.copy()), fb.find_binning_thresholds(x.copy(), 255, w.copy())
        assert a.shape == b.shape and np.array_equal(a, b)


def test_r9_weights_equal_the_ev1_implementation():
    pats = synthetic_patients()
    evp = [{"y": p["y"], "s": np.array([ev.STATES.index(x) for x in p["s"]])} for p in pats]
    assert np.allclose(L.r9_weights(pats), ev.train_weights(evp), rtol=0, atol=1e-15)


def test_design_matrix_dimensions():
    p = synthetic_patients(1)[0]
    p["F"]["fc_delta"] = np.zeros((len(p["y"]), 171))  # 4-s sensitivity cache only (EXP-010)
    p["F"]["fc_ns"] = np.zeros((len(p["y"]), 60))  # derived in confirmatory.patient (EXP-017)
    dims = {m: L.design_matrix(p, m).shape[1:] for m in L.MODELS}
    assert dims == {"b0": (19, 24), "b1": (456,), "b2": (474,), "b2_adj": (1386,), "b2_fcsurr": (987,),
                    "b2_fc": (987,), "pc1_fc_only": (513,), "b2_fc_delta": (1158,), "b2_ns": (534,)}


def test_patient_auroc_matches_sklearn_per_state_and_ied_weights():
    rng = np.random.default_rng(4)
    y = np.array([1] * 6 + [0] * 9 + [1] * 10 + [0] * 7)
    s = np.array(["wake"] * 15 + ["sleep"] * 17)
    sc = rng.standard_normal(y.size)
    a_w = roc_auc_score(y[s == "wake"], sc[s == "wake"])
    a_s = roc_auc_score(y[s == "sleep"], sc[s == "sleep"])
    assert est.patient_auroc(sc, y, s, ["wake", "sleep"]) == pytest.approx((6 * a_w + 10 * a_s) / 16)
    assert est.patient_auroc(sc, y, s, ["sleep"]) == pytest.approx(a_s)


def test_delta_summary_cells_and_patient_means():
    auroc, folds = {}, {}
    for r in range(2):
        for k, p in enumerate(["A", "B", "C", "D"]):
            folds[(r, p)] = (k + r) % 2
            auroc[("b2", r, p)] = 0.8
            auroc[("b2_fc", r, p)] = 0.8 + 0.01 * (k + 1) + 0.001 * r
    d = est.delta_summary(auroc, "b2_fc", "b2", folds)
    assert d["delta_p"]["A"] == pytest.approx(0.0105) and d["n_e"] == 4 and d["n_cells"] == 4
    assert d["delta_hat"] == pytest.approx(np.mean([0.0105, 0.0205, 0.0305, 0.0405]))


def test_b0_scores_are_max_over_channels():
    pats = synthetic_patients(4, n=40)
    m = L.fit(pats[:3], "b0", "hgb", dict(learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20,
                                           l2_regularization=0), 1)
    X = pats[3]["F"]["b1"]
    per = m.decision_function(X.reshape(-1, 24)).reshape(len(X), 19)
    assert np.allclose(L.score(m, pats[3], "b0"), per.max(axis=1))


def test_selection_is_deterministic_and_ties_go_to_simpler(monkeypatch):
    pats = synthetic_patients(6, n=40)
    inner = {p["id"]: i % 2 for i, p in enumerate(pats)}
    monkeypatch.setattr(L.est, "patient_auroc", lambda *a, **k: 0.5)  # every configuration ties
    assert L.select(pats, inner, "pc1_fc_only", "lr", 1) == {"C": 1e-3}
    assert L.select(pats, inner, "pc1_fc_only", "hgb", 1) == min(L.HGB_GRID, key=lambda c: L.simplicity_key("hgb", c))


def test_outer_split_is_reproducible_and_only_scores_evaluable_test_patients():
    pats = synthetic_patients(8, n=50)
    pats[-1]["evaluable"] = []
    fixed = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1)
    a, _ = L.outer_split(pats[:5], pats[5:], "b2_fc", "hgb", 11, fixed=fixed)
    b, _ = L.outer_split(pats[:5], pats[5:], "b2_fc", "hgb", 11, fixed=fixed)
    assert a == b and set(a) == {p["id"] for p in pats[5:7] if p["evaluable"]}
