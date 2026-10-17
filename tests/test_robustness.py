"""Run-plan robustness analyses (06 §5): bipolar montage (EXP-011) features, frozen config, summary."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("features", "preprocessing", "pipeline", "data"):
    sys.path.insert(0, str(ROOT / "src" / p))
sys.path.insert(0, str(ROOT / "tests"))
import extract as ex  # noqa: E402
import montage as mt  # noqa: E402
import robustness as rb  # noqa: E402
from test_exp010 import fake_units  # noqa: E402


def test_bipolar_is_reference_free_and_has_18_midpoint_derivations():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((19, 1000))
    common = rng.standard_normal(1000)
    assert mt.bipolar(x).shape == (18, 1000) and len(set(mt.DERIVATIONS)) == 18
    assert np.allclose(mt.bipolar(x + common), mt.bipolar(x))  # the shared stored reference cancels
    i, j = mt.NAMES.index("Fp1"), mt.NAMES.index("F7")
    assert np.array_equal(mt.bipolar(x)[0], x[i] - x[j])
    assert np.allclose(mt.XY[mt.DERIVATIONS.index("Cz-Pz")], [0, -0.25])


def test_bipolar_features_use_car_windows_and_derivations():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((19, 8 * 2000))
    eps = [{"start": i * 2000, "end": (i + 1) * 2000, "label": 0, "state": "wake", "has_ied_event": 0}
           for i in range(1, 7)]
    fc = ex.features(x, "SYN", eps, with_surrogates=False)
    fb = ex.features(x, "SYN", eps, with_surrogates=False, montage="bipolar")
    assert np.array_equal(fb["window_start"], fc["window_start"])  # same CAR-salience windows
    assert fb["b1"].shape == (6, 18, 24) and fb["adj"].shape == (6, 864) and fb["fc"].shape == (6, 459)
    assert fb["peak_z"].shape == (6, 18) and fb["b2_topo"].shape == (6, 17)
    xb = mt.bipolar(x)
    s0 = fb["window_start"][0]
    assert np.allclose(fb["b1"][0], ex.fb1.b1(xb[:, s0:s0 + 1000]))
    direct = ex.wp.real_fc(xb, fb["window_start"], 1000, reference="none")
    assert np.array_equal(fb["fc"][:, :153], direct["theta"]["wpli"])


def test_frozen_exp011_config_is_unchanged():
    c, sha = rb.load_config("EXP-011")
    assert rb.FROZEN_SHA256["EXP-011"] == sha
    assert c["unchanged_from_exp004"]["z_star"]["sha256"] == \
        hashlib.sha256((ROOT / "artifacts/pilot/zstar.json").read_bytes()).hexdigest()
    assert c["protocol"]["sha256"] == hashlib.sha256((ROOT / c["protocol"]["document"]).read_bytes()).hexdigest()
    assert c["unchanged_from_exp004"]["cohort_and_folds"]["manifest_sha256"] == \
        hashlib.sha256((ROOT / "artifacts/protocol/f3/fold_manifest.csv").read_bytes()).hexdigest()
    assert [d for d in c["factors_changed"]["montage"].split("(")[1].split(")")[0].split(", ")] == mt.DERIVATIONS


@pytest.mark.parametrize("shift,want", [(0.0, "consistent"), (-0.08, "discordant (negative beyond SESOI)")])
def test_summary_reading_and_concordance(tmp_path, shift, want):
    c, sha = rb.load_config("EXP-011")
    c = {**c, "out": str(tmp_path)}
    fake_units(tmp_path, c["models"], shift)
    res = rb.summary(c, sha, with_states=False)
    m = res["main"]["b2_fc_vs_b2"]
    assert m["n_e"] == 35 and m["concordance_with_result001"] == want and "label" not in m
    assert res["pc1"]["mean_auroc"] > 0.5
    rb.write_figures(c, res)
    assert (tmp_path / "figures" / "forest_b2_fc_vs_b2.svg").exists()


def test_spline_interpolation_reconstructs_smooth_fields_and_passes_the_reference():
    import interpolate as ip

    pos = ip.positions3d()
    assert np.allclose(np.linalg.norm(pos, axis=1), 1)
    f = pos @ np.array([0.3, -0.5, 0.8])
    x = np.tile(f[:, None], (1, 4))
    y = ip.interpolate(x, ["Cz", "O1"])
    assert np.abs(y - x).max() < 0.01  # a smooth (linear) scalp field is recovered
    assert np.allclose(ip.interpolate(x + 7.0, ["Fp2"]) - 7.0, ip.interpolate(x, ["Fp2"]))  # shared reference
    assert np.allclose(ip.matrix(["T3"]).sum(axis=1), 1)
    assert np.array_equal(ip.interpolate(x, []), x)
    good = [i for i in range(19) if ip.NAMES[i] not in ("Cz", "O1")]
    assert np.array_equal(y[good], x[good])  # good rows untouched


def test_design_adds_the_frozen_sensitivity_folds_only():
    import confirmatory as cf

    p, s = cf.design(), cf.design(sets=("primary", "sensitivity_only"))
    for key, folds in s["splits"].items():
        assert len(folds) == 76
        assert all(folds[i] == k for i, k in p["splits"][key].items())  # the 69 keep their frozen folds


def test_frozen_exp012_config_and_primary_patient_restriction(tmp_path):
    c, sha = rb.load_config("EXP-012")
    assert rb.FROZEN_SHA256["EXP-012"] == sha and rb.sets(c) == ("primary", "sensitivity_only")
    assert c["protocol"]["sha256"] == hashlib.sha256((ROOT / c["protocol"]["document"]).read_bytes()).hexdigest()
    c = {**c, "out": str(tmp_path)}
    fake_units(tmp_path, c["models"], 0.0, sets=rb.sets(c))
    res = rb.summary(c, sha, with_states=False)
    assert res["main"]["b2_fc_vs_b2"]["n_e"] == 40
    assert res["primary_patients_only"]["b2_fc_vs_b2"]["n_e"] == 35


def test_lift_restores_exactly_the_three_06_exclusions():
    import vepiset_evaluable_recount as rc
    import windows as wn

    # epochs 0..13: 5 IED with marker; 9 IED without marker; 11 non-IED with marker; 3 mixed state; 7 QC-flagged
    eps = [{"start": i * 2000, "end": (i + 1) * 2000, "label": int(i in (5, 9)), "state": "sleep",
            "has_ied_event": int(i in (5, 11))} for i in range(14)]
    eps[3]["state"] = "mixed"
    flags = [i == 7 for i in range(14)]
    locked = rc.classify(eps, flags)
    assert [locked[i] for i in (0, 3, 4, 6, 7, 8, 9, 10, 11, 13)] == \
        ["edge", "state", "adjacent", "adjacent", "epoch_qc", "adjacent", "ied_no_marker", "adjacent",
         "nonied_marker", "edge"]
    lifted = rc.classify(eps, flags, frozenset(rc.LIFTABLE))
    assert [i for i, r in enumerate(lifted) if r is None] == [1, 2, 4, 5, 6, 8, 9, 10, 11, 12]
    assert lifted[3] == "state" and lifted[7] == "epoch_qc" and lifted[0] == lifted[13] == "edge"
    with pytest.raises(ValueError):
        rc.classify(eps, flags, frozenset({"epoch_qc"}))
    kept = wn.kept_epochs(eps, flags, frozenset(rc.LIFTABLE))
    assert {e["start"] // 2000: e["locked_reason"] for e in kept if e["locked_reason"]} == \
        {4: "adjacent", 6: "adjacent", 8: "adjacent", 9: "ied_no_marker", 10: "adjacent", 11: "nonied_marker"}
    assert [e["start"] for e in wn.kept_epochs(eps, flags)] == [e["start"] for e in kept if not e["locked_reason"]]


def test_cache_check_with_lifted_epochs(tmp_path):
    import run_ladder as rl

    rng = np.random.default_rng(3)
    rec = tmp_path / "recount.csv"
    rec.write_text("eeg_id,analysis_set,n_ied_wake,n_nonied_wake,n_ied_sleep,n_nonied_sleep\nA,primary,3,5,2,4\n")
    y = np.array([1] * 3 + [0] * 5 + [1] * 2 + [0] * 4 + [0, 0, 1])
    s = np.array(["wake"] * 8 + ["sleep"] * 6 + ["sleep"] * 3)
    reason = np.array([""] * 14 + ["adjacent", "nonied_marker", "ied_no_marker"])
    n = len(y)
    arrays = dict(label=y, state=s, peak_z=rng.random((n, 19)), fc_zero_denominator=0,
                  **{k: rng.random((n, *v)) for k, v in rl.SHAPES.items()})
    np.savez_compressed(tmp_path / "A.npz", locked_reason=reason, **arrays)
    res = rl.cache_check(tmp_path, rec, lifted=("ied_no_marker", "nonied_marker", "adjacent"))
    assert res["result"] == "PASS" and res["lifted_epochs"] == {"adjacent_nonied": 1, "ied_no_marker_ied": 1,
                                                                "nonied_marker_nonied": 1}
    np.savez_compressed(tmp_path / "A.npz", locked_reason=np.where(reason == "adjacent", "epoch_qc", reason), **arrays)
    assert rl.cache_check(tmp_path, rec, lifted=("ied_no_marker", "nonied_marker", "adjacent"))["result"] == "FAIL"


def test_frozen_exp013_config_and_recount_override(tmp_path):
    import confirmatory as cf

    c, sha = rb.load_config("EXP-013")
    assert rb.FROZEN_SHA256["EXP-013"] == sha and tuple(c["lift"]) == ("ied_no_marker", "nonied_marker", "adjacent")
    assert c["protocol"]["sha256"] == hashlib.sha256((ROOT / c["protocol"]["document"]).read_bytes()).hexdigest()
    d = cf.design()
    ids = sorted(next(iter(d["splits"].values())))
    for j, i in enumerate(ids):  # every recording gets 6 IED + 6 non-IED sleep epochs: all become evaluable
        np.savez_compressed(tmp_path / f"{i}.npz", label=np.array([1] * 6 + [0] * 6), state=np.array(["sleep"] * 12))
    ov = rb.recount_override({**c, "cache": str(tmp_path)})
    assert len(ov) == 69 and all(v["evaluable_states"] == ["sleep"] and v["group"] == "E" for v in ov.values())
    assert rb.recount_override({**c, "lift": []}) is None
