"""Run-plan robustness analyses (06 §5): bipolar montage (EXP-011) features, frozen config, summary."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("features", "preprocessing", "pipeline"):
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
