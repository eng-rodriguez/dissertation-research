"""EXP-016 locked descriptive analyses on a synthetic cache, events and score files (no fitting)."""

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
import confirmatory as cf  # noqa: E402
import exp016_locked_descriptive as e16  # noqa: E402


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Evaluable recordings get 12 IED + 40 non-IED epochs per evaluable state (sleep first, then wake), others 2 + 20.
    Even-numbered evaluable recordings have 6 generalized and 6 focal IED epochs per state, the rest only focal.
    One IED marker sits at the start of every IED epoch. Scores: B2 + FC separates generalized IEDs better than
    B2 does; both score non-IED epochs within 30 s of a marker higher (harder negatives)."""
    d = cf.design()
    two = dict(list(d["splits"].items())[:2])
    monkeypatch.setattr(cf, "design", lambda manifest=None, sets=("primary",): {"recs": d["recs"], "splits": two})
    cache, scores = tmp_path / "cache", tmp_path / "scores"
    cache.mkdir()
    scores.mkdir()
    rng = np.random.default_rng(0)
    ep_rows, ev_rows, meta = [], [], {}
    ids = sorted(next(iter(two.values())))
    for n, i in enumerate(ids):
        ev = d["recs"][i]["evaluable_states"]
        y, s, t = [], [], []
        for st in ("sleep", "wake"):
            n_i, n_n = (12, 40) if st in ev else ((2, 20) if d["recs"][i]["group"] != "F" else (0, 20))
            types = ([1] * 6 + [2] * 6) if (n % 2 == 0 and st in ev) else [3] * n_i
            y += [1] * n_i + [0] * n_n
            t += types[:n_i] + [0] * n_n
            s += [st] * (n_i + n_n)
        y, t = np.array(y), np.array(t)
        order = rng.permutation(len(y))
        y, t, s = y[order], t[order], np.array(s)[order]
        start = np.arange(len(y)) * 2000 * 3  # epochs 12 s apart
        np.savez_compressed(cache / f"{i}.npz", label=y, state=s, epoch_start=start,
                            fc=rng.random((len(y), 513)), b1=rng.random((len(y), 19, 24)))
        ep_rows += [(i, int(a), int(b)) for a, b in zip(start, t)]
        ev_rows += [(i, k, a / 500, 0, "!") for k, a in enumerate(start[y == 1])]
        tm = start / 500
        near = (np.abs(tm[:, None] - tm[y == 1][None]) <= 16).any(axis=1)  # within 12 s of a marker, edge to marker
        meta[i] = (y, t, near)
    with (tmp_path / "epochs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eeg_id", "start", "label"])
        w.writerows(ep_rows)
    with (tmp_path / "events.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eeg_id", "row", "onset_s", "duration", "text"])
        w.writerows(ev_rows)
    for m in ("b2", "b2_fc"):
        for (rep, _), fs in two.items():
            for k in range(5):
                arrs = {}
                for i, fo in fs.items():
                    if fo == k:
                        y, t, near = meta[i]
                        g = 1.0 + (1.0 if (m == "b2_fc") else 0.0) * (t == 1)
                        arrs[i] = g * y + 1.0 * (near & (y == 0)) + rng.standard_normal(len(y))
                np.savez_compressed(scores / f"hgb_{m}_r{rep}_k{k}.npz", **arrs)
    c, _ = e16.load_config()
    c.update(cache=str(cache), scores=str(scores), epochs=str(tmp_path / "epochs.csv"),
             events=str(tmp_path / "events.csv"), n_bootstrap=500)
    L = e16.dx.load(cache, scores, "hgb", ["b2", "b2_fc"])
    base = {i: r["evaluable_states"] for i, r in L["recs"].items() if r.get("evaluable_states") and i in L["data"]}
    want = e16.est.delta_summary(e16.dx.aurocs(L, cf.PRIMARY, base), *cf.PRIMARY, L["folds"])["delta_hat"]
    ref = tmp_path / "ref.json"
    ref.write_text('{"contrasts": {"b2_fc_vs_b2": {"delta_hat": %r}}}' % want)
    c["reference_summary"] = str(ref)
    return c, L


def test_config_names_the_locked_protocol_and_is_not_frozen_yet():
    c, sha = e16.load_config()
    assert c["protocol"]["sha256"] == hashlib.sha256((ROOT / c["protocol"]["document"]).read_bytes()).hexdigest()
    assert e16.FROZEN_SHA256 is None
    with pytest.raises(SystemExit):
        e16.check_frozen(sha)


def test_distance_to_nearest_marker():
    L = {"data": {"x": {"start": np.array([0, 2000, 20000, 100000])}}}
    ev = Path(__file__).with_name("_tmp_events_e16.csv")
    ev.write_text("eeg_id,row,onset_s,duration,text\nx,0,6.0,0,!\nx,1,30,0,!start\nx,2,50,0,!end\n")
    try:
        d = e16.distances(L, ev)["x"]
    finally:
        ev.unlink()
    # epochs [0,4) [4,8) [40,44) [200,204): point at 6 s, run 30-50 s
    assert np.allclose(d, [2.0, 0.0, 0.0, 150.0])


def test_rq4_and_distance_bins_recover_planted_effects(synthetic):
    c, L = synthetic
    res = e16.run(c)
    assert res["check"]["matches"]
    rq4 = res["rq4_within_patient"]
    assert rq4["n_patients_both_types"] >= 10
    assert rq4["mean_difference"] > 0 and rq4["bootstrap95"][0] > 0
    db = res["distance_bins"]
    near, far = db["[0, 30)"], db["[30, 120)"]
    assert near["n_patients"] > 0 and far["n_patients"] > 0
    assert near["b2_mean_within_patient_auroc"] < far["b2_mean_within_patient_auroc"]
    assert "descriptive_label_m1" not in near["b2_fc_vs_b2"]
