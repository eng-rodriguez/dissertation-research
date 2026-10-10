"""Checks inner-fold derivation and the inner-fold validity rule on synthetic tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "design"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3_pilot_and_folds as f3
import inner_folds as inf
from test_f3_pilot_and_folds import _tables


def _setup(tmp_path, **kw):
    recs = f3.load_recordings(*_tables(tmp_path, **kw))
    pilot = {r["eeg_id"] for r in f3.draw_pilot(recs)[0]}
    rows = [r for r in f3.build_manifest(recs, pilot) if r["repeat"] == 1 and r["analysis_set"] == "primary"]
    return recs, rows


def test_inner_folds_partition_training_set_and_are_deterministic(tmp_path):
    recs, rows = _setup(tmp_path, n_e=40, n_n=5, n_f=25)
    train = sorted(r["eeg_id"] for r in rows if r["outer_fold"] != 0)
    test = [r["eeg_id"] for r in rows if r["outer_fold"] == 0]
    inner, n, bad = inf.derive(train, recs, rows[0]["seed"], 0, test)
    assert n == 4 and bad == [] and set(inner) == set(train) and set(inner.values()) == set(range(4))
    assert inf.derive(train, recs, rows[0]["seed"], 0, test)[0] == inner
    assert inf.derive(train, recs, rows[0]["seed"], 1, test)[0] != inner  # seed depends on outer fold


def test_too_few_evaluable_patients_falls_back_then_fails(tmp_path):
    recs, rows = _setup(tmp_path, n_e=30, n_n=5, n_f=25)
    train = sorted(r["eeg_id"] for r in rows if r["outer_fold"] != 0)
    n_e = sum(recs[i]["group"] == "E" for i in train)
    assert 15 <= n_e < 20  # 4 folds would leave < 5 E in some fold; 3 folds suffice
    _, n, bad = inf.derive(train, recs, rows[0]["seed"], 0)
    assert n == 3 and bad == []
    few = [i for i in train if recs[i]["group"] != "E"] + [i for i in train if recs[i]["group"] == "E"][:10]
    _, n, bad = inf.derive(sorted(few), recs, rows[0]["seed"], 0)
    assert n == 3 and any("evaluable patients in validation" in b for b in bad)


def test_violations_detect_overlap_with_test_and_missing_state(tmp_path):
    recs, rows = _setup(tmp_path, n_e=40, n_n=5, n_f=25)
    train = sorted(r["eeg_id"] for r in rows if r["outer_fold"] != 0)
    inner, n, _ = inf.derive(train, recs, rows[0]["seed"], 0)
    assert inf.violations(train, inner, n, recs, test_ids=[train[0]])
    sleep_only = [i for i in train if recs[i]["evaluable_states"] == ["sleep"]]
    sub = {i: inner[i] for i in sleep_only}
    assert any("no wake-evaluable" in b for b in inf.violations(sleep_only, sub, n, recs))
