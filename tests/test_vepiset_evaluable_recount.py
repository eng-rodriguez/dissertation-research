"""Checks the R3 post-exclusion evaluability recount on synthetic tables (no signal)."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "design"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inner_folds as inf
import vepiset_evaluable_recount as rc
from test_f3_pilot_and_folds import _tables


def _ep(start, label=0, state="wake", ev=None, length=2000):
    return {"start": start, "end": start + length, "label": label, "state": state,
            "has_ied_event": int(label > 0) if ev is None else ev}


def test_classify_reasons_in_order():
    eps = [_ep(0), _ep(2000, 2), _ep(4000), _ep(6000), _ep(8000, 2, ev=0), _ep(10000, ev=1),
           _ep(12000, state="mixed"), _ep(14000), _ep(16000), _ep(18000, length=500)]
    flags = [False] * 10
    flags[7] = True
    assert rc.classify(eps, flags) == ["edge", None, "adjacent", "adjacent", "ied_no_marker", "nonied_marker",
                                      "state", "epoch_qc", "edge", "short"]


def _interim(tmp_path, n_e=30, n_rec=60, drop=()):
    """Grid positions: edges 0 and 40 (non-IED); for E recordings IED at 2, 6, ..., 30 and non-adjacent
    non-IED at 4, 8, ..., 28 plus 33, 35 (9 in all); recordings in drop keep only 4 usable non-IED."""
    d = tmp_path / "interim"
    d.mkdir()
    rows = []
    for k in range(n_rec):
        rid = f"R{k:03d}"
        ied = list(range(2, 31, 4)) if k < n_e else []
        non = list(range(4, 29, 4)) + [33, 35]
        if rid in drop:
            non = non[:4]
        for pos, label in sorted([(0, 0), (40, 0)] + [(q, 2) for q in ied] + [(q, 0) for q in non]):
            rows.append([rid, pos * 2000, pos * 2000 + 2000, 500, label, "f", "wake", int(label > 0)])
    with (d / "epochs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eeg_id", "start", "end", "sr", "label", "folder", "state", "has_ied_event"])
        w.writerows(rows)
    return d


def test_run_counts_change_and_rule(tmp_path):
    qc_path, sm_path = _tables(tmp_path, n_e=30, n_n=5, n_f=25, n_bad=6)
    pilot = tmp_path / "pilot.csv"
    pilot.write_text("eeg_id,group,coverage\nR010,E,wake\n")
    no_flags = lambda rid, eps: [False] * len(eps)  # noqa: E731
    s = rc.run(_interim(tmp_path, drop={"R020", "R021"}), tmp_path / "out", qc_path, sm_path, pilot, no_flags)
    assert s["lost_evaluability"] == ["R020", "R021"]
    assert s["confirmatory_evaluable_after"] == s["confirmatory_evaluable_before"] - 2
    assert s["rule"]["result"].startswith("RDR REQUIRED")  # synthetic cohort is far below 33
    assert s["excluded_epochs_by_reason_and_class"]["edge|nonied"] == 120
    rows = list(csv.DictReader((tmp_path / "out" / "evaluable_recount.csv").open()))
    assert list(rows[0]) == rc.FIELDS
    json.loads((tmp_path / "out" / "evaluable_recount_summary.json").read_text())

    recs = inf.f3.load_recordings(qc_path, sm_path)
    assert inf.apply_recount(recs, tmp_path / "out" / "evaluable_recount.csv") >= 2
    assert recs["R020"]["group"] == "N" and recs["R020"]["evaluable_states"] == []


def test_signal_epoch_flags_uses_f4_code(tmp_path):
    from test_vepiset_channel_qc import _write_dataset
    root, _ = _write_dataset(tmp_path)
    eps = [{"start": s, "end": s + 2000} for s in (0, 2000, 4000)]
    flags = rc.signal_epoch_flags(root)("REC1", eps)
    assert flags == [False, False, False]
