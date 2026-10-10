"""Checks the F-1 marker-position count on synthetic annotation tables."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vepiset_marker_position as f1
from test_f3_pilot_and_folds import _tables


def test_classify_epoch_boundaries():
    t0 = 8.0
    assert f1.classify_epoch(t0, [t0 + 1.0], [])[0] == "CENTRAL"        # inclusive at 1.0 s
    assert f1.classify_epoch(t0, [t0 + 2.999], [])[0] == "CENTRAL"
    assert f1.classify_epoch(t0, [t0 + 3.0], [])[0] == "OUTER-ONLY"     # exclusive at 3.0 s
    assert f1.classify_epoch(t0, [t0 + 0.4, t0 + 3.6], [])[0] == "OUTER-ONLY"
    assert f1.classify_epoch(t0, [t0 + 0.4, t0 + 2.0], [])[0] == "CENTRAL"
    assert f1.classify_epoch(t0, [t0 + 4.0, t0 - 0.1], []) == ("NO-MARKER", [])  # markers outside the epoch
    assert f1.classify_epoch(t0, [], [(t0 - 1.0, t0 + 1.5)])[0] == "CENTRAL"    # run reaching into the centre
    assert f1.classify_epoch(t0, [], [(t0 + 3.2, t0 + 6.0)])[0] == "OUTER-ONLY"
    assert f1.classify_epoch(t0, [t0 + 0.5], [])[1] == [0.5]


def _interim(tmp_path, positions):
    """events/epochs for recordings R000.. with one IED epoch each at 8 s, marker at the given offset."""
    d = tmp_path / "interim"
    d.mkdir(parents=True)
    ev, ep = [], []
    for k, p in enumerate(positions):
        rid = f"R{k:03d}"
        ev.append([rid, 0, 0.1, "", "Waking"])
        if p is not None:
            ev.append([rid, 1, 8.0 + p, "", "!"])
        ep.append([rid, 4000, 6000, 500, 2, "frontal", "wake", int(p is not None)])
        ep.append([rid, 0, 2000, 500, 0, "non-IED", "wake", 0])
    for name, header, rows in (("events.csv", ["eeg_id", "row", "onset_s", "duration", "text"], ev),
                               ("epochs.csv", ["eeg_id", "start", "end", "sr", "label", "folder", "state",
                                               "has_ied_event"], ep)):
        with (d / name).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    return d


def test_run_decision_and_aggregate_outputs(tmp_path):
    qc_path, sm_path = _tables(tmp_path, n_e=30, n_n=5, n_f=0, n_bad=6)
    pilot = tmp_path / "pilot.csv"
    pilot.write_text("eeg_id,group,coverage\nR010,E,wake\n")
    # R006..R029 are confirmatory E (R010 is pilot): make 4 of those 23 outer-only, one with no marker
    positions = [2.0] * 35
    for k in (6, 7, 8, 9):
        positions[k] = 0.3
    positions[11] = None
    s = f1.run(_interim(tmp_path, positions), tmp_path / "out", qc_path, sm_path, pilot)
    e = s["confirmatory_E"]
    assert e["n_recordings"] == 23 and e["n_outer_only"] == 4 and e["n_no_marker"] == 1
    assert s["decision"]["value"] == round(4 / 22, 4)                     # patient-weighted over marked patients
    assert s["decision"]["result"].startswith("RETURN")
    assert s["pilot"]["n_recordings"] == 1 and s["sensitivity_only"]["n_recordings"] == 6
    text = (tmp_path / "out" / "marker_position_summary.json").read_text()
    json.loads(text)
    assert "onset" not in (tmp_path / "out" / "marker_position_by_recording.csv").read_text()

    positions = [2.0] * 35
    positions[6] = 3.5
    s = f1.run(_interim(tmp_path / "second", positions), tmp_path / "out2", qc_path, sm_path, pilot)
    assert s["decision"]["value"] <= f1.THRESHOLD and s["decision"]["result"].startswith("KEEP")
