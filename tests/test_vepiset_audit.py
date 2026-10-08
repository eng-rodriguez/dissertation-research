"""Checks the vEpiSet audit on a small synthetic archive with the real layout."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
import vepiset_audit as va  # noqa: E402
import vepiset_audit_report as va_report  # noqa: E402

FOLDERS = {0: "Non-IED", 1: "Generalized-IED", 2: "Frontal-IED", 3: "Temporal-IED",
           4: "Centro-Parietal-IED", 5: "Occipital-IED"}


def _events(rows):
    # Real files store events as an n x 3 char matrix padded with spaces.
    return np.array([[f"{a:<32}", f"{b:<32}", f"{c:<32}"] for a, b, c in rows])


def _recording(root, eeg_id, n_epochs, events, labels):
    scipy.io.savemat(root / "MAT_Files" / f"{eeg_id}.mat",
                     {"eeg_data": np.zeros((29, n_epochs * 2000)), "events": _events(events)})
    for i in range(n_epochs):
        lab = labels.get(i, 0)
        name = f"{eeg_id}_{i * 2000}_{(i + 1) * 2000}_500__{lab}.npy"
        (root / FOLDERS[lab] / name).write_bytes(b"")


def _archive(tmp_path):
    root = tmp_path / "opensource-dataset"
    for f in list(FOLDERS.values()) + ["MAT_Files"]:
        (root / f).mkdir(parents=True)
    # A: wake then sleep at 10 s (inside epoch 2), IEDs in epochs 1 and 4.
    _recording(root, "A001", 5,
               [("0.1", "0", "Waking"), ("5.0", "0", "!"), ("10.0", "0", "Sleeping"),
                ("17.0", "0", "!"), ("18.0", "0", "!end")],
               {1: 3, 4: 5})
    # B: one run spanning epochs 0-1, no state markers.
    _recording(root, "B002", 3, [("1.0", "0", "!start"), ("5.0", "0", "!end")], {0: 1, 1: 1})
    (root / "MAT_Files" / "base_info.csv").write_text("file_name,sex,age\nA001,M,30\nB002,F,40\n")
    return root


def test_epoch_state_rule():
    m = [(0.1, "wake"), (10.0, "sleep")]
    assert va.epoch_state(m, 0.0, 4.0) == "wake"
    assert va.epoch_state([(5.0, "wake")], 0.0, 4.0) == "unlabelled"
    assert va.epoch_state(m, 4.0, 8.0) == "wake"
    assert va.epoch_state(m, 8.0, 12.0) == "mixed"
    assert va.epoch_state(m, 12.0, 16.0) == "sleep"


def test_ied_intervals_flags_unmatched():
    ev = [{"onset_s": 1.0, "text": "!end"}, {"onset_s": 2.0, "text": "!"}]
    points, runs, problems = va.ied_intervals(ev)
    assert points == [2.0] and runs == [] and len(problems) == 1


def test_audit_end_to_end(tmp_path):
    root, out, interim = _archive(tmp_path), tmp_path / "out", tmp_path / "interim"
    assert va.main(["audit", "--root", str(root), "--out", str(out), "--interim", str(interim)]) == 0
    s = json.loads((out / "audit_summary.json").read_text())
    c = s["checks"]
    assert c["n_mat_files"] == 2 and c["mat_ids_equal_epoch_ids"]
    assert c["ied_epochs_without_event"] == 0 and c["non_ied_epochs_with_event"] == 0
    assert c["epochs_tile_recordings"] and c["all_recordings_29_channels"]
    assert c["base_info_ids_equal_mat_ids"]
    assert c["recordings_with_marker_problems"] == 1
    assert s["recordings_by_n_ied_classes"] == {"1": 1, "2": 1}

    recs = {r["eeg_id"]: r for r in csv.DictReader(open(out / "subject_summary.csv"))}
    a = recs["A001"]
    assert (a["ep_wake"], a["ep_mixed"], a["ep_sleep"], a["ep_unlabelled"]) == ("2", "1", "2", "0")
    assert a["ied_ep_wake"] == "1" and a["ied_ep_sleep"] == "1"
    assert a["dominant_share"] == "0.5"
    assert a["n_marker_problems"] == "1"
    assert recs["B002"]["n_ied_runs"] == "1" and recs["B002"]["ep_unlabelled"] == "3"

    classes = {r["class"]: r for r in csv.DictReader(open(out / "class_summary.csv"))}
    assert classes["generalized"]["epochs"] == "2" and classes["generalized"]["recordings"] == "1"
    assert classes["temporal"]["recordings"] == "1"

    vocab = {r["text"]: r["n_events"] for r in csv.DictReader(open(out / "event_text_vocabulary.csv"))}
    assert vocab == {"!": "2", "Waking": "1", "Sleeping": "1", "!start": "1", "!end": "2"}
    # Event-level timestamps stay in interim, never in artifacts.
    assert (interim / "events.csv").exists() and not (out / "events.csv").exists()
    problems = list(csv.DictReader(open(interim / "marker_problems.csv")))
    assert problems == [{"eeg_id": "A001", "problem": "!end at 18.0 without !start"}]
    # No event onsets in artifacts: the problem text lives only in interim.
    assert not any("without !start" in p.read_text() for p in out.glob("*"))
    # Derived outputs are written by the audit and regenerate byte-identically.
    ch = list(csv.DictReader(open(out / "channel_completeness.csv")))
    assert [r["channel_count_complete"] for r in ch] == ["True", "True"]
    report = (out / "confounding_report.md").read_bytes()
    assert b"Within-patient epoch clustering" in report
    assert va_report.main(["--audit-dir", str(out)]) == 0
    assert (out / "confounding_report.md").read_bytes() == report
    # base_info rows are not copied into artifacts.
    assert not any("sex" in p.read_text() for p in out.glob("*.csv"))
