"""Checks the F-3 pilot draw and fold manifest on synthetic aggregate tables."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "design"))
import f3_pilot_and_folds as f3

SM_FIELDS = ["eeg_id", "all_wake", "all_sleep", "all_mixed", "all_unlabelled",
             "ied_wake", "ied_sleep", "ied_mixed", "ied_unlabelled"]


def _tables(tmp_path, n_e=30, n_n=5, n_f=25, n_bad=6):
    """Synthetic recordings: E (evaluable IED), N (IED, not evaluable), F (IED-free); first n_bad fail QC."""
    sm, qc = [], []
    k = 0
    for g, n in (("E", n_e), ("N", n_n), ("F", n_f)):
        for j in range(n):
            rid = f"R{k:03d}"
            wake, sleep = [(50, 0), (0, 60), (40, 40)][j % 3]
            ied = {"E": 8, "N": 2, "F": 0}[g]
            ied_w = ied if wake else 0
            ied_s = ied if sleep else 0
            sm.append([rid, wake, sleep, 0, 3, ied_w, ied_s, 0, 0])
            qc.append({"eeg_id": rid, "primary_eligible": str(k >= n_bad),
                       "pre_qc_eligible_ied": str(g == "E"), "ied_positive": str(g != "F")})
            k += 1
    sm_path, qc_path = tmp_path / "subject_state_matrix.csv", tmp_path / "channel_qc.csv"
    with sm_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SM_FIELDS)
        w.writerows(sm)
    with qc_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qc[0]))
        w.writeheader()
        w.writerows(qc)
    return qc_path, sm_path


def test_largest_remainder_matches_expected_allocation():
    assert f3.largest_remainder({"sleep": 25, "wake": 7, "wake+sleep": 9}, 5) == {"sleep": 3, "wake": 1, "wake+sleep": 1}
    assert f3.largest_remainder({"sleep": 4, "wake": 16, "wake+sleep": 10}, 3) == {"sleep": 0, "wake": 2, "wake+sleep": 1}
    with pytest.raises(ValueError):
        f3.largest_remainder({"a": 1}, 2)


def test_groups_and_coverage(tmp_path):
    recs = f3.load_recordings(*_tables(tmp_path))
    assert Counter(r["group"] for r in recs.values()) == {"E": 30, "N": 5, "F": 25}
    assert recs["R002"]["coverage"] == "wake+sleep" and recs["R002"]["evaluable_states"] == ["wake", "sleep"]
    assert recs["R030"]["group"] == "N" and recs["R030"]["coverage"] == "wake"


def test_group_mismatch_with_qc_table_is_an_error(tmp_path):
    qc_path, sm_path = _tables(tmp_path)
    rows = list(csv.DictReader(qc_path.open()))
    rows[10]["pre_qc_eligible_ied"] = "False"
    with qc_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError):
        f3.load_recordings(qc_path, sm_path)


def test_pilot_is_qc_eligible_stratified_and_deterministic(tmp_path):
    recs = f3.load_recordings(*_tables(tmp_path))
    pilot, alloc = f3.draw_pilot(recs)
    assert Counter(r["group"] for r in pilot) == {"E": 5, "F": 3}
    assert all(recs[r["eeg_id"]]["qc_primary"] for r in pilot)
    for g in ("E", "F"):
        assert Counter(r["coverage"] for r in pilot if r["group"] == g) == {k: v for k, v in alloc[g]["drawn"].items() if v}
    assert f3.draw_pilot(recs)[0] == pilot
    assert f3.draw_pilot(recs, seed=1)[0] != pilot


def test_manifest_excludes_pilot_and_partitions_each_repeat(tmp_path):
    recs = f3.load_recordings(*_tables(tmp_path))
    pilot_ids = {r["eeg_id"] for r in f3.draw_pilot(recs)[0]}
    rows = f3.build_manifest(recs, pilot_ids)
    assert not pilot_ids & {r["eeg_id"] for r in rows}
    for rep in range(1, 11):
        rr = [r for r in rows if r["repeat"] == rep]
        ids = [r["eeg_id"] for r in rr]
        assert len(ids) == len(set(ids)) == len(recs) - len(pilot_ids)  # each recording exactly once
        prim = [r for r in rr if r["analysis_set"] == "primary"]
        sizes = Counter(r["outer_fold"] for r in prim)
        assert set(sizes) == set(range(5)) and max(sizes.values()) - min(sizes.values()) <= 1
        for stratum in {(r["group"], r["coverage"]) for r in prim}:
            c = Counter(r["outer_fold"] for r in prim if (r["group"], r["coverage"]) == stratum)
            assert max(c.get(k, 0) for k in range(5)) - min(c.get(k, 0) for k in range(5)) <= 1
        assert {r["eeg_id"] for r in rr if r["analysis_set"] == "sensitivity_only"} == \
            {i for i, r in recs.items() if not r["qc_primary"]}
    assert len({tuple(r["outer_fold"] for r in rows if r["repeat"] == rep) for rep in range(1, 11)}) == 10


def test_primary_folds_do_not_depend_on_qc_ineligible_recordings(tmp_path):
    recs = f3.load_recordings(*_tables(tmp_path))
    pilot_ids = {r["eeg_id"] for r in f3.draw_pilot(recs)[0]}
    full = [r for r in f3.build_manifest(recs, pilot_ids) if r["analysis_set"] == "primary"]
    kept = {i: r for i, r in recs.items() if r["qc_primary"]}
    assert [r for r in f3.build_manifest(kept, pilot_ids)] == full


def test_run_writes_hashes_refuses_redraw_and_verifies(tmp_path):
    qc_path, sm_path = _tables(tmp_path)
    out = tmp_path / "f3"
    s = f3.run(qc_path, sm_path, out)
    assert s["pilot"]["n"] == 8 and s["primary_set"]["n"] == 60 - 6 - 8
    sums = (out / "SHA256SUMS").read_text()
    for name in f3.OUTPUTS:
        assert f"{f3.sha256((out / name).read_bytes())}  {name}" in sums
    json.loads((out / "f3_summary.json").read_text())
    with pytest.raises(FileExistsError):
        f3.run(qc_path, sm_path, out)
    assert f3.run(qc_path, sm_path, out, verify=True)["verify_ok"]
    (out / "fold_manifest.csv").write_bytes((out / "fold_manifest.csv").read_bytes().replace(b",0\n", b",1\n", 1))
    assert not f3.run(qc_path, sm_path, out, verify=True)["verify_ok"]
