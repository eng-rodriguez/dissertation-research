"""F-1: where IED markers fall inside the authors' 4-s epochs (06 issue 1).

Issue 1 anchors the 2-s analysis window at the most salient point within [1.0, 3.0] s of
each epoch. An IED whose annotated marker lies only in the outer 1.0 s at either end cannot
itself be the anchor (the window may still contain it). This script counts how often that
happens. It reads only the interim annotation tables written by vepiset_audit.py (marker
onsets and the authors' epoch grid). No signal, feature or model is used.

Inputs (gitignored, on the Mac):
    data/interim/vepiset/events.csv   eeg_id,row,onset_s,duration,text
    data/interim/vepiset/epochs.csv   eeg_id,start,end,sr,label,folder,state,has_ied_event
Committed inputs used for analysis sets:
    artifacts/dataset_audit/channel_qc.csv, artifacts/protocol/f3/pilot_subset.csv,
    artifacts/dataset_audit/subject_state_matrix.csv (via the F-3 group definitions)

An IED epoch is CENTRAL if any point marker ("!") lies in [1.0, 3.0) s of the epoch or any
run ("!start".."!end") overlaps that interval; OUTER-ONLY if it has markers but none central;
NO-MARKER if the authors labelled it IED but no marker falls inside it.

Decision quantity (fixed before the run, 06 §6 F-1): the patient-weighted mean, over the 36
confirmatory evaluable IED-positive patients (F-3 group E, primary set), of each patient's
share of marked IED epochs that are OUTER-ONLY.
    <= 0.10  keep the approved [1.0, 3.0] s anchor search
    >  0.10  return to Josue; the recorded alternative is to search the anchor over the whole
             epoch and clamp the window centre to [1.0, 3.0] s. Nothing changes without Josue.

Usage (from the repository root):
    python src/data/vepiset_marker_position.py

Writes artifacts/dataset_audit/{marker_position_summary.json, marker_position_bins.csv,
marker_position_by_recording.csv}: counts only, no onsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "design"))
import vepiset_audit as va  # noqa: E402
import f3_pilot_and_folds as f3  # noqa: E402

EPOCH_S = va.EPOCH_SAMPLES / va.SFREQ  # 4.0
OUTER_S = 1.0
THRESHOLD = 0.10
BIN_S = 0.25
N_BINS = int(EPOCH_S / BIN_S)


def classify_epoch(t0: float, points: list[float], runs: list[tuple[float, float]]) -> tuple[str, list[float]]:
    """Return (CENTRAL | OUTER-ONLY | NO-MARKER, point positions within the epoch)."""
    t1 = t0 + EPOCH_S
    pos = [t - t0 for t in points if t0 <= t < t1]
    run_overlap = [(max(a, t0) - t0, min(b, t1) - t0) for a, b in runs if a < t1 and b >= t0]
    lo, hi = OUTER_S, EPOCH_S - OUTER_S
    central = any(lo <= p < hi for p in pos) or any(a < hi and b >= lo for a, b in run_overlap)
    if central:
        return "CENTRAL", pos
    if pos or run_overlap:
        return "OUTER-ONLY", pos
    return "NO-MARKER", pos


def analysis_sets(qc_csv: Path, state_matrix: Path, pilot_csv: Path) -> tuple[dict[str, str], dict[str, str]]:
    recs = f3.load_recordings(qc_csv, state_matrix)
    pilot = {r["eeg_id"] for r in csv.DictReader(pilot_csv.open())}
    aset = {i: ("pilot" if i in pilot else "primary" if r["qc_primary"] else "sensitivity_only")
            for i, r in recs.items()}
    return aset, {i: r["group"] for i, r in recs.items()}


def run(interim: Path, out: Path, qc_csv: Path, state_matrix: Path, pilot_csv: Path) -> dict:
    events = defaultdict(list)
    for e in csv.DictReader((interim / "events.csv").open()):
        events[e["eeg_id"]].append({"onset_s": float(e["onset_s"]), "text": e["text"]})
    epochs = [e for e in csv.DictReader((interim / "epochs.csv").open()) if int(e["label"]) in va.IED_LABELS]
    aset, group = analysis_sets(qc_csv, state_matrix, pilot_csv)

    per_rec = defaultdict(Counter)
    bins = Counter()
    n_runs_epochs = 0
    for rid in sorted({e["eeg_id"] for e in epochs}):
        points, runs, _ = va.ied_intervals(events.get(rid, []))
        for e in (x for x in epochs if x["eeg_id"] == rid):
            cat, pos = classify_epoch(int(e["start"]) / va.SFREQ, points, runs)
            per_rec[rid][cat] += 1
            per_rec[rid][f"{cat}|{e['state']}"] += 1
            t0 = int(e["start"]) / va.SFREQ
            if any(a < t0 + EPOCH_S and b >= t0 for a, b in runs):
                n_runs_epochs += 1
            for p in pos:
                bins[(aset.get(rid, "unknown"), e["state"], min(int(p / BIN_S), N_BINS - 1))] += 1

    rows = []
    for rid, c in sorted(per_rec.items()):
        marked = c["CENTRAL"] + c["OUTER-ONLY"]
        rows.append({"eeg_id": rid, "analysis_set": aset.get(rid, "unknown"), "group": group.get(rid, ""),
                     "n_ied_epochs": marked + c["NO-MARKER"], "n_central": c["CENTRAL"],
                     "n_outer_only": c["OUTER-ONLY"], "n_no_marker": c["NO-MARKER"],
                     "outer_only_share": round(c["OUTER-ONLY"] / marked, 4) if marked else ""})

    def block(sel):
        rs = [r for r in rows if sel(r)]
        tot = Counter()
        for r in rs:
            tot.update({k: r[k] for k in ("n_ied_epochs", "n_central", "n_outer_only", "n_no_marker")})
        marked = tot["n_central"] + tot["n_outer_only"]
        shares = [r["outer_only_share"] for r in rs if r["outer_only_share"] != ""]
        return {"n_recordings": len(rs), **tot,
                "outer_only_share_epoch_weighted": round(tot["n_outer_only"] / marked, 4) if marked else None,
                "outer_only_share_patient_weighted": round(sum(shares) / len(shares), 4) if shares else None}

    decision_block = block(lambda r: r["analysis_set"] == "primary" and r["group"] == "E")
    q = decision_block["outer_only_share_patient_weighted"]
    by_state = {}
    for st in ("wake", "sleep", "mixed", "unlabelled"):
        cen = sum(per_rec[r][f"CENTRAL|{st}"] for r in per_rec)
        out_ = sum(per_rec[r][f"OUTER-ONLY|{st}"] for r in per_rec)
        by_state[st] = {"n_central": cen, "n_outer_only": out_,
                        "outer_only_share": round(out_ / (cen + out_), 4) if cen + out_ else None}
    summary = {
        "definition": f"OUTER-ONLY = IED epoch whose markers all lie in the outer {OUTER_S} s at either end",
        "decision": {
            "quantity": "patient-weighted OUTER-ONLY share, confirmatory evaluable IED-positive patients (primary set, group E)",
            "value": q, "threshold": THRESHOLD,
            "result": None if q is None else ("KEEP [1.0, 3.0] s anchor search" if q <= THRESHOLD
                                              else "RETURN TO JOSUE (recorded option: whole-epoch anchor, clamped window centre)"),
            "rule_fixed_before_run": True,
        },
        "confirmatory_E": decision_block,
        "all_ied_epochs": block(lambda r: True),
        "primary_set": block(lambda r: r["analysis_set"] == "primary"),
        "pilot": block(lambda r: r["analysis_set"] == "pilot"),
        "sensitivity_only": block(lambda r: r["analysis_set"] == "sensitivity_only"),
        "by_state_all": by_state,
        "ied_epochs_overlapping_a_run": n_runs_epochs,
        "expected_under_uniform_positions": 2 * OUTER_S / EPOCH_S,
        "provenance": va.provenance() | {"script": "src/data/vepiset_marker_position.py"},
    }
    out.mkdir(parents=True, exist_ok=True)
    va.write_csv(out / "marker_position_by_recording.csv", rows, list(rows[0]) if rows else ["eeg_id"])
    bin_rows = [{"analysis_set": a, "state": s, "bin_start_s": b * BIN_S, "bin_end_s": (b + 1) * BIN_S, "n_point_markers": n}
                for (a, s, b), n in sorted(bins.items())]
    va.write_csv(out / "marker_position_bins.csv", bin_rows,
                 ["analysis_set", "state", "bin_start_s", "bin_end_s", "n_point_markers"])
    (out / "marker_position_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interim", default="data/interim/vepiset")
    ap.add_argument("--out", default="artifacts/dataset_audit")
    ap.add_argument("--qc", default="artifacts/dataset_audit/channel_qc.csv")
    ap.add_argument("--state-matrix", default="artifacts/dataset_audit/subject_state_matrix.csv")
    ap.add_argument("--pilot", default="artifacts/protocol/f3/pilot_subset.csv")
    a = ap.parse_args()
    s = run(Path(a.interim), Path(a.out), Path(a.qc), Path(a.state_matrix), Path(a.pilot))
    d = s["decision"]
    print(f"IED epochs: {s['all_ied_epochs']['n_ied_epochs']} "
          f"(central {s['all_ied_epochs']['n_central']}, outer-only {s['all_ied_epochs']['n_outer_only']}, "
          f"no marker {s['all_ied_epochs']['n_no_marker']})")
    print(f"Confirmatory E patients: {s['confirmatory_E']['n_recordings']}; "
          f"outer-only share patient-weighted {d['value']}, epoch-weighted {s['confirmatory_E']['outer_only_share_epoch_weighted']}")
    print(f"Decision (threshold {d['threshold']}): {d['result']}")


if __name__ == "__main__":
    main()
