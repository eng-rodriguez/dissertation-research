"""R3: recount evaluable patients after all 06 §4.2 and epoch-QC exclusions (labels and counts only).

06 issue 2 (R3/Q4, adopted 2026-10-10): before any model output exists, evaluability (>= 5 IED and
>= 5 non-IED epochs in a common wake/sleep state) is recomputed on the final primary analysis set.
The frozen fold manifest is not redrawn; patients who lose evaluability stay in their folds as
training-only recordings. No feature, FC or model is computed. The signal is read only to recompute
the label-blind F-4 epoch flags (flat / clipped), with the F-4 code and thresholds unchanged.

Epoch exclusions, applied in this order (first matching reason is recorded):
  short         epoch shorter than 4 s (the 65 short final epochs)
  edge          first or last full 4-s epoch of the recording (wPLI specification step 4)
  epoch_qc      F-4 epoch flag (flat or clipped segment)
  ied_no_marker IED-labelled epoch with no reconstructable marker
  nonied_marker non-IED epoch containing a marker
  adjacent      non-IED epoch whose grid neighbour (start +/- 2,000 samples) is IED-labelled
  state         state not a single wake or sleep label

Rule (Q4, Josue 2026-10-10): any change from 36 confirmatory evaluable patients is recorded in 11;
an RDR is required if the count falls below 33 (or if a change arises from a substantive protocol
alteration rather than these pre-specified exclusions).

Usage (from the repository root, on the Mac):
    python src/data/vepiset_evaluable_recount.py --root data/raw/vepiset/extracted/opensource-dataset
Writes artifacts/dataset_audit/{evaluable_recount.csv, evaluable_recount_summary.json}: counts only.
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

MIN_EPOCHS = 5
E_BEFORE = 36
RDR_BELOW = 33
REASONS = ("short", "edge", "epoch_qc", "ied_no_marker", "nonied_marker", "adjacent", "state")
FIELDS = (["eeg_id", "analysis_set", "group_before", "group_after", "evaluable_states_after"]
          + [f"n_{c}_{s}" for s in ("wake", "sleep") for c in ("ied", "nonied")]
          + [f"excluded_{r}" for r in REASONS] + ["excluded_edge_ied"])


def signal_epoch_flags(root: Path):
    """Return f(eeg_id, epochs) -> list[bool] using the unchanged F-4 code and thresholds."""
    import scipy.io
    import vepiset_channel_qc as qc

    def flags(eeg_id: str, epochs: list[dict]) -> list[bool]:
        x = scipy.io.loadmat(root / "MAT_Files" / f"{eeg_id}.mat", variable_names=["eeg_data"])["eeg_data"]
        return qc.epoch_flags(x, epochs, qc.channel_qc(x)["_clip_level"])
    return flags


def classify(epochs: list[dict], qc_flags: list[bool]) -> list[str | None]:
    """Exclusion reason per epoch (None = kept). epochs sorted by start, same order as qc_flags."""
    full = [e["start"] for e in epochs if e["end"] - e["start"] == va.EPOCH_SAMPLES]
    edges = {min(full), max(full)} if full else set()
    ied_starts = {e["start"] for e in epochs if e["label"] in va.IED_LABELS}
    out = []
    for e, f in zip(epochs, qc_flags):
        is_ied = e["label"] in va.IED_LABELS
        if e["end"] - e["start"] < va.EPOCH_SAMPLES:
            r = "short"
        elif e["start"] in edges:
            r = "edge"
        elif f:
            r = "epoch_qc"
        elif is_ied and not e["has_ied_event"]:
            r = "ied_no_marker"
        elif not is_ied and e["has_ied_event"]:
            r = "nonied_marker"
        elif not is_ied and ({e["start"] - va.EPOCH_SAMPLES, e["start"] + va.EPOCH_SAMPLES} & ied_starts):
            r = "adjacent"
        elif e["state"] not in ("wake", "sleep"):
            r = "state"
        else:
            r = None
        out.append(r)
    return out


def run(interim: Path, out: Path, qc_csv: Path, state_matrix: Path, pilot_csv: Path, epoch_flag_fn) -> dict:
    recs = f3.load_recordings(qc_csv, state_matrix)
    pilot = {r["eeg_id"] for r in csv.DictReader(pilot_csv.open())}
    by = defaultdict(list)
    for e in csv.DictReader((interim / "epochs.csv").open()):
        by[e["eeg_id"]].append({"start": int(e["start"]), "end": int(e["end"]), "label": int(e["label"]),
                                "state": e["state"], "has_ied_event": int(e["has_ied_event"])})
    rows, totals = [], Counter()
    for rid in sorted(recs):
        eps = sorted(by.get(rid, []), key=lambda e: e["start"])
        reasons = classify(eps, epoch_flag_fn(rid, eps) if eps else [])
        n = Counter()
        for e, r in zip(eps, reasons):
            cls = "ied" if e["label"] in va.IED_LABELS else "nonied"
            if r is None:
                n[f"n_{cls}_{e['state']}"] += 1
            else:
                n[f"excluded_{r}"] += 1
                totals[f"{r}|{cls}"] += 1
                if r == "edge" and cls == "ied":
                    n["excluded_edge_ied"] += 1
        ev = [s for s in ("wake", "sleep") if n[f"n_ied_{s}"] >= MIN_EPOCHS and n[f"n_nonied_{s}"] >= MIN_EPOCHS]
        before = recs[rid]["group"]
        after = "E" if ev else ("F" if before == "F" else "N")
        aset = "pilot" if rid in pilot else ("primary" if recs[rid]["qc_primary"] else "sensitivity_only")
        rows.append({"eeg_id": rid, "analysis_set": aset, "group_before": before, "group_after": after,
                     "evaluable_states_after": "+".join(ev), **{k: n[k] for k in FIELDS[5:]}})

    conf = [r for r in rows if r["analysis_set"] == "primary"]
    e_after = [r for r in conf if r["group_after"] == "E"]
    lost = sorted(r["eeg_id"] for r in conf if r["group_before"] == "E" and r["group_after"] != "E")
    gained = sorted(r["eeg_id"] for r in conf if r["group_before"] != "E" and r["group_after"] == "E")
    n_e = len(e_after)
    summary = {
        "confirmatory_evaluable_before": sum(r["group_before"] == "E" for r in conf),
        "confirmatory_evaluable_after": n_e,
        "evaluable_in_wake_after": sum("wake" in r["evaluable_states_after"] for r in e_after),
        "evaluable_in_sleep_after": sum("sleep" in r["evaluable_states_after"] for r in e_after),
        "lost_evaluability": lost, "gained_evaluability": gained,
        "rule": {"reference": E_BEFORE, "rdr_if_below": RDR_BELOW,
                 "result": ("NO CHANGE" if n_e == E_BEFORE and not gained else
                            "RDR REQUIRED (below 33)" if n_e < RDR_BELOW else "RECORD CHANGE IN 11")},
        "pilot_evaluable_after": sum(r["group_after"] == "E" for r in rows if r["analysis_set"] == "pilot"),
        "excluded_epochs_by_reason_and_class": dict(sorted(totals.items())),
        "ied_epochs_removed_by_edge_exclusion": sum(r["excluded_edge_ied"] for r in rows),
        "provenance": va.provenance() | {"script": "src/data/vepiset_evaluable_recount.py"},
    }
    out.mkdir(parents=True, exist_ok=True)
    va.write_csv(out / "evaluable_recount.csv", rows, FIELDS)
    (out / "evaluable_recount_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="extracted opensource-dataset folder (for epoch QC flags)")
    ap.add_argument("--interim", default="data/interim/vepiset")
    ap.add_argument("--out", default="artifacts/dataset_audit")
    ap.add_argument("--qc", default="artifacts/dataset_audit/channel_qc.csv")
    ap.add_argument("--state-matrix", default="artifacts/dataset_audit/subject_state_matrix.csv")
    ap.add_argument("--pilot", default="artifacts/protocol/f3/pilot_subset.csv")
    a = ap.parse_args()
    s = run(Path(a.interim), Path(a.out), Path(a.qc), Path(a.state_matrix), Path(a.pilot),
            signal_epoch_flags(Path(a.root)))
    print(f"Confirmatory evaluable patients: before {s['confirmatory_evaluable_before']}, "
          f"after {s['confirmatory_evaluable_after']} (wake {s['evaluable_in_wake_after']}, "
          f"sleep {s['evaluable_in_sleep_after']}); lost {s['lost_evaluability']}")
    print(f"IED epochs removed by first/last-epoch exclusion: {s['ied_epochs_removed_by_edge_exclusion']}")
    print(f"Rule: {s['rule']['result']}")


if __name__ == "__main__":
    main()
