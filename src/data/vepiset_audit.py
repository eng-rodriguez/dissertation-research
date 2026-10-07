"""vEpiSet structural audit: patient x spatial-class x state characterization.

Reads the unzipped Figshare archive and writes metadata only. Raw EEG values are
never loaded except to read array shapes. Event- and epoch-level tables (with
per-recording timestamps) go to data/interim/ (gitignored); per-recording and
aggregate tables go to artifacts/dataset_audit/.

Source: Lin et al. (2025), Sci Data 12:229, doi:10.1038/s41597-025-04572-1.
Archive: doi:10.6084/m9.figshare.28069568 (v2), vepiset-dataset.zip.

Layout of the archive (opensource-dataset/):
    MAT_Files/<eeg_id>.mat      eeg_data (29 x n_samples), events (n x 3 char:
                                onset seconds, duration, text)
    MAT_Files/base_info.csv     file_name, sex, age (not copied anywhere)
    <Class>-IED/, Non-IED/      authors' 4-s epochs, <eeg_id>_<start>_<end>_500__<label>.npy

Spatial class exists only in the authors' epoch labels; MAT annotations carry
"!" (IED), optional "!start"/"!end", and "Waking"/"Sleeping" state markers.

Usage (from the repository root):
    python src/data/vepiset_audit.py checksum --zip data/raw/vepiset/raw/vepiset-dataset.zip
    python src/data/vepiset_audit.py audit --root data/raw/vepiset/extracted/opensource-dataset
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FIGSHARE_V2_MD5 = "3b46ecbce87f07c438ea27be7c44b99d"
FIGSHARE_V2_BYTES = 16_917_002_806

SFREQ = 500
EPOCH_SAMPLES = 2000

# Figures reported by the dataset paper, used only as cross-checks.
PAPER = {
    "n_subjects": 84,
    "n_ied_subjects": 52,
    "n_epochs": 25_449,
    "n_ied_epochs": 2_516,
    "n_non_ied_epochs": 22_933,
}

# Label codes as documented in the paper; folder names are checked against them.
LABELS = {
    0: "non-IED",
    1: "generalized",
    2: "frontal",
    3: "temporal",
    4: "centro-parietal",
    5: "occipital",
}
IED_LABELS = [1, 2, 3, 4, 5]

NPY_NAME = re.compile(
    r"^(?P<eeg_id>[^_]+)_(?P<start>\d+)_(?P<end>\d+)_(?P<sr>\d+)__(?P<label>\d+)\.npy$"
)
STATES = {"waking": "wake", "sleeping": "sleep"}


def md5sum(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance() -> dict:
    import numpy
    import scipy

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def cmd_checksum(args: argparse.Namespace) -> int:
    p = Path(args.zip)
    size, digest = p.stat().st_size, md5sum(p)
    res = {
        "zip": str(p), "bytes": size, "md5": digest,
        "size_matches_figshare_v2": size == FIGSHARE_V2_BYTES,
        "md5_matches_figshare_v2": digest == FIGSHARE_V2_MD5,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "archive_checksum.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0 if res["md5_matches_figshare_v2"] else 1


# --------------------------------------------------------------------------- #
# Reading


def read_epochs(root: Path) -> tuple[list[dict], list[str]]:
    """Parse every authors' epoch file name. Returns (epochs, unparsed paths)."""
    epochs, unparsed = [], []
    for p in sorted(root.glob("*/*.npy")):
        m = NPY_NAME.match(p.name)
        if not m:
            unparsed.append(str(p.relative_to(root)))
            continue
        epochs.append({
            "eeg_id": m["eeg_id"], "start": int(m["start"]), "end": int(m["end"]),
            "sr": int(m["sr"]), "label": int(m["label"]), "folder": p.parent.name,
        })
    return epochs, unparsed


def read_mat_events(path: Path) -> tuple[dict, list[dict]]:
    """Return ({variable: shape}, events) without loading the signal."""
    import scipy.io

    shapes = {n: list(s) for n, s, _ in scipy.io.whosmat(path)}
    ev = scipy.io.loadmat(path, variable_names=["events"])["events"]
    events = []
    for i, row in enumerate(ev):
        cells = [str(c).strip() for c in row]
        events.append({"row": i, "onset_s": float(cells[0]),
                       "duration": cells[1] if len(cells) > 2 else "",
                       "text": cells[-1]})
    return shapes, events


# --------------------------------------------------------------------------- #
# Derivations


def ied_intervals(events: list[dict]) -> tuple[list[float], list[tuple[float, float]], list[str]]:
    """Point IEDs ("!") and runs ("!start".."!end"). Unmatched markers are reported."""
    points, runs, problems, open_at = [], [], [], None
    for e in sorted(events, key=lambda e: e["onset_s"]):
        t = e["text"].replace(" ", "").lower()
        if not t.startswith("!"):
            continue
        if "start" in t:
            if open_at is not None:
                problems.append(f"!start at {e['onset_s']} while run open since {open_at}")
            open_at = e["onset_s"]
        elif "end" in t:
            if open_at is None:
                problems.append(f"!end at {e['onset_s']} without !start")
            else:
                runs.append((open_at, e["onset_s"]))
                open_at = None
        elif t == "!":
            points.append(e["onset_s"])
        else:
            problems.append(f"unrecognised IED marker {e['text']!r} at {e['onset_s']}")
    if open_at is not None:
        problems.append(f"!start at {open_at} never closed")
    return points, runs, problems


def state_markers(events: list[dict]) -> list[tuple[float, str]]:
    return sorted((e["onset_s"], STATES[e["text"].lower()])
                  for e in events if e["text"].lower() in STATES)


def epoch_state(markers: list[tuple[float, str]], t0: float, t1: float) -> str:
    """State over [t0, t1): the state in force at t0 plus any marker inside the
    epoch. One state -> that state; two -> "mixed". Time before the first marker
    is "unlabelled", but an epoch whose first marker falls inside it takes that
    marker's state (recordings start with a marker at ~0.1 s)."""
    current = "unlabelled"
    for t, s in markers:
        if t <= t0:
            current = s
        else:
            break
    covered = {current} | {s for t, s in markers if t0 < t < t1}
    if len(covered) > 1:
        covered.discard("unlabelled")
    return covered.pop() if len(covered) == 1 else "mixed"


def inverse_simpson(counts: list[int]) -> float:
    n = sum(counts)
    return 1.0 / sum((c / n) ** 2 for c in counts if c) if n else 0.0


# --------------------------------------------------------------------------- #
# Audit


def cmd_audit(args: argparse.Namespace) -> int:
    root, out, interim = Path(args.root), Path(args.out), Path(args.interim)
    checks: dict = {}
    notes: list[str] = []

    # Files on disk, aggregated by folder.
    by_folder = collections.defaultdict(lambda: {"n_files": 0, "bytes": 0})
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            k = (p.parent.name, p.suffix.lower())
            by_folder[k]["n_files"] += 1
            by_folder[k]["bytes"] += p.stat().st_size
    write_csv(out / "file_inventory.csv",
              [{"folder": f, "ext": e, **v} for (f, e), v in sorted(by_folder.items())],
              ["folder", "ext", "n_files", "bytes"])

    epochs, unparsed = read_epochs(root)
    checks["npy_unparsed_names"] = len(unparsed)
    folder_label = collections.Counter((e["folder"], e["label"]) for e in epochs)
    checks["each_folder_has_one_label"] = all(
        sum(1 for (f, _) in folder_label if f == fol) == 1 for fol, _ in folder_label)
    checks["all_sr_500"] = all(e["sr"] == SFREQ for e in epochs)
    checks["all_starts_on_4s_grid"] = all(e["start"] % EPOCH_SAMPLES == 0 for e in epochs)
    short = [e for e in epochs if e["end"] - e["start"] != EPOCH_SAMPLES]
    checks["n_epochs_shorter_than_4s"] = len(short)
    checks["short_epochs_are_last_in_recording"] = all(
        e["end"] == max(x["end"] for x in epochs if x["eeg_id"] == e["eeg_id"]) for e in short)
    dup = collections.Counter((e["eeg_id"], e["start"]) for e in epochs)
    checks["duplicate_epochs"] = sum(1 for v in dup.values() if v > 1)

    by_id: dict[str, list[dict]] = collections.defaultdict(list)
    for e in epochs:
        by_id[e["eeg_id"]].append(e)

    mat_dir = root / "MAT_Files"
    mat_ids = sorted(p.stem for p in mat_dir.glob("*.mat"))
    checks["n_mat_files"] = len(mat_ids)
    checks["mat_ids_equal_epoch_ids"] = set(mat_ids) == set(by_id)

    vocab = collections.Counter()
    vocab_ids = collections.defaultdict(set)
    event_rows, epoch_rows, rec_rows, problem_rows = [], [], [], []
    mismatch_total = collections.Counter()

    for k, eeg_id in enumerate(mat_ids, 1):
        shapes, events = read_mat_events(mat_dir / f"{eeg_id}.mat")
        for e in events:
            vocab[e["text"]] += 1
            vocab_ids[e["text"]].add(eeg_id)
            event_rows.append({"eeg_id": eeg_id, **e})
        points, runs, problems = ied_intervals(events)
        problem_rows += [{"eeg_id": eeg_id, "problem": p} for p in problems]
        markers = state_markers(events)
        n_ch, n_samp = (shapes.get("eeg_data") or [None, None])[:2]

        eps = sorted(by_id.get(eeg_id, []), key=lambda e: e["start"])
        # Epochs containing at least one annotated IED (point or overlapping run).
        idx_with_event = set()
        for t in points:
            idx_with_event.add(int(t * SFREQ) // EPOCH_SAMPLES)
        for a, b in runs:
            idx_with_event.update(range(int(a * SFREQ) // EPOCH_SAMPLES,
                                        int(b * SFREQ) // EPOCH_SAMPLES + 1))
        mm = collections.Counter()
        cls = collections.Counter()
        st = collections.Counter()
        for e in eps:
            s = epoch_state(markers, e["start"] / SFREQ, e["end"] / SFREQ)
            has_ev = (e["start"] // EPOCH_SAMPLES) in idx_with_event
            if e["label"] > 0 and not has_ev:
                mm["ied_epoch_without_event"] += 1
            if e["label"] == 0 and has_ev:
                mm["non_ied_epoch_with_event"] += 1
            cls[e["label"]] += 1
            st[(e["label"], s)] += 1
            epoch_rows.append({**e, "state": s, "has_ied_event": int(has_ev)})
        mismatch_total.update(mm)

        ied_counts = [cls.get(c, 0) for c in IED_LABELS]
        n_ied = sum(ied_counts)
        dom = max(IED_LABELS, key=lambda c: cls.get(c, 0)) if n_ied else 0
        rec = {
            "eeg_id": eeg_id,
            "n_channels": n_ch,
            "n_samples": n_samp,
            "duration_s": n_samp / SFREQ if n_samp else None,
            "max_epoch_end_equals_n_samples": bool(eps) and eps[-1]["end"] == n_samp,
            "n_events": len(events),
            "n_ied_point_events": len(points),
            "n_ied_runs": len(runs),
            "n_marker_problems": len(problems),
            "n_state_markers": len(markers),
            "first_state": markers[0][1] if markers else "",
            "n_epochs": len(eps),
            "n_ied_epochs": n_ied,
            **{f"ep_{LABELS[c]}": cls.get(c, 0) for c in [0] + IED_LABELS},
            "n_ied_classes": sum(1 for c in ied_counts if c),
            "dominant_class": LABELS[dom] if n_ied else "",
            "dominant_share": round(max(ied_counts) / n_ied, 3) if n_ied else None,
            **{f"ep_{s}": sum(v for (lab, ss), v in st.items() if ss == s)
               for s in ["wake", "sleep", "mixed", "unlabelled"]},
            **{f"ied_ep_{s}": sum(v for (lab, ss), v in st.items() if ss == s and lab > 0)
               for s in ["wake", "sleep", "mixed", "unlabelled"]},
            "ied_epoch_without_event": mm["ied_epoch_without_event"],
            "non_ied_epoch_with_event": mm["non_ied_epoch_with_event"],
        }
        rec_rows.append(rec)
        print(f"[{k}/{len(mat_ids)}] {eeg_id}: {len(events)} events, "
              f"{n_ied} IED epochs, states {dict(collections.Counter(s for _, s in markers))}",
              flush=True)

    # ---- per-recording and matrices (artifacts) ----
    rec_fields = list(rec_rows[0]) if rec_rows else []
    write_csv(out / "subject_summary.csv", rec_rows, rec_fields)

    write_csv(out / "subject_class_matrix.csv",
              [{"eeg_id": r["eeg_id"], **{LABELS[c]: r[f"ep_{LABELS[c]}"] for c in [0] + IED_LABELS},
                "n_ied_classes": r["n_ied_classes"]} for r in rec_rows],
              ["eeg_id"] + [LABELS[c] for c in [0] + IED_LABELS] + ["n_ied_classes"])

    states = ["wake", "sleep", "mixed", "unlabelled"]
    write_csv(out / "subject_state_matrix.csv",
              [{"eeg_id": r["eeg_id"],
                **{f"all_{s}": r[f"ep_{s}"] for s in states},
                **{f"ied_{s}": r[f"ied_ep_{s}"] for s in states}} for r in rec_rows],
              ["eeg_id"] + [f"all_{s}" for s in states] + [f"ied_{s}" for s in states])

    cs = collections.Counter((e["label"], e["state"]) for e in epoch_rows)
    cs_ids = collections.defaultdict(set)
    for e in epoch_rows:
        cs_ids[(e["label"], e["state"])].add(e["eeg_id"])
    write_csv(out / "class_state_matrix.csv",
              [{"class": LABELS[c], **{f"epochs_{s}": cs.get((c, s), 0) for s in states},
                **{f"recordings_{s}": len(cs_ids[(c, s)]) for s in states}}
               for c in [0] + IED_LABELS],
              ["class"] + [f"epochs_{s}" for s in states] + [f"recordings_{s}" for s in states])

    class_rows = []
    for c in [0] + IED_LABELS:
        per = sorted((r[f"ep_{LABELS[c]}"] for r in rec_rows if r[f"ep_{LABELS[c]}"]), reverse=True)
        tot = sum(per)
        class_rows.append({
            "code": c, "class": LABELS[c], "epochs": tot,
            "recordings": len(per),
            "recordings_ge10_epochs": sum(1 for v in per if v >= 10),
            "top_recording_share": round(per[0] / tot, 3) if tot else None,
            "top3_recordings_share": round(sum(per[:3]) / tot, 3) if tot else None,
            "effective_n_recordings": round(inverse_simpson(per), 2),
            "recordings_where_dominant": sum(1 for r in rec_rows
                                             if r["dominant_class"] == LABELS[c]),
            "per_recording_epochs_desc": " ".join(map(str, per)),
        })
    write_csv(out / "class_summary.csv", class_rows, list(class_rows[0]))

    write_csv(out / "event_text_vocabulary.csv",
              [{"text": t, "n_events": n, "n_recordings": len(vocab_ids[t])}
               for t, n in vocab.most_common()],
              ["text", "n_events", "n_recordings"])

    # ---- event/epoch-level tables (interim, gitignored) ----
    write_csv(interim / "events.csv", event_rows,
              ["eeg_id", "row", "onset_s", "duration", "text"])
    # Marker problems quote event onsets, so the detail stays in interim too.
    write_csv(interim / "marker_problems.csv", problem_rows, ["eeg_id", "problem"])
    write_csv(interim / "epochs.csv", epoch_rows,
              ["eeg_id", "start", "end", "sr", "label", "folder", "state", "has_ied_event"])

    # ---- summary ----
    lab_tot = collections.Counter(e["label"] for e in epoch_rows)
    n_ied_recs = sum(1 for r in rec_rows if r["n_ied_epochs"])
    checks.update({
        "n_epochs_equals_paper": len(epochs) == PAPER["n_epochs"],
        "n_ied_epochs_equals_paper": sum(v for c, v in lab_tot.items() if c) == PAPER["n_ied_epochs"],
        "n_non_ied_epochs_equals_paper": lab_tot.get(0, 0) == PAPER["n_non_ied_epochs"],
        "n_recordings_equals_paper": len(rec_rows) == PAPER["n_subjects"],
        "n_ied_recordings_equals_paper": n_ied_recs == PAPER["n_ied_subjects"],
        "all_recordings_29_channels": all(r["n_channels"] == 29 for r in rec_rows),
        "epochs_tile_recordings": all(r["max_epoch_end_equals_n_samples"] for r in rec_rows),
        "ied_epochs_without_event": mismatch_total["ied_epoch_without_event"],
        "non_ied_epochs_with_event": mismatch_total["non_ied_epoch_with_event"],
        "recordings_with_marker_problems": sum(1 for r in rec_rows if r["n_marker_problems"]),
    })
    base_info = mat_dir / "base_info.csv"
    if base_info.exists():
        with open(base_info, newline="") as f:
            rows = list(csv.DictReader(f))
        checks["base_info_rows"] = len(rows)
        checks["base_info_columns"] = list(rows[0]) if rows else []
        ids = {r.get("file_name", "").replace(".mat", "") for r in rows}
        checks["base_info_ids_equal_mat_ids"] = ids == set(mat_ids)

    summary = {
        "provenance": provenance(),
        "root": str(root),
        "checks": checks,
        "epoch_label_totals": {LABELS[c]: lab_tot.get(c, 0) for c in [0] + IED_LABELS},
        "recordings_by_n_ied_classes": dict(sorted(collections.Counter(
            r["n_ied_classes"] for r in rec_rows).items())),
        "state_rule": "state in force at epoch start plus markers inside the epoch; "
                      "'mixed' if two states, 'unlabelled' only before any marker",
        "notes": notes,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary["checks"], indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("checksum", help="verify the archive against Figshare v2")
    a.add_argument("--zip", required=True)
    a.add_argument("--out", default="artifacts/dataset_audit")
    a.set_defaults(func=cmd_checksum)

    b = sub.add_parser("audit", help="patient x class x state audit of the unzipped archive")
    b.add_argument("--root", required=True, help="the unzipped opensource-dataset/ folder")
    b.add_argument("--out", default="artifacts/dataset_audit")
    b.add_argument("--interim", default="data/interim/vepiset")
    b.set_defaults(func=cmd_audit)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
