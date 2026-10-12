"""Per-recording feature extraction for the comparator ladder (06 issues 1, 5-8; §4.3).

For one recording: the kept epochs (locked exclusions), their analysis windows, and every representation:
B1 (n x 19 x 24), the 17 threshold-free B2 additions (n x 17; the salience count is added at model
assembly from peak_z and the pilot-fixed z*), ADJ (n x 912), FC = wPLI theta/alpha/beta (n x 513) and
FC_surr (n x 513, k IAAFT realisations per band). Optional exploratory FC bands are not computed here.

CLI (on the Mac only; raw data never leave it; the cache under data/interim/ is never committed):
    python src/features/extract.py --root data/raw/vepiset/extracted/opensource-dataset \
        --ids-from artifacts/protocol/f3/pilot_subset.csv --out data/interim/features/salience2s
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("features", "preprocessing", "connectivity", "data"):
    sys.path.insert(0, str(ROOT / "src" / p))
import adj as fadj  # noqa: E402
import b1 as fb1  # noqa: E402
import b2 as fb2  # noqa: E402
import surrogates as sg  # noqa: E402
import vepiset_audit as va  # noqa: E402
import windows as wn  # noqa: E402
import wpli as wp  # noqa: E402

K_SURR = {"salience2s": 5, "epoch4s": 3}


def features(x_scalp: np.ndarray, eeg_id: str, epochs: list[dict], mode: str = "salience2s",
             with_surrogates: bool = True) -> dict:
    """All representations for the given (already kept) epochs of one recording."""
    w = wn.windows(x_scalp, epochs, mode)
    xc, n = w["x_car"], w["n_samples"]
    f1 = np.stack([fb1.b1(xc[:, s:s + n]) for s in w["starts"]]) if len(epochs) else np.empty((0, 19, 24))
    out = {
        "epoch_start": np.array([e["start"] for e in epochs], dtype=int),
        "label": np.array([int(e["label"] in va.IED_LABELS) for e in epochs], dtype=int),
        "state": np.array([e["state"] for e in epochs]),
        "window_start": w["starts"], "t_raw": w["t_raw"], "peak_z": w["peak_z"],
        "b1": f1,
        "b2_topo": np.stack([fb2.topography(f) for f in f1]) if len(epochs) else np.empty((0, 17)),
        "adj": np.stack([fadj.adj(f) for f in f1]) if len(epochs) else np.empty((0, 912)),
    }
    fc = wp.real_fc(x_scalp, w["starts"], n)
    out["fc"] = np.hstack([fc[b]["wpli"] for b in wp.PRIMARY_BANDS]) if len(epochs) else np.empty((0, 513))
    out["fc_zero_denominator"] = sum(fc[b]["n_zero_denominator"] for b in wp.PRIMARY_BANDS)
    if with_surrogates:
        sf = sg.surrogate_fc(x_scalp, eeg_id, w["starts"], n, K_SURR[mode], epoch_starts=out["epoch_start"])
        out["fc_surr"] = np.hstack([sf[b]["wpli"] for b in wp.PRIMARY_BANDS]) if len(epochs) else np.empty((0, 513))
    return out


def window_has_marker(window_starts, n_samples: int, points, runs) -> np.ndarray:
    """Diagnostic only (06 issue 1): does each window contain an annotated IED marker (point or run overlap)?"""
    out = []
    for w0 in window_starts:
        a, b = w0 / wn.FS, (w0 + n_samples) / wn.FS
        out.append(any(a <= t < b for t in points) or any(r0 < b and r1 >= a for r0, r1 in runs))
    return np.array(out, dtype=bool)


def _one(args) -> tuple[str, float]:
    root, eeg_id, eps, out, mode = args
    t0 = time.time()
    import vepiset_evaluable_recount as rc

    flags = rc.signal_epoch_flags(root)(eeg_id, sorted(eps, key=lambda e: e["start"]))
    kept = wn.kept_epochs(eps, flags)
    mat = root / "MAT_Files" / f"{eeg_id}.mat"
    f = features(wn.load_scalp(mat), eeg_id, kept, mode)
    points, runs, _ = va.ied_intervals(va.read_mat_events(mat)[1])
    f["window_has_marker"] = window_has_marker(f["window_start"], wn.WIN if mode == "salience2s" else wn.EPOCH,
                                               points, runs)
    np.savez_compressed(out / f"{eeg_id}.npz", **f)
    return eeg_id, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--interim", default=ROOT / "data" / "interim" / "vepiset", type=Path)
    ap.add_argument("--ids-from", required=True, type=Path, help="CSV with an eeg_id column")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", default="salience2s", choices=list(K_SURR))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--allow-non-pilot", action="store_true",
                    help="extract features for recordings outside the frozen pilot (EXP-004 preparation only)")
    a = ap.parse_args()
    ids = sorted({r["eeg_id"] for r in csv.DictReader(a.ids_from.open())})
    pilot = {r["eeg_id"] for r in csv.DictReader((ROOT / "artifacts/protocol/f3/pilot_subset.csv").open())}
    if not a.allow_non_pilot and set(ids) - pilot:
        raise SystemExit(f"{len(set(ids) - pilot)} recordings are outside the frozen pilot; pass --allow-non-pilot")
    by = defaultdict(list)
    for e in csv.DictReader((a.interim / "epochs.csv").open()):
        if e["eeg_id"] in ids:
            by[e["eeg_id"]].append({"start": int(e["start"]), "end": int(e["end"]), "label": int(e["label"]),
                                    "state": e["state"], "has_ied_event": int(e["has_ied_event"])})
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [(a.root, i, by[i], a.out, a.mode) for i in ids if not (a.out / f"{i}.npz").exists()]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for eeg_id, sec in ex.map(_one, todo):
            print(f"{eeg_id} {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
