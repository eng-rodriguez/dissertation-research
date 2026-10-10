"""vEpiSet F-4: label-blind 19-channel quality control and FC eligibility.

Implements 06-EXPERIMENTAL-PROTOCOL.md issue 12 (draft v0.3). Every threshold is in
CRITERIA below and was fixed before the 84-recording run; none may be changed after
the results are seen (Josue, 2026-10-10). The QC is label-blind: channel and epoch
flags are computed from the signal only. Class labels are read afterwards, only to
report epoch exclusion rates by class and to count surviving IED-positive recordings.

Primary eligibility (confirmatory FC analysis): no bad scalp channel, which includes
satisfactory C3/C4 behaviour. No interpolation. Interpolation sensitivity set: at most
2 bad scalp channels that are not neighbours.

Stop rule (06 issue 12, fixed before the run): continue only if at least 41 of the 46
pre-QC-eligible IED-positive recordings pass the primary rule. Otherwise stop and
return for a protocol decision.

Known limitation (from the synthetic tests): the neighbour-correlation criterion uses
the 95th percentile of 3-5 neighbour correlations, which is close to the maximum, so it
is lenient. A broken channel whose 1-30 Hz signal is small next to the common mode, or
two adjacent broken channels, can escape it. Per-criterion, per-channel flag counts are
reported so such patterns stay visible.

Writes aggregate outputs only (per-recording flags and summary metrics, no samples,
no timestamps, no clinical values):
    artifacts/dataset_audit/channel_qc.csv
    artifacts/dataset_audit/channel_qc_summary.json

Usage (from the repository root):
    python src/data/vepiset_channel_qc.py --root data/raw/vepiset/extracted/opensource-dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import vepiset_audit as va
from scipy import signal

FS = 500
NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
         "Fz", "Cz", "Pz"]
SCALP = list(range(19))
C3, C4 = NAMES.index("C3"), NAMES.index("C4")
# Approximate 2-D 10-20 positions (unit head radius); same as vepiset_signal_provenance.POS.
POS = {"Fp1": (-.31, .95), "Fp2": (.31, .95), "F7": (-.81, .59), "F3": (-.39, .55), "Fz": (0, .5),
       "F4": (.39, .55), "F8": (.81, .59), "T3": (-1, 0), "C3": (-.5, 0), "Cz": (0, 0), "C4": (.5, 0),
       "T4": (1, 0), "T5": (-.81, -.59), "P3": (-.39, -.55), "Pz": (0, -.5), "P4": (.39, -.55),
       "T6": (.81, -.59), "O1": (-.31, -.95), "O2": (.31, -.95)}

# Fixed 2026-10-10, before the 84-recording run. Do not change after seeing results.
CRITERIA = {
    "units": "eeg_data in volts, converted to microvolts",
    "qc_reference": "flat and clipping use the stored referential rows; extreme amplitude and neighbour "
                    "correlation use a per-sample median-across-19-channels reference (QC only)",
    "flat": "robust SD (1.4826 x MAD) < 1.0 uV, or > 5% of first differences with |diff| < 0.01 uV",
    "clipping": "> 0.1% of samples with |x| >= 0.999 x max|x| of that channel",
    "extreme_amplitude": "z > 5, z = (log robust SD - median over channels) / max(1.4826 x MAD over channels, 0.1)",
    "low_neighbour_correlation": "in > 50% of non-overlapping 4-s windows, the 95th percentile of the channel's "
                                 "1-30 Hz correlations with its neighbours is < 0.4",
    "neighbours": "10-20 electrodes within 0.65 head radii (2-D layout)",
    "c3c4_satisfactory": "neither C3 nor C4 is flagged by any channel criterion",
    "epoch_flag": "authors' epoch with any scalp channel of robust SD < 0.5 uV within the epoch, or > 1% of "
                  "its samples at or above the recording-level clipping level",
    "primary_eligible": "no bad scalp channel (includes satisfactory C3/C4); no interpolation",
    "sensitivity_eligible": "primary eligible, or 1-2 bad scalp channels that are not neighbours "
                            "(interpolated identically for all representations; sensitivity only)",
    "stop_rule": ">= 41 of the 46 pre-QC-eligible IED-positive recordings must pass the primary rule",
}
FLAT_RSD_UV, FLAT_DIFF_UV, FLAT_DIFF_FRAC = 1.0, 0.01, 0.05
CLIP_LEVEL, CLIP_FRAC = 0.999, 0.001
EXTREME_Z, EXTREME_MAD_FLOOR = 5.0, 0.1
NB_RADIUS, NB_CORR, NB_PCTL, NB_WIN_FRAC = 0.65, 0.4, 95, 0.5
WIN = 4 * FS
EPOCH_FLAT_RSD_UV, EPOCH_CLIP_FRAC = 0.5, 0.01
ELIGIBLE_MIN_EPOCHS = 5  # pre-QC eligibility of IED-positive recordings (06 §2)
STOP_MIN_SURVIVORS, PRE_QC_ELIGIBLE = 41, 46
CHANNEL_FLAGS = ["flat", "clipping", "extreme_amplitude", "low_neighbour_correlation"]

FIELDS = (["eeg_id", "n_samples", "n_bad", "bad_channels"] + [f"{f}_channels" for f in CHANNEL_FLAGS]
          + ["r_c3_c4", "c3c4_sum_diff_var_ratio", "c3_low_corr_window_frac", "c4_low_corr_window_frac",
             "c3c4_satisfactory", "bad_adjacent_pair", "primary_eligible", "sensitivity_eligible",
             "n_epochs", "n_epochs_flagged", "n_ied_epochs", "n_ied_epochs_flagged",
             "ied_positive", "pre_qc_eligible_ied"])


def neighbours() -> list[list[int]]:
    pos = np.array([POS[n] for n in NAMES])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return [[j for j in SCALP if j != i and d[i, j] <= NB_RADIUS] for i in SCALP]


NEIGHBOURS = neighbours()


def robust_sd(x: np.ndarray, axis: int = -1) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def channel_qc(x: np.ndarray) -> dict:
    """Label-blind QC of one recording. x: (>=19, n) volts; rows 0-18 are scalp."""
    s = x[:19] * 1e6
    n = s.shape[1]
    out: dict = {"n_samples": n}
    flags = {f: np.zeros(19, bool) for f in CHANNEL_FLAGS}

    rsd = robust_sd(s)
    small_diff = np.mean(np.abs(np.diff(s, axis=1)) < FLAT_DIFF_UV, axis=1)
    flags["flat"] = (rsd < FLAT_RSD_UV) | (small_diff > FLAT_DIFF_FRAC)

    peak = np.abs(s).max(axis=1)
    clip_level = CLIP_LEVEL * peak
    flags["clipping"] = np.mean(np.abs(s) >= clip_level[:, None], axis=1) > CLIP_FRAC

    y = s - np.median(s, axis=0, keepdims=True)
    with np.errstate(divide="ignore"):
        lsd = np.log(robust_sd(y))
    finite = np.isfinite(lsd)
    lsd = np.where(finite, lsd, -np.inf)
    centre = np.median(lsd[finite]) if finite.any() else 0.0
    scale = max(1.4826 * np.median(np.abs(lsd[finite] - centre)) if finite.any() else 0.0, EXTREME_MAD_FLOOR)
    flags["extreme_amplitude"] = (lsd - centre) / scale > EXTREME_Z

    yb = signal.sosfiltfilt(signal.butter(4, [1, 30], btype="band", fs=FS, output="sos"), y, axis=-1)
    n_win = n // WIN
    low = np.zeros((19, max(n_win, 1)), bool)
    for w in range(n_win):
        seg = yb[:, w * WIN:(w + 1) * WIN]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.nan_to_num(np.corrcoef(seg), nan=0.0)
        for i in SCALP:
            low[i, w] = np.percentile(c[i, NEIGHBOURS[i]], NB_PCTL) < NB_CORR
    low_frac = low[:, :n_win].mean(axis=1) if n_win else np.ones(19)
    flags["low_neighbour_correlation"] = low_frac > NB_WIN_FRAC

    bad = np.zeros(19, bool)
    for f in CHANNEL_FLAGS:
        bad |= flags[f]
        out[f"{f}_channels"] = ";".join(NAMES[i] for i in np.flatnonzero(flags[f]))
    bad_idx = np.flatnonzero(bad)
    out["n_bad"] = int(bad.sum())
    out["bad_channels"] = ";".join(NAMES[i] for i in bad_idx)

    sb = signal.sosfiltfilt(signal.butter(4, [1, 30], btype="band", fs=FS, output="sos"), s[[C3, C4]], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["r_c3_c4"] = round(float(np.corrcoef(sb)[0, 1]), 3)
        out["c3c4_sum_diff_var_ratio"] = round(float(np.var(sb[0] + sb[1]) / np.var(sb[0] - sb[1])), 4)
    out["c3_low_corr_window_frac"] = round(float(low_frac[C3]), 3)
    out["c4_low_corr_window_frac"] = round(float(low_frac[C4]), 3)
    out["c3c4_satisfactory"] = bool(not bad[C3] and not bad[C4])

    adjacent = any(j in NEIGHBOURS[i] for i in bad_idx for j in bad_idx if i < j)
    out["bad_adjacent_pair"] = bool(adjacent)
    out["primary_eligible"] = bool(out["n_bad"] == 0 and out["c3c4_satisfactory"])
    out["sensitivity_eligible"] = bool(out["primary_eligible"] or (out["n_bad"] <= 2 and not adjacent))
    out["_clip_level"] = clip_level
    return out


def epoch_flags(x: np.ndarray, epochs: list[dict], clip_level: np.ndarray) -> list[bool]:
    """Label-blind epoch flags for the authors' epochs of one recording."""
    s = x[:19] * 1e6
    flagged = []
    for e in epochs:
        seg = s[:, e["start"]:e["end"]]
        if seg.shape[1] == 0:
            flagged.append(True)
            continue
        flat = bool(np.any(robust_sd(seg) < EPOCH_FLAT_RSD_UV))
        clip = bool(np.any(np.mean(np.abs(seg) >= clip_level[:, None], axis=1) > EPOCH_CLIP_FRAC))
        flagged.append(flat or clip)
    return flagged


def pre_qc_eligible_ied(state_matrix: Path) -> tuple[set[str], set[str]]:
    """IED-positive recordings, and those with >=5 IED and >=5 non-IED epochs in a common state (06 §2)."""
    ied, eligible = set(), set()
    with state_matrix.open() as f:
        for r in csv.DictReader(f):
            if sum(int(r[k]) for k in ("ied_wake", "ied_sleep", "ied_mixed", "ied_unlabelled")) == 0:
                continue
            ied.add(r["eeg_id"])
            for st in ("wake", "sleep"):
                pos = int(r[f"ied_{st}"])
                if pos >= ELIGIBLE_MIN_EPOCHS and int(r[f"all_{st}"]) - pos >= ELIGIBLE_MIN_EPOCHS:
                    eligible.add(r["eeg_id"])
    return ied, eligible


def summarize(rows: list[dict], ied: set[str], eligible: set[str]) -> dict:
    ids = {r["eeg_id"] for r in rows}
    primary = {r["eeg_id"] for r in rows if r["primary_eligible"]}
    sensitivity = {r["eeg_id"] for r in rows if r["sensitivity_eligible"]}
    elig_present = eligible & ids
    survivors = elig_present & primary
    n_ep = sum(r["n_epochs"] for r in rows)
    n_ep_f = sum(r["n_epochs_flagged"] for r in rows)
    n_ied = sum(r["n_ied_epochs"] for r in rows)
    n_ied_f = sum(r["n_ied_epochs_flagged"] for r in rows)
    channel_counts = {f: {n: sum(n in r[f"{f}_channels"].split(";") for r in rows) for n in NAMES}
                      for f in CHANNEL_FLAGS}
    da = next((r for r in rows if r["eeg_id"] == "DA00100Y"), None)
    stop_applicable = len(elig_present) == PRE_QC_ELIGIBLE
    return {
        "n_recordings": len(rows),
        "primary_eligible": {"total": len(primary), "ied_positive": len(primary & ied),
                             "ied_free": len(primary - ied)},
        "pre_qc_eligible_ied_positive": {"n_pre_qc": len(elig_present), "n_surviving_primary": len(survivors),
                                         "removed": sorted(elig_present - primary)},
        "sensitivity_eligible": {"total": len(sensitivity), "ied_positive": len(sensitivity & ied),
                                 "pre_qc_eligible_ied_positive": len(sensitivity & elig_present)},
        "n_recordings_by_bad_count": {str(k): sum(r["n_bad"] == k for r in rows)
                                      for k in sorted({r["n_bad"] for r in rows})},
        "bad_channel_counts_by_criterion": channel_counts,
        "DA00100Y": None if da is None else {
            k: da[k] for k in ("r_c3_c4", "c3c4_sum_diff_var_ratio", "c3_low_corr_window_frac",
                               "c4_low_corr_window_frac", "c3c4_satisfactory", "bad_channels",
                               "primary_eligible")},
        "epochs": {"n": n_ep, "flagged": n_ep_f, "ied_n": n_ied, "ied_flagged": n_ied_f,
                   "non_ied_n": n_ep - n_ied, "non_ied_flagged": n_ep_f - n_ied_f,
                   "ied_flag_rate": round(n_ied_f / n_ied, 4) if n_ied else None,
                   "non_ied_flag_rate": round((n_ep_f - n_ied_f) / (n_ep - n_ied), 4) if n_ep > n_ied else None},
        "stop_rule": {"rule": CRITERIA["stop_rule"], "applicable": stop_applicable,
                      "survivors": len(survivors), "threshold": STOP_MIN_SURVIVORS,
                      "decision": (("CONTINUE" if len(survivors) >= STOP_MIN_SURVIVORS else
                                    "STOP: return to Josue for a protocol decision") if stop_applicable
                                   else "NOT APPLICABLE: not the full 84-recording run")},
        "criteria": CRITERIA,
        "neighbours": {NAMES[i]: [NAMES[j] for j in NEIGHBOURS[i]] for i in SCALP},
        "provenance": va.provenance(),
    }


def run(root: Path, out: Path, state_matrix: Path, ids: list[str] | None = None) -> dict:
    paths = sorted((root / "MAT_Files").glob("*.mat"))
    if ids:
        paths = [p for p in paths if p.stem in set(ids)]
    epochs, _ = va.read_epochs(root)
    by_rec: dict[str, list[dict]] = {}
    for e in epochs:
        by_rec.setdefault(e["eeg_id"], []).append(e)
    ied, eligible = pre_qc_eligible_ied(state_matrix)
    rows = []
    for i, path in enumerate(paths, 1):
        x = scipy.io.loadmat(path, variable_names=["eeg_data"])["eeg_data"]
        r = channel_qc(x)
        clip_level = r.pop("_clip_level")
        eps = by_rec.get(path.stem, [])
        ef = epoch_flags(x, eps, clip_level)
        is_ied = [e["label"] in va.IED_LABELS for e in eps]
        r.update({"eeg_id": path.stem, "n_epochs": len(eps), "n_epochs_flagged": int(sum(ef)),
                  "n_ied_epochs": int(sum(is_ied)),
                  "n_ied_epochs_flagged": int(sum(f and d for f, d in zip(ef, is_ied))),
                  "ied_positive": path.stem in ied, "pre_qc_eligible_ied": path.stem in eligible})
        rows.append(r)
        print(f"[{i}/{len(paths)}] {path.stem}", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)
    va.write_csv(out / "channel_qc.csv", rows, FIELDS)
    summary = summarize(rows, ied, eligible)
    (out / "channel_qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="the unzipped opensource-dataset/ folder")
    ap.add_argument("--out", default="artifacts/dataset_audit")
    ap.add_argument("--state-matrix", default="artifacts/dataset_audit/subject_state_matrix.csv")
    ap.add_argument("--ids", nargs="*", help="restrict to these recording IDs (testing only)")
    args = ap.parse_args(argv)
    s = run(Path(args.root), Path(args.out), Path(args.state_matrix), args.ids)
    p = s["pre_qc_eligible_ied_positive"]
    print(f"recordings: {s['n_recordings']}; primary eligible: {s['primary_eligible']['total']} "
          f"({s['primary_eligible']['ied_positive']} IED-positive, {s['primary_eligible']['ied_free']} IED-free)")
    print(f"pre-QC-eligible IED-positive surviving primary rule: {p['n_surviving_primary']} of {p['n_pre_qc']}")
    print(f"DA00100Y: {s['DA00100Y']}")
    print(f"stop rule: {s['stop_rule']['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
