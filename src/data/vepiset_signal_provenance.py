"""vEpiSet signal provenance: channel structure, reference and filtering checks.

Loads `eeg_data` from every MAT recording and writes aggregate statistics only
(no samples, no timestamps, no clinical values such as heart rate). It checks
what the distributed signal already is; it does not re-reference, filter,
reject channels or assess signal quality.

Channel order is the authors' (vepiset_dataset README and base_trainer/dataietr.py
at 2ce4c9b); the MAT files store no channel names. Pass criteria were fixed after
an ad hoc probe of 4 recordings (DA00102S, DA00103K, DA001003, DC11304C) and
before the full run, with margin beyond the probe ranges.

Usage (from the repository root):
    python src/data/vepiset_signal_provenance.py --root data/raw/vepiset/extracted/opensource-dataset

Writes artifacts/dataset_audit/signal_provenance.csv (one row per recording) and
signal_provenance_summary.json (pass counts, metric ranges, criteria, provenance).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import vepiset_audit as va
from scipy import signal
from scipy.stats import spearmanr

FS = 500
NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
         "Fz", "Cz", "Pz", "PG1", "PG2", "A1", "A2", "ECG1", "ECG2", "EMG1", "EMG2", "EMG3", "EMG4"]
SCALP, AUX, ECG, EMG = list(range(19)), [19, 20, 21, 22], [23, 24], [25, 26, 27, 28]
# Approximate 2-D 10-20 positions (unit head radius), used only for the topography test.
POS = {"Fp1": (-.31, .95), "Fp2": (.31, .95), "F7": (-.81, .59), "F3": (-.39, .55), "Fz": (0, .5),
       "F4": (.39, .55), "F8": (.81, .59), "T3": (-1, 0), "C3": (-.5, 0), "Cz": (0, 0), "C4": (.5, 0),
       "T4": (1, 0), "T5": (-.81, -.59), "P3": (-.39, -.55), "Pz": (0, -.5), "P4": (.39, -.55),
       "T6": (.81, -.59), "O1": (-.31, -.95), "O2": (.31, -.95)}
LEFT = {"Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5"}
RIGHT = {"Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6"}
N_PERM = 2000

# Fixed before the 84-recording run. dB values are relative to each row's 30-40 Hz power.
CRITERIA = {
    "notch": "50 Hz dip (49.8-50.2 Hz vs 47-49 and 51-53 Hz) <= -20 dB in EEG, ECG and EMG rows",
    "lowpass_eeg": "EEG rows 1-23: 100-120 Hz <= -40 dB and 60-65 Hz >= -15 dB",
    "lowpass_eeg_only": "ECG rows 100-120 Hz at least 20 dB above EEG rows 100-120 Hz",
    "c3c4_anticorrelation": "corr(C3, C4) in 1-30 Hz <= -0.8",
    "common_reference": "no flat row; var(mean of 19 scalp rows)/mean scalp-row var >= 0.05 "
                        "(not average-referenced); corr(A1, A2) > -0.5 (not linked ears)",
    "scalp_topography": "Spearman(avg-referenced row correlation, 10-20 distance) < 0 with "
                        f"permutation p <= 0.01 ({N_PERM} label permutations)",
    "ecg_rows": "ECG1-ECG2 shows a regular rhythm: median beat interval 0.33-1.5 s, IQR/median <= 0.35",
    "emg_rows": "EMG rows 5-10 Hz <= -30 dB",
}
FLAGS = list(CRITERIA)
FIELDS = ["eeg_id", "n_channels", "n_samples", "n_flat_rows", "unique_sample_frac_fp1",
          "notch_db_eeg", "notch_db_ecg", "notch_db_emg", "eeg_60_65_db", "eeg_80_85_db",
          "eeg_100_120_db", "ecg_100_120_db", "emg_100_120_db", "emg_5_10_db",
          "r_c3_c4", "c3c4_sum_diff_var_ratio", "scalp_mean_var_ratio", "r_a1_a2",
          "rho_topography", "perm_p_topography", "aux_lateral_matches"] + [f"pass_{f}" for f in FLAGS]


def probe(x: np.ndarray) -> dict:
    """Aggregate provenance metrics for one recording (x: 29 x n, volts)."""
    out: dict = {"n_channels": x.shape[0], "n_samples": x.shape[1]}
    xu = x * 1e6
    out["n_flat_rows"] = int(np.sum(xu.std(1) == 0))
    out["unique_sample_frac_fp1"] = len(np.unique(x[0])) / x.shape[1]

    f, p = signal.welch(xu, fs=FS, nperseg=FS * 20, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(p / p[:, (f >= 30) & (f < 40)].mean(1, keepdims=True))

    def band(rows, lo, hi):
        return float(np.median(db[rows][:, (f >= lo) & (f < hi)].mean(1)))

    eeg = SCALP + AUX
    for name, rows in [("eeg", eeg), ("ecg", ECG), ("emg", EMG)]:
        out[f"notch_db_{name}"] = band(rows, 49.8, 50.2) - (band(rows, 47, 49) + band(rows, 51, 53)) / 2
    out["eeg_60_65_db"] = band(eeg, 60, 65)
    out["eeg_80_85_db"] = band(eeg, 80, 85)
    out["eeg_100_120_db"] = band(eeg, 100, 120)
    out["ecg_100_120_db"] = band(ECG, 100, 120)
    out["emg_100_120_db"] = band(EMG, 100, 120)
    out["emg_5_10_db"] = band(EMG, 5, 10)

    y = signal.filtfilt(*signal.butter(4, [1, 30], btype="band", fs=FS), xu, axis=-1)
    s = y[SCALP]
    m = s.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(y)
        out["r_c3_c4"] = float(c[4, 5])
        out["c3c4_sum_diff_var_ratio"] = float(np.var(y[4] + y[5]) / np.var(y[4] - y[5]))
        out["scalp_mean_var_ratio"] = float(m.var() / s.var(1).mean())
        out["r_a1_a2"] = float(c[21, 22])

        a = s - m
        ca = np.corrcoef(a)
        pos = np.array([POS[n] for n in NAMES[:19]])
        iu = np.triu_indices(19, 1)

        def rho(order):
            d = np.linalg.norm(pos[order][:, None] - pos[order][None], axis=-1)
            return spearmanr(ca[iu], d[iu])[0]

        out["rho_topography"] = float(rho(np.arange(19)))
        rng = np.random.default_rng(0)
        perm = np.array([rho(rng.permutation(19)) for _ in range(N_PERM)])
        out["perm_p_topography"] = float((np.sum(perm <= out["rho_topography"]) + 1) / (N_PERM + 1))

        z = np.corrcoef(np.vstack([a, y[AUX] - m]))
        matches = 0
        for i, side in zip(range(4), [LEFT, RIGHT, LEFT, RIGHT]):  # PG1, PG2, A1, A2
            top2 = {NAMES[k] for k in np.argsort(-np.nan_to_num(z[19 + i, :19], nan=-2))[:2]}
            matches += top2 <= side
        out["aux_lateral_matches"] = matches

    e = signal.filtfilt(*signal.butter(4, [5, 30], btype="band", fs=FS), xu[23] - xu[24])
    pk, _ = signal.find_peaks(np.abs(e), distance=int(0.33 * FS), height=np.percentile(np.abs(e), 99) * 0.5)
    ibi = np.diff(pk) / FS
    if len(ibi) >= 10:
        med = float(np.median(ibi))
        iqr = float(np.subtract(*np.percentile(ibi, [75, 25])))
        ecg_ok = 0.33 <= med <= 1.5 and iqr / med <= 0.35
    else:
        ecg_ok = False

    out["pass_notch"] = all(out[f"notch_db_{g}"] <= -20 for g in ("eeg", "ecg", "emg"))
    out["pass_lowpass_eeg"] = out["eeg_100_120_db"] <= -40 and out["eeg_60_65_db"] >= -15
    out["pass_lowpass_eeg_only"] = out["ecg_100_120_db"] - out["eeg_100_120_db"] >= 20
    out["pass_c3c4_anticorrelation"] = out["r_c3_c4"] <= -0.8
    out["pass_common_reference"] = (out["n_flat_rows"] == 0 and out["scalp_mean_var_ratio"] >= 0.05
                                    and out["r_a1_a2"] > -0.5)
    out["pass_scalp_topography"] = out["rho_topography"] < 0 and out["perm_p_topography"] <= 0.01
    out["pass_ecg_rows"] = bool(ecg_ok)
    out["pass_emg_rows"] = out["emg_5_10_db"] <= -30
    for k, v in out.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, float):
            out[k] = round(v, 3)
    return out


def summarize(rows: list[dict]) -> dict:
    flags = {}
    for f in FLAGS:
        fail = [r["eeg_id"] for r in rows if not r[f"pass_{f}"]]
        flags[f] = {"criterion": CRITERIA[f], "n_pass": len(rows) - len(fail), "failing": fail}
    metrics = {}
    for k in FIELDS[1:]:
        if k.startswith("pass_"):
            continue
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        metrics[k] = ({"min": round(float(v.min()), 3), "median": round(float(np.median(v)), 3),
                       "max": round(float(v.max()), 3)} if len(v) else None)
    return {"n_recordings": len(rows), "checks": flags, "metrics": metrics,
            "channel_order": NAMES, "provenance": va.provenance()}


def run(mat_dir: Path, out: Path, ids: list[str] | None = None) -> dict:
    paths = sorted(mat_dir.glob("*.mat"))
    if ids:
        paths = [p for p in paths if p.stem in set(ids)]
    rows = []
    for i, path in enumerate(paths, 1):
        x = scipy.io.loadmat(path, variable_names=["eeg_data"])["eeg_data"]
        rows.append({"eeg_id": path.stem, **probe(x)})
        print(f"[{i}/{len(paths)}] {path.stem}", file=sys.stderr)
    va.write_csv(out / "signal_provenance.csv", rows, FIELDS)
    summary = summarize(rows)
    (out / "signal_provenance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="the unzipped opensource-dataset/ folder")
    ap.add_argument("--out", default="artifacts/dataset_audit")
    ap.add_argument("--ids", nargs="*", help="restrict to these recording IDs")
    args = ap.parse_args(argv)
    summary = run(Path(args.root) / "MAT_Files", Path(args.out), args.ids)
    for f, c in summary["checks"].items():
        print(f"{f:24s} {c['n_pass']}/{summary['n_recordings']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
