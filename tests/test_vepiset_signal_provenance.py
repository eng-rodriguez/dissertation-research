"""Checks the signal provenance probe on synthetic recordings with known structure."""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
import vepiset_signal_provenance as sp

FS = sp.FS
AUX_POS = {"PG1": (-.95, .3), "PG2": (.95, .3), "A1": (-1.1, -.1), "A2": (1.1, -.1)}


def _synthetic(seconds=120, filtered=True, reference="c3c4", seed=1):
    """29-row recording: smooth scalp field, ECG, EMG; optionally filtered like vEpiSet."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    pos = np.array([sp.POS[c] for c in sp.NAMES[:19]] + [AUX_POS[c] for c in sp.NAMES[19:23]])
    centres = rng.uniform(-1, 1, (8, 2))
    gain = np.exp(-np.sum((pos[:, None] - centres[None]) ** 2, -1) / (2 * 0.35 ** 2))
    sources = signal.lfilter([1], [1, -0.95], rng.normal(size=(8, n)), axis=-1)
    eeg = gain @ sources + 0.3 * rng.normal(size=(23, n)) + 2 * np.sin(2 * np.pi * 50 * np.arange(n) / FS)
    beats = np.zeros(n)
    beats[np.arange(FS // 2, n, int(0.8 * FS))] = 1
    qrs = np.convolve(beats, np.exp(-0.5 * (np.arange(-25, 26) / 4) ** 2), "same") * 50
    ecg = np.vstack([qrs, -qrs]) + 0.5 * rng.normal(size=(2, n))
    emg = signal.filtfilt(*signal.butter(4, 20, "high", fs=FS), rng.normal(size=(4, n)), axis=-1)
    x = np.vstack([eeg, ecg, emg])
    if reference == "c3c4":
        x[:25] -= (x[4] + x[5]) / 2
    elif reference == "average":
        x[:19] -= x[:19].mean(0)
    if filtered:
        x = signal.filtfilt(*signal.iirnotch(50, 30, fs=FS), x, axis=-1)
        x[:23] = signal.sosfiltfilt(signal.butter(8, 70, output="sos", fs=FS), x[:23], axis=-1)
    return x * 1e-6


def test_probe_passes_on_vepiset_like_signal():
    out = sp.probe(_synthetic())
    failed = [f for f in sp.FLAGS if not out[f"pass_{f}"]]
    assert failed == [], (failed, out)
    assert out["aux_lateral_matches"] == 4


def test_probe_flags_unfiltered_average_referenced_signal():
    out = sp.probe(_synthetic(filtered=False, reference="average"))
    assert not out["pass_notch"]
    assert not out["pass_lowpass_eeg"]
    assert not out["pass_common_reference"]
    assert not out["pass_c3c4_anticorrelation"]
    assert out["pass_scalp_topography"]


def test_run_writes_aggregate_outputs_only(tmp_path):
    mat = tmp_path / "MAT_Files"
    mat.mkdir()
    scipy.io.savemat(mat / "A001.mat", {"eeg_data": _synthetic(seconds=60), "events": np.array([["0"]])})
    summary = sp.run(mat, tmp_path / "out")
    assert summary["n_recordings"] == 1
    assert all(c["n_pass"] == 1 for c in summary["checks"].values())
    header = (tmp_path / "out" / "signal_provenance.csv").read_text().splitlines()[0].split(",")
    assert header == sp.FIELDS
    assert not any("ibi" in h or "heart" in h for h in header)
    json.loads((tmp_path / "out" / "signal_provenance_summary.json").read_text())
