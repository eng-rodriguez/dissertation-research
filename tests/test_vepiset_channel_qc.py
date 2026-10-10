"""Checks the F-4 channel QC on synthetic recordings with known faults."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
import vepiset_channel_qc as qc

FS = qc.FS


def _electrodes(seconds=120, seed=1):
    """19 scalp electrode potentials (uV) with a smooth spatial field, before referencing."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    pos = np.array([qc.POS[c] for c in qc.NAMES])
    centres = rng.uniform(-1, 1, (8, 2))
    gain = np.exp(-np.sum((pos[:, None] - centres[None]) ** 2, -1) / (2 * 0.35 ** 2))
    sources = signal.lfilter([1], [1, -0.95], rng.normal(size=(8, n)), axis=-1)
    return 10 * (gain @ sources) + 0.3 * rng.normal(size=(19, n))


def _bad(n, seed, sd=40.0):
    """Plausible-amplitude but spatially unrelated signal (broken electrode), 0.5-40 Hz."""
    r = np.random.default_rng(seed).normal(size=n)
    r = signal.sosfiltfilt(signal.butter(4, [0.5, 40], btype="band", fs=FS, output="sos"), r)
    return sd * r / r.std()


def _record(e, seed=1):
    """vEpiSet-like 29-row recording in volts: (C3+C4)/2 reference, 50 Hz notch, 70 Hz high-cut."""
    rng = np.random.default_rng(seed + 100)
    n = e.shape[1]
    x = np.vstack([e, rng.normal(size=(10, n))])
    x[:19] -= (e[qc.C3] + e[qc.C4]) / 2
    x = signal.filtfilt(*signal.iirnotch(50, 30, fs=FS), x, axis=-1)
    x[:19] = signal.sosfiltfilt(signal.butter(8, 70, output="sos", fs=FS), x[:19], axis=-1)
    return x * 1e-6


def _idx(name):
    return qc.NAMES.index(name)


def test_neighbours_are_symmetric_and_nonempty():
    for i in qc.SCALP:
        assert len(qc.NEIGHBOURS[i]) >= 3
        for j in qc.NEIGHBOURS[i]:
            assert i in qc.NEIGHBOURS[j]


def test_clean_recording_is_primary_eligible():
    out = qc.channel_qc(_record(_electrodes()))
    assert out["n_bad"] == 0, out
    assert out["c3c4_satisfactory"] and out["primary_eligible"] and out["sensitivity_eligible"]
    assert out["r_c3_c4"] < -0.99  # stored C3/C4 are mirror images under a (C3+C4)/2 reference


def test_flat_stored_row_is_flagged():
    x = _record(_electrodes())
    x[_idx("O1")] = 0.0
    out = qc.channel_qc(x)
    assert "O1" in out["flat_channels"].split(";")
    assert not out["primary_eligible"]


def test_dead_electrode_fails_neighbour_correlation():
    e = _electrodes()
    e[_idx("P4")] = _bad(e.shape[1], 7)
    out = qc.channel_qc(_record(e))
    assert out["low_neighbour_correlation_channels"] == "P4", out
    assert out["n_bad"] == 1 and not out["primary_eligible"] and out["sensitivity_eligible"]


def test_extreme_amplitude_channel_is_flagged():
    e = _electrodes()
    e[_idx("T4")] += 400 * np.random.default_rng(8).normal(size=e.shape[1])
    out = qc.channel_qc(_record(e))
    assert "T4" in out["extreme_amplitude_channels"].split(";"), out


def test_clipped_channel_is_flagged():
    x = _record(_electrodes())
    i = _idx("Fz")
    lim = np.percentile(np.abs(x[i]), 98)
    x[i] = np.clip(x[i], -lim, lim)
    out = qc.channel_qc(x)
    assert "Fz" in out["clipping_channels"].split(";"), out


def test_bad_c3_makes_c3c4_unsatisfactory():
    e = _electrodes()
    e[qc.C3] = _bad(e.shape[1], 9)
    out = qc.channel_qc(_record(e))
    assert not out["c3c4_satisfactory"]
    assert not out["primary_eligible"]


def test_adjacent_bad_pair_is_not_sensitivity_eligible():
    e = _electrodes()
    for k, name in enumerate(("O1", "O2")):  # neighbours
        e[_idx(name)] = _bad(e.shape[1], 10 + k)
    out = qc.channel_qc(_record(e))
    assert out["n_bad"] == 2 and out["bad_adjacent_pair"]
    assert not out["sensitivity_eligible"]

    e = _electrodes()
    for k, name in enumerate(("Fp1", "O2")):  # not neighbours
        e[_idx(name)] = _bad(e.shape[1], 20 + k)
    out = qc.channel_qc(_record(e))
    assert out["n_bad"] == 2 and not out["bad_adjacent_pair"]
    assert out["sensitivity_eligible"] and not out["primary_eligible"]


def test_ied_like_transients_are_not_flagged():
    """High-amplitude focal spikes must not make a channel bad or flag its epochs."""
    e = _electrodes()
    t = np.arange(-50, 51)
    spike = 150 * np.exp(-0.5 * (t / 8) ** 2)
    for start in range(1000, e.shape[1] - 1000, 3000):
        for name, w in (("F7", 1.0), ("T3", 0.8), ("F3", 0.5)):
            e[_idx(name), start - 50:start + 51] += w * spike
    x = _record(e)
    out = qc.channel_qc(x)
    assert out["n_bad"] == 0, out
    eps = [{"start": s, "end": s + 2000} for s in range(0, x.shape[1] - 1999, 2000)]
    assert not any(qc.epoch_flags(x, eps, out["_clip_level"]))


def test_epoch_flags_detect_flat_segment():
    x = _record(_electrodes())
    out = qc.channel_qc(x)
    x[_idx("Cz"), 4000:6000] = 0.0
    eps = [{"start": s, "end": s + 2000} for s in (0, 2000, 4000, 6000)]
    assert qc.epoch_flags(x, eps, out["_clip_level"]) == [False, False, True, False]


def _write_dataset(tmp_path):
    root = tmp_path / "opensource-dataset"
    (root / "MAT_Files").mkdir(parents=True)
    good = _record(_electrodes(seconds=40, seed=2), seed=2)
    e = _electrodes(seconds=40, seed=4)
    e[_idx("P4")] = _bad(e.shape[1], 3)
    bad = _record(e, seed=4)
    for rid, x in (("REC1", good), ("REC2", bad)):
        scipy.io.savemat(root / "MAT_Files" / f"{rid}.mat", {"eeg_data": x, "events": np.array([["0"]])})
        for k in range(x.shape[1] // 2000):
            label = 2 if (rid == "REC1" and k < 3) else 0
            folder = root / ("frontal" if label else "non-IED")
            folder.mkdir(exist_ok=True)
            np.save(folder / f"{rid}_{k * 2000}_{(k + 1) * 2000}_500__{label}.npy", np.zeros(1))
    sm = tmp_path / "subject_state_matrix.csv"
    with sm.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eeg_id", "all_wake", "all_sleep", "all_mixed", "all_unlabelled",
                    "ied_wake", "ied_sleep", "ied_mixed", "ied_unlabelled"])
        w.writerow(["REC1", 10, 0, 0, 0, 5, 0, 0, 0])
        w.writerow(["REC2", 10, 0, 0, 0, 0, 0, 0, 0])
    return root, sm


def test_run_writes_aggregate_outputs_only(tmp_path):
    root, sm = _write_dataset(tmp_path)
    out = tmp_path / "out"
    summary = qc.run(root, out, sm)
    assert summary["n_recordings"] == 2
    assert summary["primary_eligible"] == {"total": 1, "ied_positive": 1, "ied_free": 0}
    assert summary["pre_qc_eligible_ied_positive"]["n_surviving_primary"] == 1
    assert summary["stop_rule"]["applicable"] is False
    assert summary["epochs"]["ied_n"] == 3
    rows = list(csv.DictReader((out / "channel_qc.csv").open()))
    assert list(rows[0]) == qc.FIELDS
    assert {r["eeg_id"]: r["bad_channels"] for r in rows} == {"REC1": "", "REC2": "P4"}
    assert "_clip_level" not in rows[0]
    text = (out / "channel_qc_summary.json").read_text()
    json.loads(text)
    assert "_clip_level" not in text


def test_stop_rule_logic():
    rows = [{"eeg_id": f"R{i}", "primary_eligible": i >= 6, "sensitivity_eligible": True, "n_bad": 0 if i >= 6 else 1,
             "n_epochs": 1, "n_epochs_flagged": 0, "n_ied_epochs": 0, "n_ied_epochs_flagged": 0,
             **{f"{f}_channels": "" for f in qc.CHANNEL_FLAGS}} for i in range(84)]
    eligible = {f"R{i}" for i in range(46)}
    s = qc.summarize(rows, set(eligible), eligible)
    assert s["pre_qc_eligible_ied_positive"]["n_surviving_primary"] == 40
    assert s["stop_rule"]["decision"].startswith("STOP")
    rows[5]["primary_eligible"] = True
    s = qc.summarize(rows, set(eligible), eligible)
    assert s["stop_rule"]["decision"] == "CONTINUE"
