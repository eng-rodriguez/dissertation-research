"""Pilot runner (EXP-003) on a synthetic feature cache, and the EXP-004 blinding gate."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
import run_ladder as rl  # noqa: E402


def fake_cache(path: Path, n: int = 150, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for i in rl.pilot_ids():
        y = (rng.random(n) < 0.3).astype(int)
        np.savez_compressed(path / f"{i}.npz", label=y, state=np.where(rng.random(n) < 0.5, "wake", "sleep"),
                            b1=rng.standard_normal((n, 19, 24)) + 0.5 * y[:, None, None],
                            b2_topo=rng.standard_normal((n, 17)), adj=rng.standard_normal((n, 912)),
                            fc=rng.random((n, 513)), fc_surr=rng.random((n, 513)),
                            peak_z=rng.gamma(2, 3, (n, 19)), window_has_marker=rng.random(n) < 0.8)


def test_zstar_then_pilot_end_to_end(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    fake_cache(cache)
    monkeypatch.setattr(rl, "OUT", tmp_path / "pilot")
    monkeypatch.setattr(rl, "HGB_MODELS", ["b2", "b2_fc"])
    a = argparse.Namespace(cache=cache)
    rl.cmd_zstar(a)
    with pytest.raises(SystemExit):
        rl.cmd_zstar(a)  # fixed once
    rl.cmd_pilot(a)
    s = json.loads((tmp_path / "pilot" / "pilot_summary.json").read_text())
    assert s["evaluable_pilot_patients"] == ["DA00100U", "DA00102U", "DA00103A", "DA00103T"]
    assert set(s["mean_auroc_by_model"]) == {"b2", "b2_fc"} and 0 < s["window_validity_share"] < 1


def test_confirmatory_refuses_without_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "GATE", tmp_path / "gate.json")
    with pytest.raises(SystemExit, match="blinded"):
        rl.cmd_confirmatory(argparse.Namespace(cache=tmp_path))
    (tmp_path / "gate.json").write_text(json.dumps({"pc3": "PASS", "coverage_check": "FAIL"}))
    with pytest.raises(SystemExit, match="blinded"):
        rl.cmd_confirmatory(argparse.Namespace(cache=tmp_path))


def test_z_star_is_median_of_recording_p95_unrounded():
    rng = np.random.default_rng(5)
    peaks = {f"R{i}": rng.gamma(2, 3, (int(rng.integers(20, 200)), 19)) for i in range(8)}
    z, per = rl.z_star_from(peaks)
    assert per["R3"] == np.percentile(peaks["R3"].ravel(), 95)
    assert z == np.median([np.percentile(v.ravel(), 95) for v in peaks.values()])
    assert z != round(z, 1)


def test_gate_requires_the_adopted_m1_interval(tmp_path, monkeypatch):
    """RDR-014 (06 v1.1): M1 is the primary interval; a gate naming NB (the v1.0 primary) is refused."""
    monkeypatch.setattr(rl, "GATE", tmp_path / "gate.json")
    gate = {"pc3": "PASS", "coverage_check": "PASS", "approved_by": "x", "approved_on": "2026-10-14"}
    (tmp_path / "gate.json").write_text(json.dumps({**gate, "primary_interval": "nb"}))
    with pytest.raises(SystemExit, match="blinded"):
        rl.check_gate()
    (tmp_path / "gate.json").write_text(json.dumps({**gate, "primary_interval": "m1"}))
    assert rl.check_gate()["primary_interval"] == "m1"


def test_protocol_versions_match_their_hashes():
    """06 v1.0 (RDR-013) stays unchanged beside v1.1 (RDR-014); both are pinned in SHA256SUMS."""
    import hashlib

    d = rl.ROOT / "experiments/protocol"
    sums = dict(line.split()[::-1] for line in (d / "SHA256SUMS").read_text().splitlines())
    assert sums["06-EXPERIMENTAL-PROTOCOL-v1.0.md"].startswith("007506f1")
    for name, h in sums.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == h
