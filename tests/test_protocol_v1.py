"""Checks the protocol v1 configs and the pre-specified inference helpers (no data)."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
import taxonomy as tx

CFG = ROOT / "configs" / "protocol_v1"


def test_configs_reference_frozen_files_and_share_one_design():
    common = json.loads((CFG / "common.json").read_text())
    for key, sha in (("manifest", "manifest_sha256"), ("pilot", "pilot_sha256")):
        assert hashlib.sha256((ROOT / common["cv"][key]).read_bytes()).hexdigest() == common["cv"][sha]
    for path, sha in common["data"]["inputs_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
    models = {p.stem: json.loads(p.read_text()) for p in (CFG / "models").glob("*.json")}
    assert set(models) == {"b0", "b1", "b2", "b2_adj", "b2_fcsurr", "b2_fc", "pc1_fc_only"}
    assert len({(m["cv"], m["learners"], m["window"], m["training_weights"]) for m in models.values()}) == 1
    fc = json.loads((CFG / "fc_wpli.json").read_text())
    assert fc["features"]["primary_dim"] == 171 * len(fc["primary_bands"])
    assert common["learners"]["secondary"]["grid"]["C"] == pytest.approx(list(np.logspace(-3, 2, 8)))


@pytest.mark.parametrize("ci95,ci90,label", [
    ((0.031, 0.08), (0.035, 0.07), "meaningfully superior"),
    ((0.005, 0.035), (0.008, 0.029), "positive, below SESOI"),
    ((0.005, 0.05), (0.008, 0.045), "positive, meaningful magnitude not established"),
    ((-0.035, -0.005), (-0.029, -0.008), "negative, within SESOI"),
    ((-0.06, -0.01), (-0.05, -0.015), "inferior"),
    ((-0.02, 0.02), (-0.015, 0.015), "equivalent / bounded null"),
    ((-0.04, 0.02), (-0.035, 0.015), "inconclusive"),
])
def test_taxonomy_rules(ci95, ci90, label):
    assert tx.classify(ci95, ci90) == label


def test_taxonomy_gate_and_exclusivity():
    assert tx.classify((-0.02, 0.02), (-0.015, 0.015), gates_pass=False) == tx.GATE_FAILED
    rng = np.random.default_rng(0)
    for _ in range(5000):  # nested intervals around a common centre: exactly one label each time
        c, h = rng.uniform(-0.08, 0.08), rng.uniform(0.001, 0.05)
        assert tx.classify((c - h, c + h), (c - 0.84 * h, c + 0.84 * h)) in tx.OUTCOMES


def test_nadeau_bengio_and_bootstrap_intervals():
    fm = np.array([0.01, 0.02, 0.0, 0.015, 0.005] * 10)
    lo, hi = tx.nadeau_bengio_ci(0.01, fm, 36, 0.90)
    sigma = np.sqrt((1 / 50 + 0.25) * fm.var(ddof=1))
    assert hi - lo == pytest.approx(2 * 1.6896 * sigma, rel=1e-3)
    blo, bhi = tx.cluster_bootstrap_ci(np.linspace(-0.02, 0.04, 36), 0.90)
    assert blo < 0.01 < bhi and tx.cluster_bootstrap_ci(np.linspace(-0.02, 0.04, 36), 0.90) == (blo, bhi)
