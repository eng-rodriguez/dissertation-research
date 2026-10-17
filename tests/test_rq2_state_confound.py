"""EXP-014 RQ-2 state-confound classifier on a synthetic cache with the real frozen design."""

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
import confirmatory as cf  # noqa: E402
import rq2_state_confound as rq  # noqa: E402


def fake_cache(path: Path, confounded: bool) -> None:
    rng = np.random.default_rng(0)
    d = cf.design()
    for i in next(iter(d["splits"].values())):
        pos = d["recs"][i]["group"] != "F"
        share = (0.8 if pos else 0.2) if confounded else 0.5
        s = np.where(rng.random(60) < share, "sleep", "wake")
        np.savez_compressed(path / f"{i}.npz", state=s, b1=rng.random((60, 19, 24)), label=np.zeros(60, int))


def test_frozen_config():
    c, sha = rq.load_config()
    assert sha == rq.FROZEN_SHA256
    assert c["protocol"]["sha256"] == hashlib.sha256((ROOT / c["protocol"]["document"]).read_bytes()).hexdigest()


def test_detects_a_planted_state_confound_and_not_a_null(tmp_path):
    c, _ = rq.load_config()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fake_cache(a, True)
    fake_cache(b, False)
    hi = rq.run(a, c)["models"]["state_label"]
    lo = rq.run(b, c)["models"]["state_label"]
    assert hi["auroc_mean_over_repeats"] > 0.9 and hi["bootstrap95"][0] > 0.75
    assert 0.25 < lo["auroc_mean_over_repeats"] < 0.75
