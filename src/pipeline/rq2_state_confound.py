"""EXP-014: RQ-2 state-confound quantification (06 v1.1 issue 9 item 7). A state-only classifier predicts patient-level
IED status from the recording's state composition (and EEG state proxies), patient-grouped on the frozen manifest.
Descriptive; reads only the EXP-004 cache (kept epochs, B1 relative delta and sigma power, state, label) and writes
aggregate numbers to artifacts/exp014/summary.json.

    python src/pipeline/rq2_state_confound.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features", "design", "pipeline"):
    sys.path.insert(0, str(ROOT / "src" / p))
import b1 as fb1  # noqa: E402
import confirmatory as cf  # noqa: E402
import run_ladder as rl  # noqa: E402

CONFIG = ROOT / "experiments/EXP-014-rq2-state-confound/config.json"
FROZEN_SHA256 = "9f07fd23575044dd7cf479b4b3390fab8c358d574004df5e4a9f724e26d4477f"  # frozen 2026-10-17
REL = [fb1.NAMES.index("rel_delta"), fb1.NAMES.index("rel_sigma")]
COLUMNS = {"state_label": [0], "state_label_plus_proxies": [0, 1, 2]}


def load_config() -> tuple[dict, str]:
    raw = CONFIG.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def recording_features(cache: Path, ids: list[str]) -> np.ndarray:
    """Per recording: [sleep share, mean rel delta, mean rel sigma] over its kept epochs (labels unused)."""
    rows = []
    for i in ids:
        with np.load(cache / f"{i}.npz") as z:
            s, b = z["state"], z["b1"]
        rows.append([float((s == "sleep").mean())] + [float(b[:, :, k].mean()) for k in REL])
    return np.array(rows)


def auroc_bootstrap(y: np.ndarray, p: np.ndarray, seed: int, n: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def run(cache: Path, c: dict) -> dict:
    d = cf.design()
    ids = sorted(next(iter(d["splits"].values())))
    y = np.array([int(d["recs"][i]["group"] != "F") for i in ids])
    X = recording_features(cache, ids)
    res = {"experiment": "EXP-014", "n_recordings": len(ids), "n_ied_positive": int(y.sum()),
           "group_means": {g: dict(zip(["sleep_share", "rel_delta", "rel_sigma"],
                                       X[np.array([d["recs"][i]["group"] == g for i in ids])].mean(axis=0).tolist()))
                           for g in ("E", "N", "F")},
           "models": {}}
    for name, cols in COLUMNS.items():
        per_rep, probs = [], np.zeros((len(d["splits"]), len(ids)))
        for r, ((rep, seed), folds) in enumerate(d["splits"].items()):
            f = np.array([folds[i] for i in ids])
            for k in sorted(set(f)):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
                m.fit(X[f != k][:, cols], y[f != k])
                probs[r, f == k] = m.predict_proba(X[f == k][:, cols])[:, 1]
            per_rep.append(float(roc_auc_score(y, probs[r])))
        avg = probs.mean(axis=0)
        res["models"][name] = {"auroc_mean_over_repeats": float(np.mean(per_rep)),
                               "auroc_repeat_range": [min(per_rep), max(per_rep)],
                               "auroc_repeat_averaged_probs": float(roc_auc_score(y, avg)),
                               "bootstrap95": auroc_bootstrap(y, avg, c["seed"])}
    return res


def main() -> None:
    c, sha = load_config()
    if FROZEN_SHA256 is None or sha != FROZEN_SHA256:
        raise SystemExit(f"EXP-014 config is not the frozen one ({sha})")
    res = run(ROOT / c["cache"], c)
    res["config_sha256"] = sha
    res["provenance"] = rl.provenance()
    out = ROOT / c["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
