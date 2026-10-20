"""EXP-015: RQ-2 epoch-level-split inflation (06 v1.1 §2 RQ-2, §6 item 3; RDR-015). Descriptive; B2 only; never
touches the FC contrast or the RESULT-001 label.

Two arms on the same 69 primary recordings, kept epochs, features, z*, HGB grid, R9 weights and repeat seeds:
  epoch    5-fold stratified K-fold on the pooled epochs (patient identity ignored), inner tuning by epoch-level
           5-fold inside each outer training set; every epoch gets one out-of-fold score per repeat.
  patient  the EXP-004 B2 scores on the frozen patient-grouped manifest (no refit; data/processed/exp004/scores).
Per repeat and arm: (a) pooled AUROC over all held-out epochs of the 69 recordings; (b) the within-patient same-state
AUROC of each of the 35 evaluable patients (06 R2 on the out-of-fold scores). Inflation = epoch - patient, averaged
over repeats; repeat range and a cluster bootstrap (recordings for (a), evaluable patients for (b)).

Commands (from the repository root, on the Mac; epoch scores never leave it):
    python src/pipeline/rq2_epoch_split.py run --workers 6
    python src/pipeline/rq2_epoch_split.py summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features", "design", "pipeline"):
    sys.path.insert(0, str(ROOT / "src" / p))
import confirmatory as cf  # noqa: E402
import estimand as est  # noqa: E402
import exp004_descriptive as dx  # noqa: E402
import inner_folds as inf  # noqa: E402
import run_ladder as rl  # noqa: E402

CONFIG = ROOT / "experiments/EXP-015-rq2-epoch-split/config.json"
FROZEN_SHA256 = None  # set when Josue approves RDR-015; run refuses until then
MODEL, KIND, N_FOLDS = "b2", "hgb", 5


def load_config() -> tuple[dict, str]:
    raw = CONFIG.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check_frozen(sha: str) -> None:
    if FROZEN_SHA256 is None or sha != FROZEN_SHA256:
        raise SystemExit(f"EXP-015 config is not the frozen one ({sha} vs {FROZEN_SHA256})")


def epoch_index(pats: dict) -> list[tuple[str, int]]:
    """(eeg_id, epoch) for every kept epoch, recordings sorted: the pooled order the epoch splits refer to."""
    return [(i, e) for i in sorted(pats) for e in range(len(pats[i]["y"]))]


def epoch_folds(y: np.ndarray, seed: int) -> np.ndarray:
    """Label-stratified 5-fold assignment of epochs: within each class a seeded permutation dealt round-robin
    (numpy default_rng, which takes the 10- and 11-digit manifest and inner seeds as they are)."""
    rng = np.random.default_rng(seed)
    f = np.empty(y.size, int)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        f[idx] = np.arange(idx.size) % N_FOLDS
    return f


def subset(p: dict, mask: np.ndarray) -> dict | None:
    """The patient restricted to some of its epochs; evaluable states recounted on them (>= 5 and >= 5)."""
    if not mask.any():
        return None
    y, s = p["y"][mask], p["s"][mask]
    return {"id": p["id"], "y": y, "s": s, "evaluable": dx.evaluable_states(y, s, 5),
            "F": {k: v[mask] for k, v in p["F"].items()}}


def oof_scores(train: dict, masks: dict, folds: dict, params: dict, seed: int, n: int) -> dict:
    """Out-of-fold scores per recording over the given epochs: fold j scored by the model fitted on the rest."""
    out = {i: np.full(int(m.sum()), np.nan) for i, m in masks.items()}
    for j in range(n):
        tr = [q for i in sorted(masks) if (q := subset(train[i], folds[i] != j)) is not None]
        m = cf.L.fit(tr, MODEL, KIND, params, seed)
        for i in sorted(masks):
            te = folds[i] == j
            if te.any():
                out[i][te] = cf.L.score(m, subset(train[i], te), MODEL)
    return out


def select(train: dict, seed: int, lseed: int) -> dict:
    """Nested inner CV with epoch-level inner folds; criterion as EXP-004 (mean within-patient same-state AUROC,
    ties -> simpler) on the inner out-of-fold scores of the patients evaluable in this training set."""
    ids = sorted(train)
    y = np.concatenate([train[i]["y"] for i in ids])
    f = epoch_folds(y, seed)
    cuts = np.cumsum([0] + [len(train[i]["y"]) for i in ids])
    folds = {i: f[cuts[n]:cuts[n + 1]] for n, i in enumerate(ids)}
    masks = {i: np.ones(len(train[i]["y"]), bool) for i in ids}
    best = None
    for c in cf.L.HGB_GRID:
        sc = oof_scores(train, masks, folds, c, lseed, N_FOLDS)
        vals = [est.patient_auroc(sc[i], train[i]["y"], train[i]["s"], train[i]["evaluable"])
                for i in ids if train[i]["evaluable"]]
        key = (-float(np.mean(vals)), cf.L.simplicity_key(KIND, c))
        if best is None or key < best[0]:
            best = (key, c)
    return best[1]


def run_unit(args) -> dict:
    cache, scores_dir, rep, seed, k, recs, z_star = args
    t0 = time.time()
    ids = sorted(i for i in recs)
    pats = {i: cf.patient(cache, i, cf.blocks_for(MODEL), recs[i]["evaluable_states"], z_star) for i in ids}
    y = np.concatenate([pats[i]["y"] for i in ids])
    f = epoch_folds(y, seed)
    cuts = np.cumsum([0] + [len(pats[i]["y"]) for i in ids])
    fold = {i: f[cuts[n]:cuts[n + 1]] for n, i in enumerate(ids)}
    train = {i: q for i in ids if (q := subset(pats[i], fold[i] != k)) is not None}
    lseed = seed + k
    cfg = select(train, inf.inner_seed(seed, k), lseed)
    m = cf.L.fit([train[i] for i in sorted(train)], MODEL, KIND, cfg, lseed)
    sc = {i: cf.L.score(m, subset(pats[i], fold[i] == k), MODEL) for i in ids if (fold[i] == k).any()}
    idx = {i: np.flatnonzero(fold[i] == k) for i in sc}
    scores_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_dir / f"epoch_{MODEL}_r{rep}_k{k}.npz",
                        **{f"{i}__score": v for i, v in sc.items()}, **{f"{i}__index": v for i, v in idx.items()})
    return {"arm": "epoch", "model": MODEL, "learner": KIND, "repeat": rep, "seed": seed, "outer_fold": k,
            "config": cfg, "n_test_epochs": int(sum(v.size for v in sc.values())), "seconds": round(time.time() - t0, 1)}


def run(cache: Path, out: Path, scores_dir: Path, z_star: float, workers: int) -> None:
    d = cf.design()
    recs = {i: d["recs"][i] for i in next(iter(d["splits"].values()))}
    path = out / "units" / f"{KIND}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {(r["repeat"], r["outer_fold"]) for r in map(json.loads, filter(str.strip, path.read_text().splitlines()))}
    todo = [(cache, scores_dir, rep, seed, k, recs, z_star) for (rep, seed) in d["splits"] for k in range(N_FOLDS)
            if (rep, k) not in done]
    print(f"{len(done)} units done, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex, path.open("a") as f:
        for res in ex.map(run_unit, todo):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(res["repeat"], res["outer_fold"], res["seconds"], flush=True)


def epoch_arm(scores_dir: Path, rep: int, n_epochs: dict) -> dict:
    sc = {i: np.full(n, np.nan) for i, n in n_epochs.items()}
    for k in range(N_FOLDS):
        with np.load(scores_dir / f"epoch_{MODEL}_r{rep}_k{k}.npz") as z:
            for i in sc:
                if f"{i}__score" in z.files:
                    sc[i][z[f"{i}__index"]] = z[f"{i}__score"]
    return sc


def patient_arm(scores_dir: Path, rep: int, folds: dict) -> dict:
    sc = {}
    for k in range(N_FOLDS):
        with np.load(scores_dir / f"{KIND}_{MODEL}_r{rep}_k{k}.npz") as z:
            sc.update({i: z[i] for i in folds if folds[i] == k})
    return sc


def pooled(sc: dict, labels: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, labels and recording position of every epoch, recordings sorted."""
    ids = sorted(sc)
    return (np.concatenate([sc[i] for i in ids]), np.concatenate([labels[i] for i in ids]),
            np.concatenate([np.full(labels[i].size, n) for n, i in enumerate(ids)]))


def pooled_auroc(s: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Pooled AUROC; integer weights w (bootstrap multiplicities) equal duplicating epochs."""
    return float(roc_auc_score(y, s, sample_weight=w))


def summary(c: dict, sha: str) -> dict:
    d = cf.design()
    ids = sorted(next(iter(d["splits"].values())))
    lab, st = {}, {}
    for i in ids:
        with np.load(ROOT / c["cache"] / f"{i}.npz") as z:
            lab[i], st[i] = z["label"].astype(int), z["state"]
    ev = {i: d["recs"][i]["evaluable_states"] for i in ids if d["recs"][i]["evaluable_states"]}
    units = [json.loads(x) for x in (ROOT / c["out"] / "units" / f"{KIND}.jsonl").read_text().splitlines() if x]
    if len({(u["repeat"], u["outer_fold"]) for u in units}) != len(d["splits"]) * N_FOLDS:
        raise SystemExit(f"EXP-015 incomplete: {len(units)} units")
    ref = {}
    for x in (ROOT / c["reference_units"]).read_text().splitlines():
        r = json.loads(x)
        if r["model"] == MODEL:
            ref.update({(r["repeat"], i): a for i, a in r["auroc"].items()})
    arms = {"epoch": {"pooled": [], "within": []}, "patient": {"pooled": [], "within": []}}
    scores = {"epoch": [], "patient": []}
    mismatch = 0.0
    for (rep, _), folds in d["splits"].items():
        for arm, sc in (("epoch", epoch_arm(ROOT / c["scores"], rep, {i: lab[i].size for i in ids})),
                        ("patient", patient_arm(ROOT / c["reference_scores"], rep, folds))):
            if any(sc[i].size != lab[i].size or np.isnan(sc[i]).any() for i in ids):
                raise SystemExit(f"{arm} arm repeat {rep}: scores do not cover every kept epoch")
            scores[arm].append(pooled(sc, lab))
            arms[arm]["pooled"].append(pooled_auroc(*scores[arm][-1][:2]))
            w = {i: est.patient_auroc(sc[i], lab[i], st[i], ev[i]) for i in ev}
            arms[arm]["within"].append(w)
            if arm == "patient":  # the reused scores must reproduce the EXP-004 units exactly
                mismatch = max(mismatch, max(abs(w[i] - ref[(rep, i)]) for i in ev))
    if mismatch > 1e-9:
        raise SystemExit(f"EXP-004 B2 scores do not reproduce the EXP-004 units (max diff {mismatch})")
    res = {"experiment": "EXP-015", "learner": KIND, "model": MODEL, "config_sha256": sha,
           "status": "descriptive (RDR-015); B2 only; does not involve the FC contrast or RESULT-001",
           "n_recordings": len(ids), "n_epochs": int(sum(v.size for v in lab.values())), "n_evaluable": len(ev),
           "exp004_scores_reproduce_units": True}
    rng = np.random.default_rng(c["seed"])
    # (a) pooled AUROC; recording-cluster bootstrap of the repeat-mean inflation
    pa = {a: np.array(arms[a]["pooled"]) for a in arms}
    boot = []
    for _ in range(c["n_bootstrap"]):
        n_draw = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        boot.append(np.mean([pooled_auroc(es, ey, n_draw[er]) - pooled_auroc(ps, py, n_draw[pr])
                             for (es, ey, er), (ps, py, pr) in zip(scores["epoch"], scores["patient"])]))
    res["pooled_auroc"] = {"epoch_split": float(pa["epoch"].mean()), "patient_grouped": float(pa["patient"].mean()),
                           "inflation": float((pa["epoch"] - pa["patient"]).mean()),
                           "inflation_repeat_range": [float((pa["epoch"] - pa["patient"]).min()),
                                                      float((pa["epoch"] - pa["patient"]).max())],
                           "bootstrap95_recordings": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}
    # (b) within-patient same-state AUROC (R2) over the evaluable patients; patient-cluster bootstrap
    wp = {a: {i: float(np.mean([r[i] for r in arms[a]["within"]])) for i in ev} for a in arms}
    dp = [wp["epoch"][i] - wp["patient"][i] for i in sorted(ev)]
    per_rep = [np.mean([e[i] - p[i] for i in ev]) for e, p in zip(arms["epoch"]["within"], arms["patient"]["within"])]
    res["within_patient_auroc"] = {"epoch_split": float(np.mean(list(wp["epoch"].values()))),
                                   "patient_grouped": float(np.mean(list(wp["patient"].values()))),
                                   "inflation": float(np.mean(dp)),
                                   "inflation_repeat_range": [float(min(per_rep)), float(max(per_rep))],
                                   "bootstrap95_patients": list(cf.tx.cluster_bootstrap_ci(dp, 0.95, c["n_bootstrap"],
                                                                                           c["seed"])),
                                   "patients_inflated": int(sum(x > 0 for x in dp))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    c, sha = load_config()
    check_frozen(sha)
    out = ROOT / c["out"]
    if a.command == "run":
        if (out / "summary.json").exists():
            raise SystemExit("EXP-015 runs once: a summary already exists")
        chk = json.loads((ROOT / c["reference_cache_check"]).read_text())
        if chk["result"] != "PASS":
            raise SystemExit("the EXP-004 cache check must have passed")
        z_star = json.loads((ROOT / c["z_star"]["file"]).read_text())["z_star"]
        if z_star != c["z_star"]["value"]:
            raise SystemExit("pilot z* differs from the frozen value")
        run(ROOT / c["cache"], out, ROOT / c["scores"], z_star, a.workers)
    else:
        res = summary(c, sha)
        res["provenance"] = rl.provenance()
        (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
