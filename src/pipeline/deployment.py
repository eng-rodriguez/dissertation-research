"""EXP-009: key secondary deployment metrics for EXP-004 (06 issue 2), run plan A (approved by Josue 2026-10-17).

For each EXP-004 outer unit of B2 and B2 + FC (both learners), with the configuration EXP-004 selected (read from
artifacts/exp004/units/{learner}.jsonl, so no re-tuning):
  1. refit the outer model with the same seed and check its held-out scores equal the saved EXP-004 scores
     (aborts otherwise: the deployment metrics must describe the same models);
  2. out-of-fold training scores: refit on each inner-training set (the EXP-004 inner folds) and score the inner
     validation recordings;
  3. thresholds: per target false-positive rate T in {0.5, 1, 2} per minute, the threshold whose out-of-fold
     false-positive rate over the training fold's IED-free recordings is at most T (one threshold for both states);
  4. Platt calibration: unweighted logistic regression of the training out-of-fold labels on their scores.
Held-out outputs per unit: per E patient, sensitivity over IED epochs of its evaluable states; per IED-free
recording, false positives per minute (all epochs and per state); per recording, calibrated log-loss.

  python src/pipeline/deployment.py run --learner hgb --workers 6
  python src/pipeline/deployment.py summary --learner hgb

Descriptive (06 §4.4): estimates with M1 intervals; Holm-adjusted only if any is described as a test. They never
relabel RESULT-001. Calibrated probabilities are kept on the Mac (data/processed/exp009/), never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features", "design", "pipeline"):
    sys.path.insert(0, str(ROOT / "src" / p))
import confirmatory as cf  # noqa: E402
import estimand as est  # noqa: E402
import inner_folds as inf  # noqa: E402
import ladder as L  # noqa: E402
import taxonomy as tx  # noqa: E402

CONFIG = ROOT / "experiments/EXP-009-deployment/config.json"
OUT = ROOT / "artifacts/exp009"
EPS = 1e-12


def threshold_for(neg_scores: np.ndarray, fp_per_min: float, epoch_s: float) -> float:
    """Threshold t (score >= t is a detection) allowing at most fp_per_min false positives per minute of analysed
    EEG on these negatives: just above the (allowed + 1)-th largest score, so ties never add detections."""
    s = np.sort(np.asarray(neg_scores, float))[::-1]
    allowed = int(np.floor(fp_per_min * len(s) * epoch_s / 60.0))
    return -np.inf if allowed >= len(s) else float(np.nextafter(s[allowed], np.inf))


def platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(scores, float)[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def calibrated(scores: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(ab[0] * np.asarray(scores, float) + ab[1])))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run_unit(args) -> dict:
    cache, saved_dir, prob_dir, model, kind, rep, seed, k, folds, recs, z_star, cfg, c = args
    t0 = time.time()
    blocks = cf.blocks_for(model)
    ids = sorted(folds)
    pats = {i: cf.patient(cache, i, blocks, recs[i]["evaluable_states"], z_star) for i in ids}
    test_ids = [i for i in ids if folds[i] == k]
    train_ids = [i for i in ids if folds[i] != k]
    lseed = seed + k
    m = L.fit([pats[i] for i in train_ids], model, kind, cfg, lseed)
    test_sc = {i: L.score(m, pats[i], model) for i in test_ids}
    with np.load(saved_dir / f"{kind}_{model}_r{rep}_k{k}.npz") as z:
        diff = max(float(np.max(np.abs(test_sc[i] - z[i]))) for i in test_ids)
    if diff > 1e-9:
        raise RuntimeError(f"{kind} {model} r{rep} k{k}: refit differs from the EXP-004 scores by {diff}")
    inner, _, bad = inf.derive(train_ids, recs, seed, k, test_ids)
    if bad:
        raise RuntimeError(f"repeat {rep} fold {k}: inner-fold validity rule failed {bad}")
    oof = {}
    for j in sorted(set(inner.values())):
        mj = L.fit([pats[i] for i in train_ids if inner[i] != j], model, kind, cfg, lseed)
        for i in train_ids:
            if inner[i] == j:
                oof[i] = L.score(mj, pats[i], model)
    free_tr = [i for i in train_ids if recs[i]["group"] == "F"]
    neg = np.concatenate([oof[i] for i in free_tr])
    thr = {str(t): threshold_for(neg, t, c["epoch_seconds"]) for t in c["fp_per_min_targets"]}
    ab = platt(np.concatenate([oof[i] for i in train_ids]), np.concatenate([pats[i]["y"] for i in train_ids]))
    sens, fp, ll = defaultdict(dict), defaultdict(dict), {}
    probs = {}
    for i in test_ids:
        p, sc = pats[i], test_sc[i]
        probs[i] = calibrated(sc, ab)
        ll[i] = {"log_loss": log_loss(probs[i], p["y"]), "n": int(len(p["y"]))}
        if p["evaluable"]:
            ev = np.isin(p["s"], p["evaluable"])
            ll[i]["log_loss_evaluable"] = log_loss(probs[i][ev], p["y"][ev])
        for t, th in thr.items():
            if p["evaluable"]:
                ied = (p["y"] == 1) & np.isin(p["s"], p["evaluable"])
                sens[t][i] = float((sc[ied] >= th).mean())
            if recs[i]["group"] == "F":
                row = {"all": [int((sc >= th).sum()), len(sc) * c["epoch_seconds"] / 60.0]}
                for st in ("wake", "sleep"):
                    w = p["s"] == st
                    row[st] = [int((sc[w] >= th).sum()), int(w.sum()) * c["epoch_seconds"] / 60.0]
                fp[t][i] = row
    prob_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prob_dir / f"{kind}_{model}_r{rep}_k{k}.npz", **probs)
    return {"model": model, "learner": kind, "repeat": rep, "seed": seed, "outer_fold": k, "config": cfg,
            "refit_max_abs_diff": diff, "thresholds": thr, "platt": list(ab), "n_free_train": len(free_tr),
            "sensitivity": sens, "fp": fp, "log_loss": ll, "seconds": round(time.time() - t0, 1)}


def load_config() -> tuple[dict, str]:
    raw = CONFIG.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def run(cache: Path, saved_dir: Path, prob_dir: Path, kind: str, z_star: float, workers: int,
        out: Path = OUT, exp004_units: Path = ROOT / "artifacts/exp004/units") -> None:
    c, _ = load_config()
    d = cf.design()
    cfgs = {}
    for line in (exp004_units / f"{kind}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cfgs[(r["model"], r["repeat"], r["outer_fold"])] = r["config"]
    path = out / "units" / f"{kind}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["model"], r["repeat"], r["outer_fold"]))
    todo = [(cache, saved_dir, prob_dir, m, kind, rep, seed, k, folds, d["recs"], z_star, cfgs[(m, rep, k)], c)
            for m in c["models"] for (rep, seed), folds in d["splits"].items() for k in range(cf.f3.N_FOLDS)
            if (m, rep, k) not in done]
    print(f"{len(done)} units done, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex, path.open("a") as f:
        for res in ex.map(run_unit, todo):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(res["model"], res["repeat"], res["outer_fold"], res["seconds"], flush=True)


def _contrast(per: dict, folds: dict, a: str, b: str) -> dict:
    d = est.delta_summary(per, a, b, folds)
    return {"delta_hat": d["delta_hat"], "n": d["n_e"], "intervals": cf.intervals(d, tx.PRIMARY_INTERVAL),
            "mean_ref": float(np.mean([per[k] for k in per if k[0] == b])),
            "mean_model": float(np.mean([per[k] for k in per if k[0] == a]))}


def summary(kind: str, out: Path = OUT, prob_dir: Path | None = None, cache: Path | None = None) -> dict:
    c, sha = load_config()
    a, b = c["models"][1], c["models"][0]  # b2_fc vs b2
    units = [json.loads(x) for x in (out / "units" / f"{kind}.jsonl").read_text().splitlines() if x.strip()]
    d = cf.design()
    folds = {(rep, i): k for (rep, _), fs in d["splits"].items() for i, k in fs.items()}
    res = {"experiment": "EXP-009", "learner": kind, "config_sha256": sha, "contrast": f"{a}_vs_{b}",
           "status": "descriptive key secondary metrics (06 issue 2); never relabel RESULT-001",
           "max_refit_diff": max(u["refit_max_abs_diff"] for u in units), "n_units": len(units),
           "sensitivity": {}, "fp_per_min": {}, "log_loss": {}}
    for t in map(str, c["fp_per_min_targets"]):
        per = {(u["model"], u["repeat"], i): v for u in units for i, v in u["sensitivity"][t].items()}
        res["sensitivity"][t] = _contrast(per, folds, a, b)
        fpm = {}
        for scope in ("all", "wake", "sleep"):
            rows = defaultdict(list)
            for u in units:
                for i, r in u["fp"][t].items():
                    if r[scope][1] > 0:
                        rows[u["model"]].append(r[scope][0] / r[scope][1])
            fpm[scope] = {m: float(np.mean(v)) for m, v in rows.items()}
        res["fp_per_min"][t] = fpm
    per = {(u["model"], u["repeat"], i): v["log_loss_evaluable"] for u in units for i, v in u["log_loss"].items()
           if "log_loss_evaluable" in v}
    res["log_loss"]["evaluable_patients"] = _contrast(per, folds, a, b)
    pooled = defaultdict(list)
    for u in units:
        for i, v in u["log_loss"].items():
            pooled[(u["model"], u["repeat"])].append((v["log_loss"] * v["n"], v["n"]))
    res["log_loss"]["all_recordings_pooled"] = {
        m: float(np.mean([sum(x for x, _ in v) / sum(n for _, n in v) for (mm, _), v in pooled.items() if mm == m]))
        for m in c["models"]}
    if prob_dir is not None:
        res["calibration"] = calibration_summary(units, prob_dir, kind, c, cache)
    return res


def calibration_summary(units: list[dict], prob_dir: Path, kind: str, c: dict, cache: Path) -> dict:
    """Per model: calibration slope and intercept of held-out epochs pooled within a repeat (mean over repeats),
    and a 10-bin reliability table pooled over all repeats."""
    from sklearn.linear_model import LogisticRegression

    labels = {}
    out = {}
    for m in c["models"]:
        by_rep = defaultdict(lambda: ([], []))
        for u in units:
            if u["model"] != m:
                continue
            with np.load(prob_dir / f"{kind}_{m}_r{u['repeat']}_k{u['outer_fold']}.npz") as z:
                for i in z.files:
                    if i not in labels:
                        with np.load(cache / f"{i}.npz") as zz:
                            labels[i] = zz["label"]
                    by_rep[u["repeat"]][0].append(z[i])
                    by_rep[u["repeat"]][1].append(labels[i])
        slopes, inters, allp, ally = [], [], [], []
        for p, y in by_rep.values():
            p, y = np.concatenate(p), np.concatenate(y)
            lg = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1))
            fit = LogisticRegression(C=1e6, max_iter=1000).fit(lg[:, None], y)
            slopes.append(float(fit.coef_[0, 0]))
            inters.append(float(fit.intercept_[0]))
            allp.append(p)
            ally.append(y)
        p, y = np.concatenate(allp), np.concatenate(ally)
        bins = np.clip((p * 10).astype(int), 0, 9)
        table = [{"bin": f"{j / 10:.1f}-{(j + 1) / 10:.1f}", "n": int((bins == j).sum()),
                  "mean_predicted": float(p[bins == j].mean()) if (bins == j).any() else None,
                  "observed_rate": float(y[bins == j].mean()) if (bins == j).any() else None} for j in range(10)]
        out[m] = {"slope_mean": float(np.mean(slopes)), "intercept_mean": float(np.mean(inters)), "reliability": table}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run", "summary"])
    ap.add_argument("--learner", default="hgb", choices=["hgb", "lr"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    c, _ = load_config()
    cache, saved = ROOT / c["cache"], ROOT / c["exp004_scores"]
    prob_dir = ROOT / c["calibrated_probs"]
    if a.command == "run":
        z_star = json.loads((ROOT / "artifacts/pilot/zstar.json").read_text())["z_star"]
        run(cache, saved, prob_dir, a.learner, z_star, a.workers)
    else:
        import run_ladder as rl

        res = summary(a.learner, prob_dir=prob_dir, cache=cache)
        res["provenance"] = rl.provenance()
        (OUT / f"summary_{a.learner}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("max_refit_diff", "sensitivity", "fp_per_min")}, indent=2))


if __name__ == "__main__":
    main()
