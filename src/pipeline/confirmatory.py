"""EXP-004 confirmatory ladder on the frozen 5 x 10 manifest (06 issues 2, 3, 10; §4.1). Called only from
run_ladder.py after the unblinding gate passes.

Unit of work: (model, repeat, outer fold). For each unit: the outer training set is the primary-set recordings
outside the test fold; inner folds come from src/design/inner_folds.derive on the post-exclusion recount (R3);
the configuration is chosen by nested inner CV on the locked grid; the refit model scores every test recording.
One JSON line per unit (chosen configuration, AUROC_p for each evaluable test patient, timing) goes to
<out>/units/<learner>.jsonl; epoch scores for every test recording go to <scores>/ (Mac only, never committed).
Resumable: finished units are skipped.

Summary: per-repeat AUROCs give Delta_{p,r}, Delta_p, Delta-hat and the 50 fold-repeat means; the primary
interval named in the gate file labels the primary contrast (B2 + FC vs B2) with the taxonomy; PC-1 uses the
same interval on AUROC_p - 0.5; secondary contrasts are reported without multiplicity control (R10).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features", "design"):
    sys.path.insert(0, str(ROOT / "src" / p))
import b2 as fb2  # noqa: E402
import estimand as est  # noqa: E402
import f3_pilot_and_folds as f3  # noqa: E402
import inner_folds as inf  # noqa: E402
import node_strength as fns  # noqa: E402
import ladder as L  # noqa: E402
import taxonomy as tx  # noqa: E402

QC_CSV = ROOT / "artifacts/dataset_audit/channel_qc.csv"
STATE_MATRIX = ROOT / "artifacts/dataset_audit/subject_state_matrix.csv"
RECOUNT_CSV = ROOT / "artifacts/dataset_audit/evaluable_recount.csv"
MANIFEST = ROOT / "artifacts/protocol/f3/fold_manifest.csv"
PRIMARY = ("b2_fc", "b2")
SECONDARY = [("b2_fc", "b2_adj"), ("b2_fc", "b2_fcsurr"), ("b2_fcsurr", "b2"), ("b2_adj", "b2"), ("b2", "b1"),
             ("b1", "b0")]
BASE_KEYS = ("label", "state", "peak_z")


def design(manifest: Path = MANIFEST, sets: tuple = ("primary",)) -> dict:
    """Post-recount recording table and the outer splits of the given manifest analysis sets (primary; the
    issue-12 sensitivity run adds "sensitivity_only"): {(repeat, seed): {eeg_id: fold}}."""
    recs = f3.load_recordings(QC_CSV, STATE_MATRIX)
    inf.apply_recount(recs, RECOUNT_CSV)
    splits = defaultdict(dict)
    for r in csv.DictReader(manifest.open()):
        if r["analysis_set"] in sets:
            splits[(int(r["repeat"]), int(r["seed"]))][r["eeg_id"]] = int(r["outer_fold"])
    return {"recs": recs, "splits": dict(sorted(splits.items()))}


_CACHE: dict = {}


def patient(cache: Path, eeg_id: str, blocks: tuple, evaluable: list, z_star: float) -> dict:
    key = (str(cache), eeg_id, blocks)
    if key not in _CACHE:
        need = set(blocks) | set(BASE_KEYS) | ({"fc"} if "fc_ns" in blocks else set())
        with np.load(cache / f"{eeg_id}.npz") as z:
            F = {k: z[k] for k in need if k in z.files}
        if "fc_ns" in blocks and "fc_ns" not in F:  # EXP-017: derived from the cached edges, not stored
            F["fc_ns"] = fns.node_strength(F["fc"])
        F["b2_count"] = fb2.salience_count(F["peak_z"], z_star)[:, None]
        _CACHE[key] = {"id": eeg_id, "y": F["label"], "s": F["state"], "evaluable": evaluable, "F": F}
    return _CACHE[key]


def blocks_for(model: str) -> tuple:
    return tuple(sorted({"b1" if b == "b1_channel" else b for b in L.MODELS[model]} - {"b2_count"}))


def run_unit(args) -> dict:
    cache, scores_dir, model, kind, rep, seed, k, folds, recs, z_star = args
    t0 = time.time()
    blocks = blocks_for(model)
    ids = sorted(folds)
    pats = {i: patient(cache, i, blocks, recs[i]["evaluable_states"], z_star) for i in ids}
    test_ids = [i for i in ids if folds[i] == k]
    train_ids = [i for i in ids if folds[i] != k]
    inner, n_inner, bad = inf.derive(train_ids, recs, seed, k, test_ids)
    if bad:
        raise RuntimeError(f"repeat {rep} fold {k}: inner-fold validity rule failed {bad} (protocol deviation)")
    train = [pats[i] for i in train_ids]
    lseed = seed + k  # 06 §4.1: learner random_state = repeat seed + outer fold
    cfg = L.select(train, inner, model, kind, lseed)
    m = L.fit(train, model, kind, cfg, lseed)
    auroc, sc = {}, {}
    for i in test_ids:
        p = pats[i]
        sc[i] = L.score(m, p, model)
        if p["evaluable"]:
            auroc[i] = est.patient_auroc(sc[i], p["y"], p["s"], p["evaluable"])
    scores_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_dir / f"{kind}_{model}_r{rep}_k{k}.npz", **sc)
    return {"model": model, "learner": kind, "repeat": rep, "seed": seed, "outer_fold": k, "n_inner": n_inner,
            "config": cfg, "auroc": auroc, "seconds": round(time.time() - t0, 1)}


def run(cache: Path, out: Path, scores_dir: Path, models: list, kind: str, z_star: float, workers: int,
        sets: tuple = ("primary",), recs_override: dict | None = None) -> None:
    """recs_override (excluded-epochs sensitivity only): per recording, group / coverage / evaluable states
    recounted on that analysis's epoch set, replacing the F-6 recount (R3 applied to the sensitivity set)."""
    d = design(sets=sets)
    for i, v in (recs_override or {}).items():
        d["recs"][i].update(v)
    path = out / "units" / f"{kind}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["model"], r["repeat"], r["outer_fold"]))
    todo = [(cache, scores_dir, m, kind, rep, seed, k, folds, d["recs"], z_star)
            for m in models for (rep, seed), folds in d["splits"].items() for k in range(f3.N_FOLDS)
            if (m, rep, k) not in done]
    print(f"{len(done)} units done, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex, path.open("a") as f:
        for res in ex.map(run_unit, todo):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(res["model"], res["repeat"], res["outer_fold"], res["seconds"], flush=True)


def intervals(d: dict, method: str) -> dict:
    """90% and 95% intervals for a delta summary with the named primary method ("nb" or "m1")."""
    dp = list(d["delta_p"].values())
    out = {}
    for lvl in (0.90, 0.95):
        if method == "m1":
            ci = tx.m1_ci(d["delta_hat"], d["fold_means"], dp, lvl)
        elif method == "nb":
            ci = tx.nadeau_bengio_ci(d["delta_hat"], d["fold_means"], d["n_e"], lvl)
        else:
            raise ValueError(method)
        out[f"{lvl:.2f}"] = [float(ci[0]), float(ci[1])]
        out[f"bootstrap_{lvl:.2f}"] = [float(x) for x in tx.cluster_bootstrap_ci(dp, lvl)]
    return out


def heterogeneity(d: dict) -> dict:
    """06 R6: SD of Delta_p, 95% prediction interval for a new patient, range, counts beyond +/- SESOI."""
    from scipy import stats

    dp = np.array(list(d["delta_p"].values()))
    n, s = dp.size, float(dp.std(ddof=1))
    h = stats.t.ppf(0.975, n - 1) * s * np.sqrt(1 + 1 / n)
    return {"sd_delta_p": s, "prediction_interval_95": [float(dp.mean() - h), float(dp.mean() + h)],
            "range": [float(dp.min()), float(dp.max())], "n_above_sesoi": int((dp > tx.SESOI).sum()),
            "n_below_minus_sesoi": int((dp < -tx.SESOI).sum())}


def summarise(out: Path, kind: str, method: str, pc3_pass: bool) -> dict:
    d0 = design()
    folds = {(rep, i): k for (rep, _), fs in d0["splits"].items() for i, k in fs.items()}
    auroc = {}
    for line in (out / "units" / f"{kind}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            for i, a in r["auroc"].items():
                auroc[(r["model"], r["repeat"], i)] = a
    models = {m for m, _, _ in auroc}
    for (m, r, i) in list(auroc):
        if m == "pc1_fc_only":
            auroc[("chance", r, i)] = 0.5
    res = {"learner": kind, "primary_interval": method, "contrasts": {}}
    if "pc1_fc_only" in models:
        pc1 = est.delta_summary(auroc, "pc1_fc_only", "chance", folds)
        ci = intervals(pc1, method)
        res["pc1"] = {"mean_auroc": 0.5 + pc1["delta_hat"], "lower95": 0.5 + ci["0.95"][0],
                      "pass": bool(ci["0.95"][0] > 0)}
    gates = bool(pc3_pass and res.get("pc1", {}).get("pass", False))
    for a, b in [PRIMARY] + SECONDARY:
        if a in models and b in models:
            d = est.delta_summary(auroc, a, b, folds)
            ci = intervals(d, method)
            lab = tx.classify(tuple(ci["0.95"]), tuple(ci["0.90"]), gates_pass=gates)
            res["contrasts"][f"{a}_vs_{b}"] = {
                "role": "PRIMARY (confirmatory)" if (a, b) == PRIMARY else "secondary, not multiplicity-controlled",
                "delta_hat": d["delta_hat"], "n_e": d["n_e"], "n_cells": d["n_cells"], "intervals": ci,
                "label": lab,
                "label_sensitivity_bootstrap": tx.classify(tuple(ci["bootstrap_0.95"]), tuple(ci["bootstrap_0.90"]),
                                                           gates_pass=gates),
                "heterogeneity": heterogeneity(d)}
    res["gates"] = {"pc3": pc3_pass, "pc1": res.get("pc1", {}).get("pass"), "both_pass": gates}
    return res
