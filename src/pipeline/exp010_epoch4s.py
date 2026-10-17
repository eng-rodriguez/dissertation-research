"""EXP-010: native 4-s whole-epoch sensitivity analysis incl. exploratory delta FC (06 v1.1 issue 1 item 8, §4.3,
§5; run plan B). A pre-specified sensitivity analysis, not a rescue analysis; descriptive (06 §4.4); it never
assigns or revises the RESULT-001 label. Everything except the window (and the factors listed under
"factors_changed" in experiments/EXP-010-epoch4s/config.json) is the EXP-004 machinery, called unchanged.

Commands (from the repository root, on the Mac; raw data and epoch scores never leave it):
  cache-check  the 4-s cache must hold exactly the EXP-004 kept epochs (F-6 recount per class and state), every
               block in its locked shape (fc_delta: 171 edges) and finite B1/B2/ADJ. Writes
               artifacts/exp010/cache_check.json.
  run          refuses unless the config hash is the frozen one, the cache check passed and no summary exists
               (runs once); then src/pipeline/confirmatory.run with the 4-s cache, HGB, all eight models.
               Resumable.
  summary      Delta-hat, M1 and bootstrap intervals, heterogeneity (R6), the descriptive taxonomy reading and
               its concordance with RESULT-001 (rule fixed in the config), the delta rung, PC-1 on the 4-s
               window, per-state strata (bootstrap, no label; R10) from the saved scores, a forest plot and the
               per-patient table. Writes artifacts/exp010/summary_hgb.json, figures/ and tables/.

    python src/features/extract.py --mode epoch4s --allow-non-pilot --ids-from data/interim/primary_ids.csv \
        --root data/raw/vepiset/extracted/opensource-dataset --out data/interim/features/epoch4s
    python src/pipeline/exp010_epoch4s.py cache-check
    python src/pipeline/exp010_epoch4s.py run --workers 6
    python src/pipeline/exp010_epoch4s.py summary
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features", "design", "pipeline", "visualization"):
    sys.path.insert(0, str(ROOT / "src" / p))
import confirmatory as cf  # noqa: E402
import estimand as est  # noqa: E402
import run_ladder as rl  # noqa: E402
import taxonomy as tx  # noqa: E402

CONFIG = ROOT / "experiments/EXP-010-epoch4s/config.json"
FROZEN_SHA256 = "532000e25176ed82dff6af4064a0110656e7fb880039b68de6c12e03a1f14ddd"  # frozen 2026-10-17
SHAPES = {**rl.SHAPES, "fc_delta": (171,)}
KIND = "hgb"


def load_config() -> tuple[dict, str]:
    raw = CONFIG.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def paths(c: dict) -> tuple[Path, Path, Path]:
    return ROOT / c["cache"], ROOT / c["scores"], ROOT / c["out"]


def check_frozen(sha: str) -> None:
    if FROZEN_SHA256 is None or sha != FROZEN_SHA256:
        raise SystemExit(f"EXP-010 config is not the frozen one ({sha} vs {FROZEN_SHA256})")


def cmd_cache_check(a) -> None:
    c, _ = load_config()
    cache, _, out = paths(c)
    res = rl.cache_check(cache, shapes=SHAPES)
    res["cache"] = c["cache"]
    res["provenance"] = rl.provenance()
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache_check.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("cache_sha256", "provenance")}, indent=2))


def cmd_run(a) -> None:
    c, sha = load_config()
    check_frozen(sha)
    cache, scores, out = paths(c)
    chk = out / "cache_check.json"
    if not chk.exists() or json.loads(chk.read_text())["result"] != "PASS":
        raise SystemExit("EXP-010 needs a passing 4-s cache check first (exp010_epoch4s.py cache-check)")
    if (out / f"summary_{KIND}.json").exists():
        raise SystemExit("EXP-010 runs once: a summary already exists")
    z_star = json.loads((ROOT / c["unchanged_from_exp004"]["z_star"]["file"]).read_text())["z_star"]
    if z_star != c["unchanged_from_exp004"]["z_star"]["value"]:
        raise SystemExit("pilot z* differs from the frozen value")
    cf.run(cache, out, scores, c["models"], KIND, z_star, a.workers)


def contrast(auroc: dict, folds: dict, a: str, b: str, concordance: dict | None = None) -> dict:
    d = est.delta_summary(auroc, a, b, folds)
    ci = cf.intervals(d, tx.PRIMARY_INTERVAL)
    reading = tx.classify(tuple(ci["0.95"]), tuple(ci["0.90"]))
    res = {"delta_hat": d["delta_hat"], "n_e": d["n_e"], "n_cells": d["n_cells"], "intervals": ci,
           "descriptive_reading_m1": reading, "heterogeneity": cf.heterogeneity(d)}
    if concordance is not None:
        res["concordance_with_result001"] = concordance[reading]
    return res


def mean_auroc(auroc: dict, model: str) -> float:
    """Mean over evaluable patients of the patient's AUROC averaged over repeats (equal patient weight)."""
    by = defaultdict(list)
    for (m, _, p), v in auroc.items():
        if m == model:
            by[p].append(v)
    return float(np.mean([np.mean(v) for v in by.values()]))


def per_state(cache: Path, scores: Path, models: list[str]) -> dict:
    import exp004_descriptive as dx

    L = dx.load(cache, scores, KIND, models)
    base = {i: r["evaluable_states"] for i, r in L["recs"].items() if r.get("evaluable_states") and i in L["data"]}
    out = {}
    for st in dx.STATES:
        ev = {i: [st] for i, s in base.items() if st in s}
        out[st] = {f"{a}_vs_{b}": dx.contrast(L, a, b, ev, labelled=False)
                   for a, b in [("b2_fc", "b2"), ("b2_fc_delta", "b2_fc")]}
    return out


def summary(c: dict, sha: str, cache: Path | None = None, scores: Path | None = None) -> dict:
    import exp004_report as rep

    _, _, out = paths(c)
    units = rep.load_units(KIND, out)
    auroc, folds = rep.per_repeat(units)
    models = {m for m, _, _ in auroc}
    missing = set(c["models"]) - models
    if missing or len(units) != len(c["models"]) * 50:
        raise SystemExit(f"EXP-010 incomplete: {len(units)} units, missing models {sorted(missing)}")
    conc = c["reading"]["concordance"]
    a, b = c["main_contrast"]
    res = {"experiment": "EXP-010", "learner": KIND, "config_sha256": sha,
           "status": "pre-specified sensitivity analysis (06 issue 1 item 8); descriptive; never relabels RESULT-001",
           "reference": c["reading"]["reference"],
           "main": {f"{a}_vs_{b}": contrast(auroc, folds, a, b, conc)},
           "delta_band": {f"{x}_vs_{y}": contrast(auroc, folds, x, y) for x, y in c["delta_contrasts"]},
           "other": {f"{x}_vs_{y}": contrast(auroc, folds, x, y) for x, y in cf.SECONDARY},
           "mean_auroc": {m: mean_auroc(auroc, m) for m in sorted(models)}}
    for (m, r, p) in list(auroc):
        if m == "pc1_fc_only":
            auroc[("chance", r, p)] = 0.5
    pc1 = est.delta_summary(auroc, "pc1_fc_only", "chance", folds)
    ci = cf.intervals(pc1, tx.PRIMARY_INTERVAL)
    res["pc1_4s"] = {"mean_auroc": 0.5 + pc1["delta_hat"], "lower95_m1": 0.5 + ci["0.95"][0],
                     "above_chance": bool(ci["0.95"][0] > 0), "note": "reported, not a gate for this analysis"}
    ref = res["reference"]["delta_hat"]
    m = res["main"][f"{a}_vs_{b}"]
    res["comparison"] = {"delta_4s_minus_delta_2s": m["delta_hat"] - ref,
                         "same_sign": bool((m["delta_hat"] > 0) == (ref > 0)),
                         "concordance": m["concordance_with_result001"]}
    if cache is not None and scores is not None:
        res["per_state"] = per_state(cache, scores, ["b2", "b2_fc", "b2_fc_delta"])
        res["per_state"]["note"] = "estimate + 95% percentile bootstrap CI, no outcome label (R10); wake underpowered"
    return res


def write_figures(c: dict, res: dict) -> None:
    import exp004_report as rep

    _, _, out = paths(c)
    units = rep.load_units(KIND, out)
    counts = rep.ied_counts()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for (x, y), key, sect in [(tuple(c["main_contrast"]), "b2_fc_vs_b2", "main"),
                              (("b2_fc_delta", "b2_fc"), "b2_fc_delta_vs_b2_fc", "delta_band")]:
        dl = rep.patient_deltas(units, x, y)
        r = res[sect][key]
        title = f"EXP-010 (4-s epoch, sensitivity): {key}, HGB"
        svg = rep.forest_svg(dl, counts, r["delta_hat"], r["intervals"]["0.90"], r["intervals"]["0.95"], title,
                             axis=f"Δ within-patient AUROC ({x} − {y})")
        (out / "figures" / f"forest_{key}.svg").write_text(svg)
        with (out / "tables" / f"patient_delta_{key}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["eeg_id", "delta_p", "repeat_min", "repeat_max", "evaluable_ied_epochs"])
            for p, (d, lo, hi) in sorted(dl.items()):
                w.writerow([p, f"{d:.6f}", f"{lo:.6f}", f"{hi:.6f}", counts[p]])


def cmd_summary(a) -> None:
    c, sha = load_config()
    check_frozen(sha)
    cache, scores, out = paths(c)
    res = summary(c, sha, cache, scores)
    res["provenance"] = rl.provenance()
    (out / f"summary_{KIND}.json").write_text(json.dumps(res, indent=2) + "\n")
    write_figures(c, res)
    print(json.dumps({"main": res["main"], "delta_band": {k: {kk: v[kk] for kk in ("delta_hat", "intervals",
                                                                                     "descriptive_reading_m1")}
                                                          for k, v in res["delta_band"].items()},
                      "pc1_4s": res["pc1_4s"], "comparison": res["comparison"]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["cache-check", "run", "summary"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    {"cache-check": cmd_cache_check, "run": cmd_run, "summary": cmd_summary}[a.command](a)


if __name__ == "__main__":
    main()
