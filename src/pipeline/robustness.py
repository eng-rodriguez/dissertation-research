"""Pre-specified robustness analyses after EXP-004 (06 v1.1 §5; run plan C onward), one frozen config each.

Each experiment re-runs the EXP-004 ladder (src/pipeline/confirmatory.run, unchanged) on its own feature cache and
reports the primary contrast descriptively (06 §4.4): M1 and bootstrap intervals, heterogeneity, the taxonomy as a
descriptive reading only, and concordance with RESULT-001 by the rule fixed in the config. Never a label for H-1.
The summary pieces are shared with EXP-010 (src/pipeline/exp010_epoch4s.py).

Commands (from the repository root, on the Mac; raw data and epoch scores never leave it):
    python src/pipeline/robustness.py cache-check --exp EXP-011
    python src/pipeline/robustness.py run --exp EXP-011 --workers 6
    python src/pipeline/robustness.py summary --exp EXP-011
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features", "design", "pipeline", "visualization"):
    sys.path.insert(0, str(ROOT / "src" / p))
import confirmatory as cf  # noqa: E402
import estimand as est  # noqa: E402
import exp010_epoch4s as e10  # noqa: E402
import run_ladder as rl  # noqa: E402
import taxonomy as tx  # noqa: E402

CONFIGS = {"EXP-011": ROOT / "experiments/EXP-011-bipolar/config.json"}
FROZEN_SHA256 = {"EXP-011": "800175f87391ead8684861beb6bf97dd2631e64f968de385fa5954f7b6000463"}  # frozen 2026-10-17
KIND = "hgb"


def load_config(exp: str) -> tuple[dict, str]:
    raw = CONFIGS[exp].read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check_frozen(exp: str, sha: str) -> None:
    if FROZEN_SHA256.get(exp) is None or sha != FROZEN_SHA256[exp]:
        raise SystemExit(f"{exp} config is not the frozen one ({sha} vs {FROZEN_SHA256.get(exp)})")


def shapes(c: dict) -> dict:
    return {k: tuple(v) for k, v in c["shapes"].items()}


def cmd_cache_check(a) -> None:
    c, _ = load_config(a.exp)
    out = ROOT / c["out"]
    res = rl.cache_check(ROOT / c["cache"], shapes=shapes(c))
    res["cache"] = c["cache"]
    res["provenance"] = rl.provenance()
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache_check.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("cache_sha256", "provenance")}, indent=2))


def cmd_run(a) -> None:
    c, sha = load_config(a.exp)
    check_frozen(a.exp, sha)
    out = ROOT / c["out"]
    chk = out / "cache_check.json"
    if not chk.exists() or json.loads(chk.read_text())["result"] != "PASS":
        raise SystemExit(f"{a.exp} needs a passing cache check first (robustness.py cache-check --exp {a.exp})")
    if (out / f"summary_{KIND}.json").exists():
        raise SystemExit(f"{a.exp} runs once: a summary already exists")
    zs = c["unchanged_from_exp004"]["z_star"]
    z_star = json.loads((ROOT / zs["file"]).read_text())["z_star"]
    if z_star != zs["value"]:
        raise SystemExit("pilot z* differs from the frozen value")
    cf.run(ROOT / c["cache"], out, ROOT / c["scores"], c["models"], KIND, z_star, a.workers)


def summary(c: dict, sha: str, with_states: bool = True) -> dict:
    import exp004_report as rep

    out = ROOT / c["out"]
    units = rep.load_units(KIND, out)
    auroc, folds = rep.per_repeat(units)
    models = {m for m, _, _ in auroc}
    missing = set(c["models"]) - models
    if missing or len(units) != len(c["models"]) * 50:
        raise SystemExit(f"{c['experiment']} incomplete: {len(units)} units, missing models {sorted(missing)}")
    a, b = c["main_contrast"]
    main = e10.contrast(auroc, folds, a, b, c["reading"]["concordance"])
    res = {"experiment": c["experiment"], "learner": KIND, "config_sha256": sha,
           "status": "pre-specified robustness analysis (06 §5); descriptive; never relabels RESULT-001",
           "reference": c["reading"]["reference"], "main": {f"{a}_vs_{b}": main},
           "extra": {f"{x}_vs_{y}": e10.contrast(auroc, folds, x, y) for x, y in c["extra_contrasts"]},
           "other": {f"{x}_vs_{y}": e10.contrast(auroc, folds, x, y) for x, y in cf.SECONDARY},
           "mean_auroc": {m: e10.mean_auroc(auroc, m) for m in sorted(models)}}
    if "pc1_fc_only" in models:
        for (m, r, p) in list(auroc):
            if m == "pc1_fc_only":
                auroc[("chance", r, p)] = 0.5
        pc1 = est.delta_summary(auroc, "pc1_fc_only", "chance", folds)
        ci = cf.intervals(pc1, tx.PRIMARY_INTERVAL)
        res["pc1"] = {"mean_auroc": 0.5 + pc1["delta_hat"], "lower95_m1": 0.5 + ci["0.95"][0],
                      "above_chance": bool(ci["0.95"][0] > 0), "note": "reported, not a gate for this analysis"}
    ref = res["reference"]["delta_hat"]
    res["comparison"] = {"delta_minus_result001": main["delta_hat"] - ref,
                         "same_sign": bool((main["delta_hat"] > 0) == (ref > 0)),
                         "concordance": main["concordance_with_result001"]}
    if with_states:
        import exp004_descriptive as dx

        L = dx.load(ROOT / c["cache"], ROOT / c["scores"], KIND, [a, b])
        base = {i: r["evaluable_states"] for i, r in L["recs"].items() if r.get("evaluable_states") and i in L["data"]}
        res["per_state"] = {st: {f"{a}_vs_{b}": dx.contrast(L, a, b, {i: [st] for i, s in base.items() if st in s},
                                                            labelled=False)} for st in dx.STATES}
        res["per_state"]["note"] = "estimate + 95% percentile bootstrap CI, no outcome label (R10); wake underpowered"
    return res


def write_figures(c: dict, res: dict) -> None:
    import exp004_report as rep

    out = ROOT / c["out"]
    units = rep.load_units(KIND, out)
    counts = rep.ied_counts()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    pairs = [(tuple(c["main_contrast"]), "main")] + [(tuple(x), "extra") for x in c["extra_contrasts"]]
    for (x, y), sect in pairs:
        key = f"{x}_vs_{y}"
        dl = rep.patient_deltas(units, x, y)
        r = res[sect][key]
        svg = rep.forest_svg(dl, counts, r["delta_hat"], r["intervals"]["0.90"], r["intervals"]["0.95"],
                             f"{c['experiment']} (sensitivity): {key}, HGB", axis=f"Δ within-patient AUROC ({x} − {y})")
        (out / "figures" / f"forest_{key}.svg").write_text(svg)
        with (out / "tables" / f"patient_delta_{key}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["eeg_id", "delta_p", "repeat_min", "repeat_max", "evaluable_ied_epochs"])
            for p, (d, lo, hi) in sorted(dl.items()):
                w.writerow([p, f"{d:.6f}", f"{lo:.6f}", f"{hi:.6f}", counts[p]])


def cmd_summary(a) -> None:
    c, sha = load_config(a.exp)
    check_frozen(a.exp, sha)
    res = summary(c, sha)
    res["provenance"] = rl.provenance()
    out = ROOT / c["out"]
    (out / f"summary_{KIND}.json").write_text(json.dumps(res, indent=2) + "\n")
    write_figures(c, res)
    print(json.dumps({"main": res["main"], "pc1": res.get("pc1"), "comparison": res["comparison"]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["cache-check", "run", "summary"])
    ap.add_argument("--exp", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    {"cache-check": cmd_cache_check, "run": cmd_run, "summary": cmd_summary}[a.command](a)


if __name__ == "__main__":
    main()
