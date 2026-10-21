"""EXP-016: the two locked descriptive analyses not yet run, from the saved EXP-004 HGB predictions (no refit).

  rq4_within_patient  06 §2 RQ-4: in recordings with both generalized and focal IED epochs, the FC increment
                      (B2 + FC - B2) restricted to generalized positives minus the same restricted to focal positives,
                      within the same patient; positives of the other type dropped, all same-state non-IED epochs
                      kept, evaluability (>= 5 and >= 5 per state) recomputed per type. Mean over patients with a
                      95% patient-cluster percentile bootstrap; per-patient values. Descriptive, no label.
  distance_bins       06 §4.2 (D3): each kept non-IED epoch's temporal distance (s) to the nearest IED marker
                      (point "!" or run "!start".."!end", data/interim/vepiset/events.csv), measured to the closer
                      edge of the epoch; per bin (config), negatives restricted to that bin, evaluability recomputed;
                      B2 + FC - B2 (estimate + bootstrap 95%, no label, R10) and the mean within-patient AUROC of B2.

Writes artifacts/exp016/summary.json (aggregates only). Runs once.

    python src/pipeline/exp016_locked_descriptive.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features", "design", "pipeline", "data"):
    sys.path.insert(0, str(ROOT / "src" / p))
import confirmatory as cf  # noqa: E402
import estimand as est  # noqa: E402
import exp004_descriptive as dx  # noqa: E402
import run_ladder as rl  # noqa: E402
import vepiset_audit as va  # noqa: E402

CONFIG = ROOT / "experiments/EXP-016-locked-descriptive/config.json"
FROZEN_SHA256 = "573fbbfa9917a75a65551082dff7db01e622abffe8382671ad338c22ebfae590"  # frozen 2026-10-21 (bins chosen by Josue)
EPOCH_S = va.EPOCH_SAMPLES / va.SFREQ


def load_config() -> tuple[dict, str]:
    raw = CONFIG.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check_frozen(sha: str) -> None:
    if FROZEN_SHA256 is None or sha != FROZEN_SHA256:
        raise SystemExit(f"EXP-016 config is not the frozen one ({sha} vs {FROZEN_SHA256})")


def type_codes(L: dict, epochs_csv: Path) -> dict:
    lab = {(r["eeg_id"], int(r["start"])): int(r["label"]) for r in csv.DictReader(epochs_csv.open())}
    return {i: np.array([lab[(i, int(s0))] for s0 in x["start"]]) for i, x in L["data"].items()}


def per_patient_delta(L: dict, a: str, b: str, keep: dict) -> dict:
    """Delta_p (mean over repeats) for patients evaluable on their kept epochs; keep[eeg_id] -> bool mask."""
    by = defaultdict(list)
    for (rep, i), sa in L["scores"][a].items():
        if i not in keep:
            continue
        x, k = L["data"][i], keep[i]
        ev = dx.evaluable_states(x["y"][k], x["s"][k], 5)
        if ev:
            sb = L["scores"][b][(rep, i)]
            by[i].append(est.patient_auroc(sa[k], x["y"][k], x["s"][k], ev)
                         - est.patient_auroc(sb[k], x["y"][k], x["s"][k], ev))
    return {i: float(np.mean(v)) for i, v in by.items()}


def rq4_within_patient(L: dict, codes: dict, c: dict) -> dict:
    a, b = cf.PRIMARY
    d = {}
    for name, cs in dx.IED_TYPES.items():
        keep = {i: (L["data"][i]["y"] == 0) | np.isin(codes[i], list(cs)) for i in L["data"]}
        d[name] = per_patient_delta(L, a, b, keep)
    both = sorted(set(d["generalized"]) & set(d["focal"]))
    diff = [d["generalized"][i] - d["focal"][i] for i in both]
    res = {"n_patients_both_types": len(both),
           "per_patient": {i: {"delta_generalized": d["generalized"][i], "delta_focal": d["focal"][i],
                               "difference": d["generalized"][i] - d["focal"][i]} for i in both},
           "note": "difference = Delta_p(generalized positives) - Delta_p(focal positives), B2 + FC - B2; "
                   "descriptive, no label (06 §2 RQ-4, explicitly underpowered)"}
    if len(both) >= 3:
        res["mean_difference"] = float(np.mean(diff))
        res["bootstrap95"] = list(cf.tx.cluster_bootstrap_ci(diff, 0.95, c["n_bootstrap"], c["seed"]))
    return res


def distances(L: dict, events_csv: Path) -> dict:
    """Seconds from each kept epoch to the nearest IED marker (0 if a marker falls inside; inf if none)."""
    ev = defaultdict(list)
    for e in csv.DictReader(events_csv.open()):
        ev[e["eeg_id"]].append({"onset_s": float(e["onset_s"]), "text": e["text"]})
    out = {}
    for i, x in L["data"].items():
        points, runs, _ = va.ied_intervals(ev.get(i, []))
        iv = [(p, p) for p in points] + list(runs)
        t0 = x["start"] / va.SFREQ
        t1 = t0 + EPOCH_S
        if iv:
            lo, hi = np.array([u for u, _ in iv]), np.array([v for _, v in iv])
            out[i] = np.maximum(0, np.maximum(t0[:, None] - hi[None], lo[None] - t1[:, None])).min(axis=1)
        else:
            out[i] = np.full(t0.size, np.inf)
    return out


def distance_bins(L: dict, dist: dict, base: dict, c: dict) -> dict:
    a, b = cf.PRIMARY
    res = {}
    for lo, hi in c["distance_bins_s"]:
        hi = np.inf if hi is None else hi
        key = f"[{lo}, {hi})"

        def select(i, x, lo=lo, hi=hi):
            inb = (dist[i] >= lo) & (dist[i] < hi)
            return (x["y"] == 1) | ((x["y"] == 0) & inb), x["y"]

        row = {f"{a}_vs_{b}": dx.contrast_selected(L, a, b, select, labelled=False)}
        keep = {i: select(i, L["data"][i])[0] for i in base}
        au = defaultdict(list)
        for (rep, i), s in L["scores"][b].items():
            if i in keep:
                x, k = L["data"][i], keep[i]
                evs = dx.evaluable_states(x["y"][k], x["s"][k], 5)
                if evs:
                    au[i].append(est.patient_auroc(s[k], x["y"][k], x["s"][k], evs))
        row["b2_mean_within_patient_auroc"] = float(np.mean([np.mean(v) for v in au.values()])) if au else None
        row["n_patients"] = len(au)
        row["n_negatives"] = int(sum(((L["data"][i]["y"] == 0) & select(i, L["data"][i])[0]).sum() for i in base))
        res[key] = row
    allneg = np.concatenate([dist[i][L["data"][i]["y"] == 0] for i in base])
    res["negatives_distance_quantiles_s"] = {q: float(np.quantile(allneg, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    res["note"] = "negatives restricted by distance to the nearest IED marker; IED epochs all kept; R10 format"
    return res


def run(c: dict) -> dict:
    L = dx.load(ROOT / c["cache"], ROOT / c["scores"], "hgb", list(cf.PRIMARY))
    base = {i: r["evaluable_states"] for i, r in L["recs"].items() if r.get("evaluable_states") and i in L["data"]}
    a, b = cf.PRIMARY
    chk = est.delta_summary(dx.aurocs(L, (a, b), base), a, b, L["folds"])["delta_hat"]
    want = json.loads((ROOT / c["reference_summary"]).read_text())["contrasts"][f"{a}_vs_{b}"]["delta_hat"]
    if abs(chk - want) > 1e-12:
        raise SystemExit(f"score files disagree with the EXP-004 summary: {chk} vs {want}")
    res = {"experiment": "EXP-016", "learner": "hgb", "check": {"primary_delta_recomputed": chk, "matches": True},
           "status": "descriptive; locked 06 analyses; never relabels RESULT-001",
           "rq4_within_patient": rq4_within_patient(L, type_codes(L, ROOT / c["epochs"]), c)}
    if c.get("distance_bins_s"):
        res["distance_bins"] = distance_bins(L, distances(L, ROOT / c["events"]), base, c)
    return res


def main() -> None:
    c, sha = load_config()
    check_frozen(sha)
    out = ROOT / c["out"]
    if (out / "summary.json").exists():
        raise SystemExit("EXP-016 runs once: a summary already exists")
    res = run(c)
    res["config_sha256"] = sha
    res["provenance"] = rl.provenance()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=2)[:4000])


if __name__ == "__main__":
    main()
