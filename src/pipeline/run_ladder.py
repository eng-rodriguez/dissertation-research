"""Run the comparator ladder on cached features: the pilot (EXP-003) now; the confirmatory run (EXP-004) only
behind its gate.

Commands (from the repository root, on the Mac; inputs are the per-recording caches of src/features/extract.py):
  zstar  Fix the B2 salience threshold z* from the pilot, label-blind (spec §3, amended by Josue 2026-10-12):
         per pilot recording, the 95th percentile of all per-channel window-peak salience z over its kept
         windows; z* = the median of the 8 recording-level values, unrounded. Labels are never read.
         Writes artifacts/pilot/zstar.json. Refuses to overwrite an existing value.
  pilot  EXP-003: leave-one-recording-out over the 8 frozen pilot recordings with the plan-A fixed HGB
         configuration (the inner-fold validity rule cannot be met on 8 recordings). Writes aggregate,
         pilot-only outputs to artifacts/pilot/: per evaluable pilot patient and model, the within-patient
         same-state AUROC; the B2 level (ceiling check); the window-validity share (diagnostic only).
  timing Fit time per ladder model on synthetic data of the real primary-set size (no real data read).
  cache-check  Before EXP-004, no model fitted: for every primary recording, the cached kept-epoch counts per
         class and state must equal the F-6 recount, every block must have its locked shape, and B1/B2/ADJ must
         be finite. Writes aggregate artifacts/exp004/cache_check.json with each cache file's SHA-256.
  confirmatory, confirmatory-summary  EXP-004 (src/pipeline/confirmatory.py). Refuse to start unless
         artifacts/protocol/unblinding_gate.json records PC-3 PASS, the pre-confirmatory coverage check PASS,
         the adopted primary interval ("m1", RDR-014) and Josue's written approval (approved_by, approved_on),
         and the cache check passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features"):
    sys.path.insert(0, str(ROOT / "src" / p))
import b2 as fb2  # noqa: E402
import ladder as L  # noqa: E402
import taxonomy as tx  # noqa: E402

PILOT_CSV = ROOT / "artifacts/protocol/f3/pilot_subset.csv"
RECOUNT_CSV = ROOT / "artifacts/dataset_audit/evaluable_recount.csv"
OUT = ROOT / "artifacts/pilot"
GATE = ROOT / "artifacts/protocol/unblinding_gate.json"
CACHE_CHECK = ROOT / "artifacts/exp004/cache_check.json"
SHAPES = {"b1": (19, 24), "b2_topo": (17,), "adj": (912,), "fc": (513,), "fc_surr": (513,)}
PLAN_A = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1)
PILOT_SEED = 20261010
HGB_MODELS = ["b0", "b1", "b2", "b2_adj", "b2_fcsurr", "b2_fc", "pc1_fc_only"]


def pilot_ids() -> list[str]:
    return sorted(r["eeg_id"] for r in csv.DictReader(PILOT_CSV.open()))


def evaluable_states() -> dict[str, list[str]]:
    return {r["eeg_id"]: [s for s in r["evaluable_states_after"].split("+") if s]
            for r in csv.DictReader(RECOUNT_CSV.open())}


def load_patient(cache: Path, eeg_id: str, evaluable: list[str], z_star: float) -> dict:
    with np.load(cache / f"{eeg_id}.npz") as z:
        F = {k: z[k] for k in z.files}
    F["b2_count"] = fb2.salience_count(F["peak_z"], z_star)[:, None]
    return {"id": eeg_id, "y": F["label"], "s": F["state"], "evaluable": evaluable, "F": F}


def provenance() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        commit = "unknown"
    import sklearn
    return {"git_commit": commit, "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__}


def z_star_from(peaks_by_recording: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    """Median over recordings of each recording's 95th percentile of window-peak salience z (unrounded)."""
    per = {i: float(np.percentile(np.asarray(z).ravel(), 95)) for i, z in sorted(peaks_by_recording.items())}
    return float(np.median(list(per.values()))), per


def cmd_zstar(a) -> None:
    out = OUT / "zstar.json"
    if out.exists():
        raise SystemExit(f"{out} exists; z* is fixed once (spec §3)")
    peaks = {}
    for i in pilot_ids():
        with np.load(a.cache / f"{i}.npz") as z:
            peaks[i] = z["peak_z"]  # labels are never loaded here
    z_star, per = z_star_from(peaks)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"z_star": z_star, "z_star_repr": repr(z_star),
                               "rule": "per pilot recording, the 95th percentile of all per-channel window-peak "
                                       "salience z over its kept windows (labels unused); z* = median of the 8 "
                                       "recording-level values; not rounded (spec §3, amended 2026-10-12)",
                               "per_recording_p95": per,
                               "n_windows": {i: int(np.asarray(z).shape[0]) for i, z in peaks.items()},
                               "provenance": provenance()}, indent=2) + "\n")
    print(repr(z_star))


def cmd_pilot(a) -> None:
    z_star = json.loads((OUT / "zstar.json").read_text())["z_star"]
    ev = evaluable_states()
    ids = pilot_ids()
    pats = {i: load_patient(a.cache, i, ev[i], z_star) for i in ids}
    rows, t0 = [], time.time()
    for i in ids:
        if not pats[i]["evaluable"]:
            continue
        train = [pats[j] for j in ids if j != i]
        for m in HGB_MODELS:
            res, _ = L.outer_split(train, [pats[i]], m, "hgb", PILOT_SEED, fixed=PLAN_A)
            rows.append({"eeg_id": i, "model": m, "auroc": round(res[i], 4)})
            print(i, m, rows[-1]["auroc"], flush=True)
    by = {m: [r["auroc"] for r in rows if r["model"] == m] for m in HGB_MODELS}
    ied = [(pats[i]["F"]["window_has_marker"][pats[i]["y"] == 1]) for i in ids]
    summary = {
        "experiment": "EXP-003", "status": "exploratory/debug; never a scientific result (07)",
        "design": "leave-one-recording-out over the 8 frozen pilot recordings; plan-A fixed HGB configuration",
        "z_star": z_star, "evaluable_pilot_patients": sorted({r["eeg_id"] for r in rows}),
        "mean_auroc_by_model": {m: round(float(np.mean(v)), 4) for m, v in by.items() if v},
        "b2_ceiling_check": {"mean": round(float(np.mean(by["b2"])), 4), "min": min(by["b2"]), "max": max(by["b2"])},
        "window_validity_share": round(float(np.concatenate(ied).mean()), 4),
        "window_validity_note": "diagnostic only; must not be used to tune the window rule (07 EXP-003)",
        "n_epochs": {i: int(len(pats[i]["y"])) for i in ids}, "runtime_s": round(time.time() - t0),
        "provenance": provenance(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "pilot_auroc.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eeg_id", "model", "auroc"])
        w.writeheader()
        w.writerows(rows)
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def check_gate() -> dict:
    need = ("pc3", "coverage_check", "primary_interval", "approved_by", "approved_on")
    gate = json.loads(GATE.read_text()) if GATE.exists() else {}
    missing = [k for k in need if not gate.get(k)]
    if missing or gate.get("pc3") != "PASS" or gate.get("coverage_check") != "PASS" \
            or gate.get("primary_interval") != tx.PRIMARY_INTERVAL:
        raise SystemExit(f"EXP-004 is blinded: gate file incomplete or not passed ({missing or gate})")
    return gate


def cache_check(cache: Path, recount_csv: Path = RECOUNT_CSV, shapes: dict = SHAPES,
                sets: tuple = ("primary",), lifted: tuple = ()) -> dict:
    """lifted (excluded-epochs sensitivity only): the epochs whose locked_reason is "" must reproduce the
    recount exactly, and every other kept epoch must carry one of the lifted reasons."""
    import hashlib
    from collections import Counter

    lifted_counts: Counter = Counter()

    rows = [r for r in csv.DictReader(recount_csv.open()) if r["analysis_set"] in sets]
    problems, files, nonfinite_fc, zero_den = {}, {}, 0, 0
    for r in rows:
        i, path = r["eeg_id"], cache / f"{r['eeg_id']}.npz"
        if not path.exists():
            problems[i] = ["missing"]
            continue
        files[i] = hashlib.sha256(path.read_bytes()).hexdigest()
        bad = []
        with np.load(path) as z:
            n = len(z["label"])
            base = np.ones(n, bool)
            if lifted:
                reason = z["locked_reason"] if "locked_reason" in z.files else np.full(n, "?")
                base = reason == ""
                if set(reason[~base]) - set(lifted):
                    bad.append(f"unexpected locked reasons {sorted(set(reason[~base]) - set(lifted))}")
                lifted_counts.update(f"{x}_{'ied' if lab else 'nonied'}" for x, lab in zip(reason[~base], z["label"][~base]))
            for c, lab in (("ied", 1), ("nonied", 0)):
                for s in ("wake", "sleep"):
                    got = int((base & (z["label"] == lab) & (z["state"] == s)).sum())
                    if got != int(r[f"n_{c}_{s}"]):
                        bad.append(f"n_{c}_{s} {got} != {r[f'n_{c}_{s}']}")
            if int(base.sum()) != int(sum(int(r[f"n_{c}_{s}"]) for c in ("ied", "nonied") for s in ("wake", "sleep"))):
                bad.append(f"{n} epochs, some outside wake/sleep")
            for k, shape in shapes.items():
                if k not in z.files or z[k].shape != (n, *shape):
                    bad.append(f"{k} shape {z[k].shape if k in z.files else 'absent'}")
                elif k in ("b1", "b2_topo", "adj") and not np.isfinite(z[k]).all():
                    bad.append(f"{k} non-finite")
                elif k.startswith("fc"):
                    nonfinite_fc += int((~np.isfinite(z[k])).sum())
            if z["peak_z"].shape != (n, shapes["b1"][0]):
                bad.append(f"peak_z shape {z['peak_z'].shape}")
            zero_den += int(z["fc_zero_denominator"])
        if bad:
            problems[i] = bad
    return {"result": "FAIL" if problems or len(files) != len(rows) else "PASS", "n_recordings": len(rows),
            "problems": problems, "fc_nonfinite_values": nonfinite_fc, "fc_zero_denominator": zero_den,
            **({"lifted_epochs": dict(sorted(lifted_counts.items()))} if lifted else {}), "cache_sha256": files}


def cmd_cache_check(a) -> None:
    res = cache_check(a.cache)
    res["cache"] = str(a.cache.relative_to(ROOT)) if a.cache.is_relative_to(ROOT) else str(a.cache)
    res["provenance"] = provenance()
    CACHE_CHECK.parent.mkdir(parents=True, exist_ok=True)
    CACHE_CHECK.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("cache_sha256", "provenance")}, indent=2))


def cmd_confirmatory(a) -> None:
    check_gate()
    if not CACHE_CHECK.exists() or json.loads(CACHE_CHECK.read_text())["result"] != "PASS":
        raise SystemExit("EXP-004 needs a passing cache check first (run_ladder.py cache-check)")
    import confirmatory as cf

    z_star = json.loads((OUT / "zstar.json").read_text())["z_star"]
    cf.run(a.cache, ROOT / "artifacts/exp004", ROOT / "data/processed/exp004/scores", a.models, a.learner, z_star,
           a.workers)


def cmd_confirmatory_summary(a) -> None:
    gate = check_gate()
    import confirmatory as cf

    res = cf.summarise(ROOT / "artifacts/exp004", a.learner, gate["primary_interval"], gate["pc3"] == "PASS")
    res["gate"] = gate
    res["provenance"] = provenance()
    out = ROOT / "artifacts/exp004" / f"summary_{a.learner}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


def cmd_timing(a) -> None:
    """Fit time per ladder model on SYNTHETIC data with the real primary-set size (no real data read)."""
    import confirmatory as cf

    d = cf.design()
    counts = {r["eeg_id"]: sum(int(r[f"n_{c}_{s}"]) for c in ("ied", "nonied") for s in ("wake", "sleep"))
              for r in csv.DictReader(RECOUNT_CSV.open()) if r["analysis_set"] == "primary"}
    (rep, seed), folds = next(iter(d["splits"].items()))
    train_ids = [i for i in sorted(folds) if folds[i] != 0]
    rng = np.random.default_rng(0)
    dims = {"b1": (19, 24), "b2_topo": (17,), "adj": (912,), "fc": (513,), "fc_surr": (513,)}
    pats = []
    for i in train_ids:
        n = max(counts.get(i, 0), 10)
        y = (rng.random(n) < 0.1).astype(int)
        F = {k: rng.standard_normal((n, *v)) for k, v in dims.items()}
        F["b2_count"] = rng.integers(0, 19, (n, 1)).astype(float)
        pats.append({"id": i, "y": y, "s": np.where(rng.random(n) < 0.6, "sleep", "wake"), "evaluable": [], "F": F})
    n_fits = 50 * (len(L.HGB_GRID) * 4 + 1)
    rows = {}
    for m in a.models:
        t0 = time.time()
        L.fit(pats, m, "hgb", PLAN_A, 1)
        sec = time.time() - t0
        rows[m] = {"seconds_per_outer_fit": round(sec, 1), "hours_for_model": round(sec * n_fits * 0.85 / 3600, 1)}
        print(m, rows[m], flush=True)
    total = sum(r["hours_for_model"] for r in rows.values())
    print(json.dumps({"n_train_epochs": int(sum(len(p["y"]) for p in pats)), "per_model": rows,
                      "approx_total_hours_single_stream": round(total, 1),
                      "note": "inner fits use about 3/4 of the outer training set (factor 0.85 applied)"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["zstar", "pilot", "timing", "cache-check", "confirmatory", "confirmatory-summary"])
    ap.add_argument("--cache", type=Path, default=ROOT / "data/interim/features/salience2s")
    ap.add_argument("--models", nargs="+", default=HGB_MODELS)
    ap.add_argument("--learner", default="hgb", choices=["hgb", "lr"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    {"zstar": cmd_zstar, "pilot": cmd_pilot, "timing": cmd_timing, "cache-check": cmd_cache_check,
     "confirmatory": cmd_confirmatory,
     "confirmatory-summary": cmd_confirmatory_summary}[a.command](a)


if __name__ == "__main__":
    main()
