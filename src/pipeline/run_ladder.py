"""Run the comparator ladder on cached features: the pilot (EXP-003) now; the confirmatory run (EXP-004) only
behind its gate.

Commands (from the repository root, on the Mac; inputs are the per-recording caches of src/features/extract.py):
  zstar  Fix the B2 salience threshold z* from the pilot, label-blind (spec §3): the 95th percentile of every
         per-channel window-peak salience z over all kept pilot windows, labels never read; rounded to 0.1.
         Writes artifacts/pilot/zstar.json. Refuses to overwrite an existing value.
  pilot  EXP-003: leave-one-recording-out over the 8 frozen pilot recordings with the plan-A fixed HGB
         configuration (the inner-fold validity rule cannot be met on 8 recordings). Writes aggregate,
         pilot-only outputs to artifacts/pilot/: per evaluable pilot patient and model, the within-patient
         same-state AUROC; the B2 level (ceiling check); the window-validity share (diagnostic only).
  confirmatory  EXP-004. Refuses to start unless artifacts/protocol/unblinding_gate.json records PC-3 PASS,
         the pre-confirmatory coverage check PASS, the adopted primary interval, and Josue's written approval.
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

PILOT_CSV = ROOT / "artifacts/protocol/f3/pilot_subset.csv"
RECOUNT_CSV = ROOT / "artifacts/dataset_audit/evaluable_recount.csv"
OUT = ROOT / "artifacts/pilot"
GATE = ROOT / "artifacts/protocol/unblinding_gate.json"
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


def cmd_zstar(a) -> None:
    out = OUT / "zstar.json"
    if out.exists():
        raise SystemExit(f"{out} exists; z* is fixed once (spec §3)")
    peaks = []
    for i in pilot_ids():
        with np.load(a.cache / f"{i}.npz") as z:
            peaks.append(z["peak_z"].ravel())  # labels are never loaded here
    allz = np.concatenate(peaks)
    z_star = round(float(np.percentile(allz, 95)), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"z_star": z_star, "rule": "95th percentile of per-channel window-peak salience z "
                               "over all kept pilot windows, labels unused, rounded to 0.1 (spec §3)",
                               "n_windows": int(allz.size // 19), "provenance": provenance()}, indent=2) + "\n")
    print(z_star)


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


def cmd_confirmatory(a) -> None:
    need = ("pc3", "coverage_check", "primary_interval", "approved_by", "approved_on")
    gate = json.loads(GATE.read_text()) if GATE.exists() else {}
    missing = [k for k in need if not gate.get(k)]
    if missing or gate.get("pc3") != "PASS" or gate.get("coverage_check") != "PASS":
        raise SystemExit(f"EXP-004 is blinded: gate file incomplete or not passed ({missing or gate})")
    raise SystemExit("EXP-004 runner not implemented yet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["zstar", "pilot", "confirmatory"])
    ap.add_argument("--cache", type=Path, default=ROOT / "data/interim/features/salience2s")
    a = ap.parse_args()
    {"zstar": cmd_zstar, "pilot": cmd_pilot, "confirmatory": cmd_confirmatory}[a.command](a)


if __name__ == "__main__":
    main()
