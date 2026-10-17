"""F-2-style precision feasibility with the candidate M1 interval (11 LOG-2026-10-12-M1-CANDIDATE).

Reruns the F-2 precision question (src/design/f2_precision_sim.py, verdict MARGINAL on 2026-10-10) for the
interval that would replace the locked Nadeau-Bengio interval after the EV-1 failure. It uses what is now known
instead of F-2's removal grid:
- the 35 post-F-6 evaluable patients and their real evaluable-state epoch counts
  (artifacts/dataset_audit/evaluable_recount.csv);
- the frozen 5 x 10 fold manifest (artifacts/protocol/f3/fold_manifest.csv) for the fold means S^2.

Model (as F-2, model-light, no signal): per-patient Delta_p from binormal epoch scores, calibrated to a baseline
mean within-patient AUROC and a true mean gain, with patient-to-patient gain SD tau_g and score correlation r
(f2_precision_sim.calibrate / simulate_patient_deltas). Two noise terms the intervals must absorb:
- sigma_f: refit noise shared within each (repeat, outer fold), as in F-2 (visible in S^2);
- sigma_T: a dataset-level shift shared by all 50 fold models, not visible in S^2 or s(Delta_p) (new; the
  component EV-1 showed NB misses; grid 0 / 0.004 / 0.008 from the EV-1 null and SESOI residuals).

For each replicate both intervals are computed (90%: t_{0.95, n_E-1}):
  NB: SE^2 = (1/50 + 1/4) S^2;  M1: SE^2 = NB SE^2 + s^2(Delta_p)/n_E.

Decision rule (fixed before the run; F-2's rule applied to M1): P(equivalence | Delta = 0), equivalence = 90%
interval inside +/-0.03.
- central cells: AUROC 0.90, tau_g 0.25, r 0.90, sigma_f 0.01, sigma_T in {0, 0.004};
- pessimistic cells: AUROC in {0.85, 0.90}, tau_g 0.50, r 0.80, sigma_f 0.02, sigma_T 0.008.
central min >= 0.80 and pessimistic min >= 0.50 -> SUPPORTABLE; central min >= 0.50 -> MARGINAL;
else NOT REALISTICALLY SUPPORTABLE. NB is reported alongside for comparison, never for the verdict.

Usage (from the repository root):
    python src/design/f2m1_precision.py
Writes artifacts/protocol/f2m1_precision/{grid.csv, summary.json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "design"))
import f2_precision_sim as f2  # noqa: E402

SESOI = 0.03
BASELINE_AUROC = (0.85, 0.90, 0.95)
TAU_G = (0.10, 0.25, 0.50)
R_SCORE = (0.80, 0.90)
SIGMA_F = (0.0, 0.01, 0.02)
SIGMA_T = (0.0, 0.004, 0.008)
TRUE_DELTA = (0.0, 0.03)
J, RATIO = 50, 0.25


def load_e_patients(recount: Path, manifest: Path) -> tuple[list, np.ndarray]:
    """Per E patient: [(n_ied, n_nonied)] over evaluable states; and folds (10 repeats x n_E)."""
    pats, ids = [], []
    for r in csv.DictReader(recount.open()):
        if r["analysis_set"] == "primary" and r["group_after"] == "E":
            states = [s for s in r["evaluable_states_after"].split("+") if s]
            pats.append([(int(r[f"n_ied_{s}"]), int(r[f"n_nonied_{s}"])) for s in states])
            ids.append(r["eeg_id"])
    folds = {}
    for r in csv.DictReader(manifest.open()):
        if r["eeg_id"] in ids:
            folds.setdefault(int(r["repeat"]), {})[r["eeg_id"]] = int(r["outer_fold"])
    return pats, np.array([[folds[rep][i] for i in ids] for rep in sorted(folds)])


def summarise(deltas: np.ndarray, folds: np.ndarray, sigma_f: float, sigma_t: float, true_delta: float,
              rng: np.random.Generator) -> dict:
    n_rep, n = deltas.shape
    n_r, k = folds.shape[0], 5
    q90, q95 = stats.t.ppf(0.95, n - 1), stats.t.ppf(0.975, n - 1)
    out = {m: {"hw90": np.empty(n_rep), "est": np.empty(n_rep), "se": np.empty(n_rep)} for m in ("nb", "m1")}
    for i in range(n_rep):
        shifts = sigma_f * rng.standard_normal((n_r, k))
        d_pr = deltas[i][None, :] + shifts[np.arange(n_r)[:, None], folds]       # (repeats, patients)
        fm = np.array([d_pr[r, folds[r] == f].mean() for r in range(n_r) for f in range(k)])
        d_p = d_pr.mean(axis=0)
        est = d_p.mean() + sigma_t * rng.standard_normal()
        nb2 = (1 / J + RATIO) * fm.var(ddof=1)
        for m, v in (("nb", nb2), ("m1", nb2 + d_p.var(ddof=1) / n)):
            out[m]["se"][i], out[m]["est"][i], out[m]["hw90"][i] = np.sqrt(v), est, q90 * np.sqrt(v)
    res = {"n_e": n}
    for m, o in out.items():
        est, se = o["est"], o["se"]
        res |= {f"{m}_hw90_median": float(np.median(o["hw90"])),
                f"{m}_hw95_median": float(np.median(q95 * se)),
                f"{m}_p_equiv": float(np.mean(np.abs(est) + o["hw90"] < SESOI)),
                f"{m}_p_lb95_gt0": float(np.mean(est - q95 * se > 0)),
                f"{m}_p_lb95_gt_sesoi": float(np.mean(est - q95 * se > SESOI)),
                f"{m}_cov90": float(np.mean(np.abs(est - true_delta) <= o["hw90"]))}
    res["sd_estimate"] = float(out["m1"]["est"].std(ddof=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recount", default="artifacts/dataset_audit/evaluable_recount.csv")
    ap.add_argument("--manifest", default="artifacts/protocol/f3/fold_manifest.csv")
    ap.add_argument("--out", default="artifacts/protocol/f2m1_precision")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()
    pats, folds = load_e_patients(ROOT / a.recount, ROOT / a.manifest)
    assert len(pats) == 35 and folds.shape == (10, 35)
    rows = []
    for ai, auc in enumerate(BASELINE_AUROC):
        for ti, tau in enumerate(TAU_G):
            for ri, r in enumerate(R_SCORE):
                for di, td in enumerate(TRUE_DELTA):
                    mu_d, gamma = f2.calibrate(auc, tau, td)
                    rng = np.random.default_rng([a.seed, ai, ti, ri, di])
                    deltas = f2.simulate_patient_deltas(pats, mu_d, gamma, tau, r, a.reps, rng)
                    for fi, sf in enumerate(SIGMA_F):
                        for si, st in enumerate(SIGMA_T):
                            srng = np.random.default_rng([a.seed, ai, ti, ri, di, fi, si, 1])
                            rows.append({"baseline_auroc": auc, "tau_g": tau, "r": r, "true_delta": td,
                                         "sigma_f": sf, "sigma_t": st,
                                         **summarise(deltas, folds, sf, st, td, srng)})
                    print(f"done auc={auc} tau={tau} r={r} delta={td}", file=sys.stderr)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    def pick(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    central = [r for r in pick(baseline_auroc=0.90, tau_g=0.25, r=0.90, sigma_f=0.01, true_delta=0.0)
               if r["sigma_t"] in (0.0, 0.004)]
    pess = [r for r in pick(tau_g=0.50, r=0.80, sigma_f=0.02, sigma_t=0.008, true_delta=0.0)
            if r["baseline_auroc"] in (0.85, 0.90)]

    def verdict(m):
        c, p = min(r[f"{m}_p_equiv"] for r in central), min(r[f"{m}_p_equiv"] for r in pess)
        v = "SUPPORTABLE" if c >= 0.80 and p >= 0.50 else ("MARGINAL" if c >= 0.50 else "NOT REALISTICALLY SUPPORTABLE")
        return {"verdict": v, "central_min_p_equiv": c, "pessimistic_min_p_equiv": p}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        commit = "unknown"
    summary = {"m1": verdict("m1"), "nb_for_comparison": verdict("nb"),
               "rule": "F-2 rule on P(90% interval inside +/-0.03 | Delta = 0): central min >= 0.80 and pessimistic "
                       "min >= 0.50 -> SUPPORTABLE; central min >= 0.50 -> MARGINAL; else NOT REALISTICALLY "
                       "SUPPORTABLE (fixed before the run; verdict on M1)",
               "central_cells": central, "pessimistic_cells": pess,
               "provenance": {"script": "src/design/f2m1_precision.py", "git_commit": commit, "reps": a.reps,
                              "seed": a.seed, "python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("m1", "nb_for_comparison")}, indent=2))


if __name__ == "__main__":
    main()
