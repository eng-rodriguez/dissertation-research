"""F-2: precision-feasibility simulation for the primary estimand in 06 (draft v0.3).

Estimand (06, issue 2): equal-patient-weight mean of the within-patient, same-state
difference in AUROC (B2 + FC minus B2) over unseen IED-positive patients with at least
5 IED and 5 non-IED epochs in a common state.

This is a precision-feasibility analysis, not a power guarantee. It reads only the
committed aggregate table artifacts/dataset_audit/subject_state_matrix.csv (per-recording
epoch counts by class and state). No signal, feature or model is involved; baseline
AUROC, the patient-to-patient spread of the FC gain, the correlation between the two
models' scores and cross-validation refit noise are unknown and enter as scenario grids.
The model and the decision rule were fixed in 06 §6 before this script was run.

Usage (from the repository root):
    python src/design/f2_precision_sim.py

Writes artifacts/protocol/f2_precision/{f2_grid.csv, f2_summary.json, f2_report.md}.
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
from scipy import optimize, stats

N_IED_TOTAL, N_REC_TOTAL = 52, 84
MIN_EPOCHS = 5
SD_D = 0.5  # patient-to-patient SD of baseline separation (probit scale)
BASELINE_AUROC = (0.85, 0.90, 0.95)
TAU_G = (0.10, 0.25, 0.50)
R_SCORE = (0.80, 0.90)
SIGMA_F = (0.0, 0.01, 0.02)
PILOT = (0, 6, 8, 10)
QC_LOSS = (0, 5, 10)
TRUE_DELTA = (0.0, 0.03)
SESOI = 0.03
N_FOLDS, N_REPEATS = 5, 10
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)  # quadrature for E over a normal


def pilot_ied(n_pilot: int) -> int:
    """IED recordings in a pilot stratified by IED status (52 of 84 recordings)."""
    return int(round(n_pilot * N_IED_TOTAL / N_REC_TOTAL))


def load_patients(path: Path, count_scale: float = 1.0) -> list[list[tuple[int, int]]]:
    """Per IED recording: list of (n_ied, n_non_ied) per state with >=1 of each.

    Returns all 52 IED recordings; ineligible ones (no state with >=5 and >=5) are
    returned as empty lists so random removal draws from the full IED set.
    count_scale < 1 shrinks the simulated epoch counts (eligibility still uses the real
    counts), a crude stand-in for within-patient temporal clustering of epochs.
    """
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            n_ied_any = sum(int(r[k]) for k in ("ied_wake", "ied_sleep", "ied_mixed", "ied_unlabelled"))
            if n_ied_any == 0:
                continue
            cells = []
            for s in ("wake", "sleep"):
                pos = int(r[f"ied_{s}"])
                neg = int(r[f"all_{s}"]) - pos
                if pos >= 1 and neg >= 1:
                    cells.append((pos, neg))
            eligible = any(p >= MIN_EPOCHS and n >= MIN_EPOCHS for p, n in cells)
            cells = [(max(1, round(p * count_scale)), max(1, round(n * count_scale))) for p, n in cells]
            out.append(cells if eligible else [])
    assert len(out) == N_IED_TOTAL, len(out)
    return out


def mean_auroc(mu: float, sd: float) -> float:
    """E[Phi(D/sqrt2)] for D ~ N(mu, sd^2) (binormal AUROC, unit within-class variance)."""
    return float(np.sum(GH_W * stats.norm.cdf((mu + sd * GH_X) / np.sqrt(2))) / np.sum(GH_W))


def calibrate(target_auc: float, tau_g: float, target_delta: float) -> tuple[float, float]:
    mu_d = optimize.brentq(lambda m: mean_auroc(m, SD_D) - target_auc, -5, 10)
    sd_fc = np.sqrt(SD_D**2 + tau_g**2)
    gamma = optimize.brentq(lambda g: mean_auroc(mu_d + g, sd_fc) - target_auc - target_delta, -3, 3)
    return mu_d, gamma


def auroc_rows(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Row-wise Mann-Whitney AUROC; pos (R, n1), neg (R, n0); continuous scores, no ties."""
    n1, n0 = pos.shape[1], neg.shape[1]
    allv = np.concatenate([pos, neg], axis=1)
    ranks = np.argsort(np.argsort(allv, axis=1), axis=1) + 1
    return (ranks[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def simulate_patient_deltas(patients, mu_d, gamma, tau_g, r, n_rep, rng) -> np.ndarray:
    """(n_rep, 52) per-patient within-patient same-state DeltaAUROC; NaN if ineligible."""
    out = np.full((n_rep, len(patients)), np.nan)
    c = np.sqrt(1 - r**2)
    for j, cells in enumerate(patients):
        if not cells:
            continue
        d = mu_d + SD_D * rng.standard_normal(n_rep)
        g = gamma + tau_g * rng.standard_normal(n_rep)
        num, den = np.zeros(n_rep), 0
        for pos_n, neg_n in cells:
            za, zb = rng.standard_normal((n_rep, neg_n)), rng.standard_normal((n_rep, neg_n))
            neg_a, neg_b = za, r * za + c * zb
            pa, pb = rng.standard_normal((n_rep, pos_n)), rng.standard_normal((n_rep, pos_n))
            pos_a = d[:, None] + pa
            pos_b = (d + g)[:, None] + r * pa + c * pb
            delta = auroc_rows(pos_b, neg_b) - auroc_rows(pos_a, neg_a)
            num += pos_n * delta
            den += pos_n
        out[:, j] = num / den
    return out


def summarise(deltas, n_pilot, qc_loss, sigma_f, true_delta, rng) -> dict:
    n_rep, n_pat = deltas.shape
    n_remove = pilot_ied(n_pilot) + qc_loss
    est, hw_naive, n_elig = np.empty(n_rep), np.empty(n_rep), np.empty(n_rep)
    for i in range(n_rep):
        keep = np.ones(n_pat, bool)
        keep[rng.choice(n_pat, n_remove, replace=False)] = False
        x = deltas[i, keep]
        x = x[~np.isnan(x)]
        n = len(x)
        if sigma_f > 0:
            folds = rng.integers(0, N_FOLDS, size=(N_REPEATS, n))
            shifts = sigma_f * rng.standard_normal((N_REPEATS, N_FOLDS))
            x = x + shifts[np.arange(N_REPEATS)[:, None], folds].mean(axis=0)
        est[i] = x.mean()
        hw_naive[i] = stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / np.sqrt(n)
        n_elig[i] = n
    sd_rep = est.std(ddof=1)
    hw_honest = stats.norm.ppf(0.95) * sd_rep
    centre = est.mean()  # simulated true value (calibration target, up to MC error)
    lb95_naive = est - hw_naive * stats.norm.ppf(0.975) / stats.norm.ppf(0.95)
    lb95_honest = est - stats.norm.ppf(0.975) * sd_rep
    return {
        "n_eligible_mean": float(n_elig.mean()),
        "estimate_mean": float(centre),
        "hw90_naive_median": float(np.median(hw_naive)),
        "hw90_honest": float(hw_honest),
        "naive90_coverage": float(np.mean(np.abs(est - true_delta) <= hw_naive)),
        "p_equiv_honest": float(np.mean(np.abs(est) + hw_honest < SESOI)),
        "p_equiv_naive": float(np.mean(np.abs(est) + hw_naive < SESOI)),
        "p_lb95_gt0_honest": float(np.mean(lb95_honest > 0)),
        "p_lb95_gt0_naive": float(np.mean(lb95_naive > 0)),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--counts", default="artifacts/dataset_audit/subject_state_matrix.csv")
    ap.add_argument("--out", default="artifacts/protocol/f2_precision")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--count-scale", type=float, default=1.0,
                    help="shrink simulated epoch counts (sensitivity for within-patient clustering)")
    args = ap.parse_args()

    patients = load_patients(Path(args.counts), args.count_scale)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ai, auc in enumerate(BASELINE_AUROC):
        for ti, tau in enumerate(TAU_G):
            for ri, r in enumerate(R_SCORE):
                for di, true_delta in enumerate(TRUE_DELTA):
                    mu_d, gamma = calibrate(auc, tau, true_delta)
                    rng = np.random.default_rng([args.seed, ai, ti, ri, di])
                    deltas = simulate_patient_deltas(patients, mu_d, gamma, tau, r, args.reps, rng)
                    for sf in SIGMA_F:
                        for pilot in PILOT:
                            for qc in QC_LOSS:
                                srng = np.random.default_rng([args.seed, ai, ti, ri, di, int(sf * 1000), pilot, qc])
                                s = summarise(deltas, pilot, qc, sf, true_delta, srng)
                                rows.append({"baseline_auroc": auc, "tau_g": tau, "r": r, "true_delta": true_delta,
                                             "sigma_f": sf, "pilot": pilot, "pilot_ied": pilot_ied(pilot),
                                             "qc_loss": qc, "mu_d": round(mu_d, 4), "gamma": round(gamma, 4), **s})
                    print(f"done auc={auc} tau={tau} r={r} delta={true_delta}", file=sys.stderr)

    with (out / "f2_grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    def pick(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    central = pick(baseline_auroc=0.90, tau_g=0.25, r=0.90, sigma_f=0.01, true_delta=0.0)
    central = [r for r in central if r["pilot"] in (6, 8) and r["qc_loss"] in (0, 5)]
    pess = [r for r in pick(tau_g=0.50, r=0.80, sigma_f=0.02, pilot=8, qc_loss=10, true_delta=0.0)
            if r["baseline_auroc"] in (0.85, 0.90)]
    c_min = min(r["p_equiv_honest"] for r in central)
    p_min = min(r["p_equiv_honest"] for r in pess)
    if c_min >= 0.80 and p_min >= 0.50:
        verdict = "SUPPORTABLE"
    elif c_min >= 0.50:
        verdict = "MARGINAL"
    else:
        verdict = "NOT REALISTICALLY SUPPORTABLE"
    pilot_cells = {p: pick(baseline_auroc=0.90, tau_g=0.25, r=0.90, sigma_f=0.01, true_delta=0.0,
                           pilot=p, qc_loss=0)[0]["p_equiv_honest"] for p in PILOT}
    ok = [p for p in (6, 8, 10) if pilot_cells[6] - pilot_cells[p] <= 0.05]
    pilot_size = max(ok) if ok else 6
    summary = {
        "verdict": verdict,
        "rule": "central min P(equiv|Delta=0, honest CI) >= 0.80 and pessimistic min >= 0.50 -> SUPPORTABLE; "
                "central >= 0.50 -> MARGINAL; else NOT REALISTICALLY SUPPORTABLE (06 §6, fixed before run)",
        "central_cells": central,
        "pessimistic_cells": pess,
        "central_min_p_equiv": c_min,
        "pessimistic_min_p_equiv": p_min,
        "pilot_p_equiv_central_qc0": pilot_cells,
        "pilot_size_by_rule": pilot_size,
        "provenance": {"script": "src/design/f2_precision_sim.py", "git_commit": git_commit(),
                       "counts": args.counts, "count_scale": args.count_scale, "reps": args.reps, "seed": args.seed,
                       "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    (out / "f2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("verdict", "central_min_p_equiv", "pessimistic_min_p_equiv",
                                              "pilot_p_equiv_central_qc0", "pilot_size_by_rule")}, indent=2))


if __name__ == "__main__":
    main()
