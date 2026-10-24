"""EV-1 (06 §6.1; 07 EXP-002): synthetic coverage check of the locked Nadeau–Bengio interval with HGB.

Synthetic data only; no vEpiSet signal is read. The design mirrors the locked analysis:
- one synthetic patient per real primary-set recording, with its post-F-6 group (E / N / F), its
  per-state evaluable states and its per-state IED / non-IED epoch counts (capped for compute) from
  artifacts/dataset_audit/evaluable_recount.csv;
- the frozen F-3 outer folds (5 x 10, artifacts/protocol/f3/fold_manifest.csv) and, for the tuned mode,
  inner folds from src/design/inner_folds.py with its validity rule;
- HGB with the locked weights (1/n_p per epoch; negatives x pi+_s / pi-_s per training set), the locked
  per-repeat estimand (Mann–Whitney AUROC per evaluable state, IED-count weights, mean over repeats,
  equal patient weight) and the locked interval (src/evaluation/taxonomy.py nadeau_bengio_ci).

Plan A (Josue 2026-10-10; 11 LOG-2026-10-10-EV1-SCOPE): mode "fixed" uses one HGB configuration fixed in
advance; mode "tuned" runs the 16-configuration grid with nested inner CV (tuning-impact check only).
early_stopping is off for every fit.

The data-generating process ports the biostatistics-lens simulation (reviews/2026-10-10-biostatistics-
lens-review.md): per-patient covariate offset, IED shift along a patient-perturbed direction in the
baseline block, and a patient-varying shift g_p ~ N(gamma, tau) in an extra ("FC-like") block; plus a
sleep offset on every feature. The truth theta is the procedure-level estimand by Monte Carlo: the
procedure is trained on fresh patients with an outer-training composition and applied to fresh E patients.

Usage (from the repository root):
    python src/design/ev1_coverage.py run --config C --scenario S --range 0:1000 [--mode fixed|tuned]
    python src/design/ev1_coverage.py theta --config C --scenario S [--m 200] [--gamma G]
    python src/design/ev1_coverage.py summarise --config C
Each dataset writes one JSON line to <out>/<mode>/<scenario>.jsonl; runs resume by skipping done indices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "design"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
import inner_folds as inf  # noqa: E402
import taxonomy as tx  # noqa: E402

STATES = ("wake", "sleep")
GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, min_samples_leaf=msl, l2_regularization=l2)
        for lr in (0.05, 0.1) for ml in (15, 31) for msl in (20, 100) for l2 in (0, 1)]


def simplicity_key(c: dict) -> tuple:
    """Tie-break order for 'simpler configuration' (11 LOG-2026-10-10-EV1-SCOPE): smaller max_leaf_nodes,
    larger min_samples_leaf, larger l2_regularization, smaller learning_rate."""
    return (c["max_leaf_nodes"], -c["min_samples_leaf"], -c["l2_regularization"], c["learning_rate"])


# ---------------------------------------------------------------- design inputs

def load_design(cfg: dict, scen: str | None = None) -> dict:
    """Primary-set profiles (group, evaluable states, capped per-state counts) and the frozen outer folds.

    A scenario with "demote_e_to_n": k turns the k E patients with the fewest evaluable IED epochs (ties by
    eeg_id) into training-only N patients, as the protocol does for patients that lose evaluability (R3);
    their epochs, folds and random draws are unchanged (EV-1b n_E boundary case)."""
    d = cfg["dgp"]
    prof = {}
    for r in csv.DictReader((ROOT / d["recount_csv"]).open()):
        if r["analysis_set"] != "primary":
            continue
        counts = {s: (min(int(r[f"n_ied_{s}"]), d["ied_cap_per_state"]), min(int(r[f"n_nonied_{s}"]), d["nonied_cap_per_state"]))
                  for s in STATES}
        prof[r["eeg_id"]] = {"group": r["group_after"], "counts": counts,
                             "evaluable": [s for s in r["evaluable_states_after"].split("+") if s]}
    k = cfg["scenarios"][scen].get("demote_e_to_n", 0) if scen else 0
    if k:
        n_ied = {i: sum(p["counts"][s][0] for s in p["evaluable"]) for i, p in prof.items() if p["group"] == "E"}
        for i in sorted(n_ied, key=lambda i: (n_ied[i], i))[:k]:
            prof[i] = dict(prof[i], group="N", evaluable=[])
    splits = {}
    for r in csv.DictReader((ROOT / cfg["design"]["manifest"]).open()):
        if r["analysis_set"] == "primary":
            splits.setdefault((int(r["repeat"]), int(r["seed"])), {})[r["eeg_id"]] = int(r["outer_fold"])
    assert set(next(iter(splits.values()))) == set(prof), "manifest and recount disagree on the primary set"
    return {"profiles": prof, "splits": [(rep, seed, folds) for (rep, seed), folds in sorted(splits.items())]}


def inner_recs(cfg: dict) -> dict:
    """Recording table for inner_folds.derive, with the post-F-6 groups (as the confirmatory run uses)."""
    d = cfg["design"]
    recs = inf.f3.load_recordings(ROOT / d["channel_qc"], ROOT / d["state_matrix"])
    inf.apply_recount(recs, ROOT / cfg["dgp"]["recount_csv"])
    return recs


# ---------------------------------------------------------------- data-generating process

def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_patient(rng: np.random.Generator, prof: dict, dgp: dict, sc: dict) -> dict:
    db, df = dgp["d_base"], sc["d_F"]
    d = db + df
    rows, ys, st = [], [], []
    a = rng.normal(0, dgp["patient_offset_sd"], d)
    n_pos = sum(c[0] for c in prof["counts"].values())
    if n_pos:
        dir_b = unit(np.ones(db) / np.sqrt(db) + dgp["direction_heterogeneity"] * rng.normal(0, 1, db) / np.sqrt(db))
        dir_f = unit(np.ones(df) / np.sqrt(df) + dgp["direction_heterogeneity"] * rng.normal(0, 1, df) / np.sqrt(df))
        delta = rng.normal(*dgp["base_shift"])
        g = rng.normal(sc["gamma"], sc["tau"]) if sc["tau"] > 0 else float(sc["gamma"])
    for s_i, s in enumerate(STATES):
        n_ied, n_non = prof["counts"][s]
        n = n_ied + n_non
        if n == 0:
            continue
        x = rng.normal(0, 1, (n, d)) + a + dgp["sleep_offset"] * s_i
        if n_ied:
            x[:n_ied, :db] += delta * dir_b
            x[:n_ied, db:] += g * dir_f
        rows.append(x)
        ys.append(np.r_[np.ones(n_ied), np.zeros(n_non)])
        st.append(np.full(n, s_i))
    return {"X": np.vstack(rows), "y": np.concatenate(ys), "s": np.concatenate(st), "evaluable": prof["evaluable"]}


# ---------------------------------------------------------------- learner, weights, metric

def train_weights(pats: list[dict]) -> np.ndarray:
    """06 R9: 1/n_p per epoch; negatives x pi+_s / pi-_s (patient-weighted state shares in this training set)."""
    w = np.concatenate([np.full(p["y"].size, 1.0 / p["y"].size) for p in pats])
    y = np.concatenate([p["y"] for p in pats])
    s = np.concatenate([p["s"] for p in pats])
    pos, neg = y == 1, y == 0
    ratio = {}  # both shares from the 1/n_p weights, before any negative is rescaled
    for k in range(len(STATES)):
        pi_neg = w[neg & (s == k)].sum() / w[neg].sum()
        ratio[k] = (w[pos & (s == k)].sum() / w[pos].sum()) / pi_neg if pi_neg > 0 else 1.0
    for k, r in ratio.items():
        w[neg & (s == k)] *= r
    return w


def fit(pats: list[dict], cols: slice, params: dict, seed: int, max_iter: int = 300) -> HistGradientBoostingClassifier:
    X = np.vstack([p["X"][:, cols] for p in pats])
    y = np.concatenate([p["y"] for p in pats])
    m = HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=False, random_state=seed, **params)
    return m.fit(X, y, sample_weight=train_weights(pats))


def mw_auroc(score: np.ndarray, y: np.ndarray) -> float:
    r = rankdata(score)
    n1 = int(y.sum())
    n0 = y.size - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def patient_auroc(model, p: dict, cols: slice) -> float:
    """06 R2: Mann–Whitney AUROC per evaluable state, combined with IED-count weights."""
    sc = model.decision_function(p["X"][:, cols])
    num = den = 0.0
    for s in p["evaluable"]:
        k = p["s"] == STATES.index(s)
        n_ied = p["y"][k].sum()
        num += n_ied * mw_auroc(sc[k], p["y"][k])
        den += n_ied
    return num / den


# ---------------------------------------------------------------- procedure (fixed or tuned)

def procedure(train: list[dict], ids: list[str], cols: slice, mode: str, cfg: dict, seed: int,
              inner: dict | None, n_inner: int | None) -> HistGradientBoostingClassifier:
    it = cfg["design"]["max_iter"]
    if mode == "fixed":
        return fit(train, cols, cfg["design"]["fixed_config"], seed, it)
    best = None
    for c in GRID:
        vals = []
        for j in range(n_inner):
            tr = [p for p, i in zip(train, ids) if inner[i] != j]
            m = fit(tr, cols, c, seed, it)
            vals += [patient_auroc(m, p, cols) for p, i in zip(train, ids) if inner[i] == j and p["evaluable"]]
        key = (-np.mean(vals), simplicity_key(c))
        if best is None or key < best[0]:
            best = (key, c)
    return fit(train, cols, best[1], seed, it)


def dataset(args) -> dict:
    cfg, scen, i, mode = args
    sc, dgp = cfg["scenarios"][scen], cfg["dgp"]
    design = load_design(cfg, scen)
    recs = inner_recs(cfg) if mode == "tuned" else None
    s_idx = list(cfg["scenarios"]).index(scen)
    rng = np.random.default_rng([cfg["seeds"]["datasets"], s_idx, i])
    prof = design["profiles"]
    ids = sorted(prof)
    pats = {r: make_patient(rng, prof[r], dgp, sc) for r in ids}
    e_ids = [r for r in ids if prof[r]["group"] == "E"]
    base, full = slice(0, dgp["d_base"]), slice(None)
    d_pr = {r: [] for r in e_ids}
    fold_means = []
    for rep, rseed, folds in design["splits"]:
        for k in range(5):
            tr_ids = [r for r in ids if folds[r] != k]
            te_ids = [r for r in e_ids if folds[r] == k]
            inner = n_inner = None
            if mode == "tuned":
                inner, n_inner, bad = inf.derive(tr_ids, recs, rseed, k, [r for r in ids if folds[r] == k])
                if bad:
                    raise RuntimeError(f"inner-fold validity rule failed: repeat {rep} fold {k}: {bad}")
            train = [pats[r] for r in tr_ids]
            m_b = procedure(train, tr_ids, base, mode, cfg, rseed + k, inner, n_inner)
            m_f = procedure(train, tr_ids, full, mode, cfg, rseed + k, inner, n_inner)
            ds = [patient_auroc(m_f, pats[r], full) - patient_auroc(m_b, pats[r], base) for r in te_ids]
            for r, v in zip(te_ids, ds):
                d_pr[r].append(v)
            fold_means.append(float(np.mean(ds)))
    d_p = np.array([np.mean(d_pr[r]) for r in e_ids])
    est = float(d_p.mean())
    out = {"scenario": scen, "i": i, "mode": mode, "estimate": est, "n_e": len(e_ids),
           "s2_fold_repeat": float(np.var(fold_means, ddof=1)), "sd_delta_p": float(d_p.std(ddof=1))}
    for lvl in (0.90, 0.95):
        out[f"nb_{lvl:.2f}"] = list(tx.nadeau_bengio_ci(est, fold_means, len(e_ids), lvl))
        if cfg["inference"].get("decision_interval", "nb") == "m1":
            out[f"m1_{lvl:.2f}"] = list(tx.m1_ci(est, fold_means, d_p, lvl))
    return out


def theta_draw(args) -> float:
    """One Monte Carlo draw of the procedure-level estimand: fresh training set, fresh E test patients."""
    cfg, scen, m, gamma = args
    sc = dict(cfg["scenarios"][scen], **({"gamma": gamma} if gamma is not None else {}))
    dgp, t = cfg["dgp"], cfg["truth"]
    design = load_design(cfg, scen)
    prof = design["profiles"]
    s_idx = list(cfg["scenarios"]).index(scen)
    rng = np.random.default_rng([cfg["seeds"]["theta"], s_idx, m])
    _, rseed, folds = design["splits"][(m // 5) % len(design["splits"])]
    k = m % 5
    tr_ids = [r for r in sorted(prof) if folds[r] != k]
    train = [make_patient(rng, prof[r], dgp, sc) for r in tr_ids]
    base, full = slice(0, dgp["d_base"]), slice(None)
    m_b = procedure(train, tr_ids, base, "fixed", cfg, rseed + k, None, None)
    m_f = procedure(train, tr_ids, full, "fixed", cfg, rseed + k, None, None)
    e_prof = [prof[r] for r in sorted(prof) if prof[r]["group"] == "E"]
    test = [make_patient(rng, e_prof[j], dgp, sc) for j in rng.integers(len(e_prof), size=t["test_patients_per_set"])]
    return float(np.mean([patient_auroc(m_f, p, full) - patient_auroc(m_b, p, base) for p in test]))


# ---------------------------------------------------------------- CLI

def _out(cfg: dict, a) -> Path:
    return Path(a.out or cfg["output_dir"])


def read_results(path: Path, repair: bool = False) -> dict[int, dict]:
    """Finished datasets by index. Each index has a fixed seed, so a repeat must be identical and counts once.
    A truncated last line (process killed mid-write) is dropped, and with repair=True cut from the file."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    if lines[-1]:  # no trailing newline: the last write did not finish
        if repair:
            path.write_bytes(raw[: len(raw) - len(lines[-1])])
    out = {}
    for line in filter(None, lines[:-1]):
        r = json.loads(line)
        if r["i"] in out and out[r["i"]] != r:
            raise RuntimeError(f"{path}: dataset {r['i']} recorded twice with different results")
        out[r["i"]] = r
    return out


def cmd_run(cfg, a):
    lo, hi = map(int, a.range.split(":"))
    path = _out(cfg, a) / a.mode / f"{a.scenario}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set(read_results(path, repair=True))
    todo = [(cfg, a.scenario, i, a.mode) for i in range(lo, hi) if i not in done]
    with ProcessPoolExecutor(max_workers=a.workers) as ex, path.open("a") as f:
        for res in ex.map(dataset, todo):
            f.write(json.dumps(res) + "\n")
            f.flush()


def cmd_theta(cfg, a):
    m = a.m or cfg["truth"]["monte_carlo_training_sets"]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        vals = np.array(list(ex.map(theta_draw, [(cfg, a.scenario, j, a.gamma) for j in range(m)])))
    res = {"scenario": a.scenario, "gamma_override": a.gamma, "m": m, "theta": float(vals.mean()),
           "theta_mc_se": float(vals.std(ddof=1) / np.sqrt(m)), "sd_over_training_sets": float(vals.std(ddof=1))}
    out = _out(cfg, a) / "theta"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.scenario}" + (f"_gamma{a.gamma}" if a.gamma is not None else "")
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res))


def tuning_check(cfg: dict, base: Path) -> dict | None:
    """Descriptive tuned-vs-fixed comparison on the same datasets (plan A); never part of the verdict."""
    if "tuning_check" not in cfg:
        return None
    scen = cfg["tuning_check"]["scenario"]
    path = base / "tuned" / f"{scen}.jsonl"
    if not path.exists():
        return None
    tuned = read_results(path)
    fixed = read_results(base / "fixed" / f"{scen}.jsonl")
    both = sorted(set(tuned) & set(fixed))
    hw = lambda r: (r["nb_0.95"][1] - r["nb_0.95"][0]) / 2  # noqa: E731
    d_est = np.array([tuned[i]["estimate"] - fixed[i]["estimate"] for i in both])
    return {"scenario": scen, "n_paired": len(both), "role": cfg["tuning_check"]["role"],
            "mean_estimate_diff_tuned_minus_fixed": float(d_est.mean()),
            "sd_estimate_diff": float(d_est.std(ddof=1)) if len(both) > 1 else None,
            "median_halfwidth95_ratio_tuned_over_fixed": float(np.median([hw(tuned[i]) / hw(fixed[i]) for i in both])),
            "median_s2_ratio_tuned_over_fixed": float(np.median([tuned[i]["s2_fold_repeat"] / fixed[i]["s2_fold_repeat"]
                                                                 for i in both]))}


def cmd_summarise(cfg, a):
    """Coverage per scenario and the approved decision rule (Josue 2026-10-10; 07 EXP-002)."""
    c, base = cfg["criteria"], _out(cfg, a)
    key = cfg["inference"].get("decision_interval", "nb")
    descr = [k for k in ("nb", "m1") if k != key]
    rows, verdicts = {}, []
    for scen in cfg["scenarios"]:
        th = json.loads((base / "theta" / f"{scen}.json").read_text())
        res = list(read_results(base / "fixed" / f"{scen}.jsonl").values())
        n = len(res)
        need = c["datasets_per_scenario"]
        need = need[scen] if isinstance(need, dict) else need
        row = {"n_datasets": n, "decision_interval": key, "theta": th["theta"], "theta_mc_se": th["theta_mc_se"]}
        for lvl, lim in (("0.95", c["min_coverage_95"]), ("0.90", c["min_coverage_90"])):
            cov = np.mean([r[f"{key}_{lvl}"][0] <= th["theta"] <= r[f"{key}_{lvl}"][1] for r in res])
            se = float(np.sqrt(cov * (1 - cov) / n))
            row[f"coverage_{lvl}"], row[f"mc_se_{lvl}"], row[f"limit_{lvl}"] = float(cov), se, lim
            for d in descr:  # descriptive only; never part of the verdict
                if all(f"{d}_{lvl}" in r for r in res):
                    row[f"descriptive_{d}_coverage_{lvl}"] = float(np.mean(
                        [r[f"{d}_{lvl}"][0] <= th["theta"] <= r[f"{d}_{lvl}"][1] for r in res]))
        fail = row["coverage_0.95"] < c["min_coverage_95"] or row["coverage_0.90"] < c["min_coverage_90"]
        imprecise = max(row["mc_se_0.95"], row["mc_se_0.90"]) > c["max_mc_se"] or n < need \
            or th["theta_mc_se"] > cfg["truth"]["max_theta_mc_se"]
        row["verdict"] = "FAIL" if fail else ("INDETERMINATE" if imprecise else "PASS")
        rows[scen] = row
        verdicts.append(row["verdict"])
    overall = "FAIL" if "FAIL" in verdicts else ("INDETERMINATE" if "INDETERMINATE" in verdicts else "PASS")
    out = {"experiment": cfg["experiment"], "result": overall,
           "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), "scenarios": rows,
           "tuning_check": tuning_check(cfg, base)}
    (base / "ev1_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run", "theta", "summarise"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenario")
    ap.add_argument("--range", default="0:1000")
    ap.add_argument("--mode", default="fixed", choices=["fixed", "tuned"])
    ap.add_argument("--m", type=int)
    ap.add_argument("--gamma", type=float, help="calibration only: override the scenario's gamma")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    {"run": cmd_run, "theta": cmd_theta, "summarise": cmd_summarise}[a.command](cfg, a)


if __name__ == "__main__":
    main()
