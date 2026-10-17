"""EXP-004 pre-specified descriptive analyses from the saved epoch scores (no model is refitted).

Every quantity here is descriptive (06 §3.6, §4.4 multiplicity note, §5); none can change the confirmatory label.
Runs on the Mac, where the epoch scores (data/processed/exp004/scores/) and the feature cache live; writes only
aggregate numbers to artifacts/exp004/descriptive/descriptive.json.

  check            the primary Delta-hat recomputed from the score files must equal summary_{learner}.json
                   (aborts otherwise: the scores and the unit results would disagree)
  per_state        every ladder contrast restricted to wake, then to sleep, over patients evaluable in that state
                   (06 issue 3, item 6; wake is flagged as underpowered); M1 and bootstrap intervals. Also B0 vs
                   B2 + FC and B0 vs B2 (not pre-specified; the B0 engineering finding's paired error analysis)
  thresholds       the primary contrast with evaluability at >= 3 and >= 10 IED and non-IED epochs per state instead
                   of >= 5 (06 §5); evaluability recomputed from the cached labels and states
  epoch_weighted   the primary Delta weighted by each patient's evaluable IED epochs instead of equally (06 §5)
  pooled           deployment-like: per repeat and state, AUROC and AUPRC of all held-out epochs pooled over all 69
                   primary recordings (IED-free included), B2 + FC minus B2, averaged over repeats (06 issue 2,
                   secondary metrics; AUPRC never compared across strata)
  pc2              PC-2 (06 issue 4): per evaluable patient and band, median global mean wPLI of IED windows minus
                   non-IED windows within each evaluable state, IED-count-weighted over states; Wilcoxon signed-rank
                   and sign test over patients, Holm-adjusted across the three bands. Reported whatever the outcome.

Not here (they need training-fold thresholds or calibration, i.e. refits): sensitivity at a fixed false-positive
rate per minute, Delta log-loss after calibration, false positives in IED-free patients at an operating point.

  sleep_depth      06 issue 3 item 8: negatives restricted to the relative-delta-power range of the patient's IED
                   windows in the same state (primary contrast and B0)
  by_ied_type      06 §5: positives restricted to generalized (label 1) or focal (labels 2-5, per 05) IED epochs
                   (primary contrast and B0); reads data/interim/vepiset/epochs.csv for the per-epoch label

    python src/pipeline/exp004_descriptive.py --learner hgb
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features", "design", "pipeline", "connectivity"):
    sys.path.insert(0, str(ROOT / "src" / p))
import confirmatory as cf  # noqa: E402
import estimand as est  # noqa: E402
import taxonomy as tx  # noqa: E402
import wpli as wp  # noqa: E402

STATES = ("wake", "sleep")
ENGINEERING = [("b0", "b2_fc"), ("b0", "b2")]  # Josue 2026-10-16: B0 as a secondary engineering finding
REL_DELTA = 18  # b1.NAMES index of relative delta power
IED_TYPES = {"generalized": {1}, "focal": {2, 3, 4, 5}}  # 05: 1 generalized; 2-5 frontal/temporal/centro-parietal/occipital
N_EDGES = 19 * 18 // 2
EXP004_MODELS = [m for m in cf.L.MODELS if m != "b2_fc_delta"]  # b2_fc_delta exists only in EXP-010 (4-s)


def evaluable_states(y: np.ndarray, s: np.ndarray, min_n: int) -> list[str]:
    return [st for st in STATES if ((y == 1) & (s == st)).sum() >= min_n and ((y == 0) & (s == st)).sum() >= min_n]


def load(cache: Path, scores_dir: Path, kind: str, models: list[str]) -> dict:
    d = cf.design()
    folds = {(rep, i): k for (rep, _), fs in d["splits"].items() for i, k in fs.items()}
    ids = sorted({i for (_, i) in folds})
    data = {}
    for i in ids:
        with np.load(cache / f"{i}.npz") as z:
            data[i] = {"y": z["label"], "s": z["state"], "fc": z["fc"], "start": z["epoch_start"],
                       "rel_delta": z["b1"][:, :, REL_DELTA].mean(axis=1)}
    sc = defaultdict(dict)
    for m in models:
        for (rep, seed), fs in d["splits"].items():
            for k in range(cf.f3.N_FOLDS):
                with np.load(scores_dir / f"{kind}_{m}_r{rep}_k{k}.npz") as z:
                    for i in z.files:
                        sc[m][(rep, i)] = z[i]
    return {"recs": d["recs"], "folds": folds, "data": data, "scores": sc}


def aurocs(L: dict, models, evaluable: dict) -> dict:
    """auroc[(model, repeat, eeg_id)] for patients with at least one evaluable state in `evaluable`."""
    out = {}
    for m in models:
        for (rep, i), score in L["scores"][m].items():
            if evaluable.get(i):
                x = L["data"][i]
                out[(m, rep, i)] = est.patient_auroc(score, x["y"], x["s"], evaluable[i])
    return out


def _report(d: dict, ci: dict, labelled: bool) -> dict:
    """Subgroups (state strata, 06 §4.4 R10; IED type likewise) report estimate + percentile-bootstrap CI and no
    outcome label; the other robustness rows keep a descriptive M1 label."""
    if not labelled:
        return {"delta_hat": d["delta_hat"], "n_e": d["n_e"], "reported_interval": "bootstrap (06 §4.4 R10)",
                "intervals": ci}
    return {"delta_hat": d["delta_hat"], "n_e": d["n_e"], "intervals": ci,
            "descriptive_label_m1": tx.classify(tuple(ci["0.95"]), tuple(ci["0.90"]))}


def contrast(L: dict, a: str, b: str, evaluable: dict, labelled: bool = True) -> dict:
    au = aurocs(L, (a, b), evaluable)
    if len({i for (_, _, i) in au}) < 3:
        return {"delta_hat": None, "n_e": len({i for (_, _, i) in au}), "note": "fewer than 3 patients"}
    d = est.delta_summary(au, a, b, L["folds"])
    ci = cf.intervals(d, tx.PRIMARY_INTERVAL)
    return _report(d, ci, labelled)


def contrast_selected(L: dict, a: str, b: str, select, min_n: int = 5, labelled: bool = True) -> dict:
    """A contrast where each patient's epochs are first reduced by select(eeg_id, data) -> (keep mask, labels);
    evaluability (>= min_n IED and non-IED per state) is recomputed on what is kept."""
    au = {}
    for m in (a, b):
        for (rep, i), score in L["scores"][m].items():
            keep, y = select(i, L["data"][i])
            s = L["data"][i]["s"][keep]
            ev = evaluable_states(y[keep], s, min_n)
            if ev:
                au[(m, rep, i)] = est.patient_auroc(score[keep], y[keep], s, ev)
    n = len({i for (_, _, i) in au})
    if n < 3:
        return {"delta_hat": None, "n_e": n, "note": "fewer than 3 patients"}
    d = est.delta_summary(au, a, b, L["folds"])
    ci = cf.intervals(d, tx.PRIMARY_INTERVAL)
    return _report(d, ci, labelled)


def sleep_depth_restricted(L: dict, base: dict) -> dict:
    """06 issue 3 item 8: within each patient and state, keep only negatives whose window relative delta power
    (mean over the 19 channels) lies within [min, max] of that patient's IED windows in the same state."""
    def select(i, x):
        keep = np.zeros(len(x["y"]), bool)
        if i not in base:
            return keep, x["y"]
        for st in base[i]:
            ied, non = (x["y"] == 1) & (x["s"] == st), (x["y"] == 0) & (x["s"] == st)
            lo, hi = x["rel_delta"][ied].min(), x["rel_delta"][ied].max()
            keep |= ied | (non & (x["rel_delta"] >= lo) & (x["rel_delta"] <= hi))
        return keep, x["y"]
    res = {f"{a}_vs_{b}": contrast_selected(L, a, b, select) for a, b in [cf.PRIMARY] + ENGINEERING}
    kept = [select(i, L["data"][i])[0] for i in base]
    res["share_negatives_kept"] = float(np.mean(np.concatenate(
        [k[(L["data"][i]["y"] == 0) & np.isin(L["data"][i]["s"], base[i])] for i, k in zip(base, kept)])))
    res["note"] = "proxy: relative delta power of the analysis window, channel mean; range = min-max of IED windows"
    return res


def by_ied_type(L: dict, epochs_csv: Path) -> dict:
    """06 §5 error analysis by IED type (05: label 1 generalized, 2-5 focal): positives restricted to one type,
    all same-state non-IED epochs kept as negatives."""
    lab = {}
    for r in csv.DictReader(epochs_csv.open()):
        lab[(r["eeg_id"], int(r["start"]))] = int(r["label"])
    res = {}
    for name, codes in IED_TYPES.items():
        def select(i, x, codes=codes):
            t = np.array([lab[(i, int(s0))] for s0 in x["start"]])
            return (x["y"] == 0) | np.isin(t, list(codes)), x["y"]
        res[name] = {f"{a}_vs_{b}": contrast_selected(L, a, b, select, labelled=False)
                     for a, b in [cf.PRIMARY] + ENGINEERING}
    return res


def per_state(L: dict, base: dict) -> dict:
    out = {}
    for st in STATES:
        ev = {i: [st] for i, s in base.items() if st in s}
        out[st] = {f"{a}_vs_{b}": contrast(L, a, b, ev, labelled=False)
                   for a, b in [cf.PRIMARY] + cf.SECONDARY + ENGINEERING}
    out["note"] = "wake is underpowered (06 issue 3, item 6)"
    return out


def epoch_weighted(L: dict, base: dict) -> dict:
    a, b = cf.PRIMARY
    d = est.delta_summary(aurocs(L, (a, b), base), a, b, L["folds"])
    w = {i: sum(int(((L["data"][i]["y"] == 1) & (L["data"][i]["s"] == s)).sum()) for s in base[i])
         for i in d["delta_p"]}
    tot = sum(w.values())
    return {"delta_weighted": sum(w[i] * v for i, v in d["delta_p"].items()) / tot,
            "delta_equal_weight": d["delta_hat"], "n_e": d["n_e"]}


def pooled(L: dict, a: str, b: str) -> dict:
    reps = sorted({r for (r, _) in L["folds"]})
    res = {}
    for st in STATES:
        rows = defaultdict(list)
        for r in reps:
            y, sa, sb = [], [], []
            for (rr, i), score in L["scores"][a].items():
                if rr != r:
                    continue
                k = L["data"][i]["s"] == st
                y.append(L["data"][i]["y"][k])
                sa.append(score[k])
                sb.append(L["scores"][b][(r, i)][k])
            y, sa, sb = np.concatenate(y), np.concatenate(sa), np.concatenate(sb)
            rows["d_auroc"].append(roc_auc_score(y, sa) - roc_auc_score(y, sb))
            rows["d_auprc"].append(average_precision_score(y, sa) - average_precision_score(y, sb))
            rows["auprc_ref"].append(average_precision_score(y, sb))
            rows["prevalence"].append(float(y.mean()))
        res[st] = {k: {"mean": float(np.mean(v)), "repeat_min": float(np.min(v)), "repeat_max": float(np.max(v))}
                   for k, v in rows.items()}
    res["note"] = "per state, all 69 primary recordings pooled within a repeat, averaged over repeats"
    return res


def pc2(L: dict, base: dict) -> dict:
    out = {}
    for j, band in enumerate(wp.PRIMARY_BANDS):
        diffs = {}
        for i, states in base.items():
            x = L["data"][i]
            g = x["fc"][:, j * N_EDGES:(j + 1) * N_EDGES].mean(axis=1)
            num = den = 0.0
            for st in states:
                ied, non = (x["y"] == 1) & (x["s"] == st), (x["y"] == 0) & (x["s"] == st)
                num += ied.sum() * (np.median(g[ied]) - np.median(g[non]))
                den += ied.sum()
            diffs[i] = float(num / den)
        v = np.array(list(diffs.values()))
        out[band] = {"median_diff": float(np.median(v)), "mean_diff": float(v.mean()), "n_positive": int((v > 0).sum()),
                     "n": int(v.size), "p_wilcoxon": float(stats.wilcoxon(v).pvalue),
                     "p_sign": float(stats.binomtest(int((v > 0).sum()), int((v != 0).sum())).pvalue)}
    order = sorted(out, key=lambda k: out[k]["p_wilcoxon"])
    running = 0.0
    for r, band in enumerate(order):  # Holm step-down across the three bands
        running = max(running, min(1.0, (len(order) - r) * out[band]["p_wilcoxon"]))
        out[band]["p_wilcoxon_holm"] = running
    out["note"] = "global mean wPLI over 171 edges; IED minus non-IED, same state; reported whatever the outcome"
    return out


def run(cache: Path, scores_dir: Path, kind: str, summary: dict, epochs_csv: Path | None = None) -> dict:
    L = load(cache, scores_dir, kind, EXP004_MODELS)
    base = {i: r["evaluable_states"] for i, r in L["recs"].items() if r.get("evaluable_states") and i in L["data"]}
    a, b = cf.PRIMARY
    chk = est.delta_summary(aurocs(L, (a, b), base), a, b, L["folds"])["delta_hat"]
    want = summary["contrasts"][f"{a}_vs_{b}"]["delta_hat"]
    if abs(chk - want) > 1e-12:
        raise SystemExit(f"score files disagree with the unit results: {chk} vs {want}")
    out = {"learner": kind, "status": "descriptive; never relabels the confirmatory result (06 §4.4)",
           "check": {"primary_delta_recomputed": chk, "matches_summary": True},
           "per_state": per_state(L, base), "epoch_weighted": epoch_weighted(L, base),
           "pooled": pooled(L, a, b), "pc2": pc2(L, base), "thresholds": {}}
    for n in (3, 10):
        ev = {i: evaluable_states(x["y"], x["s"], n) for i, x in L["data"].items()}
        out["thresholds"][f">={n}"] = contrast(L, a, b, ev)
    out["sleep_depth_restricted"] = sleep_depth_restricted(L, base)
    if epochs_csv is not None:
        out["by_ied_type"] = by_ied_type(L, epochs_csv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--learner", default="hgb", choices=["hgb", "lr"])
    ap.add_argument("--cache", type=Path, default=ROOT / "data/interim/features/salience2s")
    ap.add_argument("--scores", type=Path, default=ROOT / "data/processed/exp004/scores")
    ap.add_argument("--epochs", type=Path, default=ROOT / "data/interim/vepiset/epochs.csv")
    a = ap.parse_args()
    summary = json.loads((ROOT / f"artifacts/exp004/summary_{a.learner}.json").read_text())
    res = run(a.cache, a.scores, a.learner, summary, a.epochs)
    sys.path.insert(0, str(ROOT / "src" / "pipeline"))
    import run_ladder as rl

    res["provenance"] = rl.provenance()
    out = ROOT / "artifacts/exp004/descriptive"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"descriptive_{a.learner}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ("check", "epoch_weighted", "thresholds")}, indent=2))


if __name__ == "__main__":
    main()
