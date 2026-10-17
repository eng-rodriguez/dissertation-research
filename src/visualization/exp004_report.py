"""EXP-004 figures and tables from the committed aggregate outputs (no raw data, no epoch scores).

Reads artifacts/exp004/units/{hgb,lr}.jsonl and summary_{hgb,lr}.json; writes
  artifacts/exp004/figures/forest_b2fc_vs_b2_{kind}.svg   R6 forest plot: Delta_p per evaluable patient, ordered
                                                          by evaluable IED epochs, with the across-repeat min-max
                                                          and the M1 interval of Delta-hat (06 §3, §5)
  artifacts/exp004/tables/contrasts.csv                   every contrast, both learners: Delta-hat, M1, bootstrap
                                                          and (descriptive) NB intervals, labels
  artifacts/exp004/tables/model_auroc.csv                 mean within-patient AUROC per model and learner
  artifacts/exp004/tables/patient_delta_{kind}.csv        Delta_p, its repeat min/max and IED count per patient

The SVG is written directly so the figure needs no plotting dependency beyond the pinned environment.

    python src/visualization/exp004_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("evaluation", "pipeline"):
    sys.path.insert(0, str(ROOT / "src" / p))
import confirmatory as cf  # noqa: E402
import estimand as est  # noqa: E402

OUT = ROOT / "artifacts/exp004"
SESOI = 0.03
INK, INK2, GRID, SERIES = "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"


def load_units(kind: str, out: Path = OUT) -> list[dict]:
    return [json.loads(line) for line in (out / "units" / f"{kind}.jsonl").read_text().splitlines() if line.strip()]


def per_repeat(units: list[dict]) -> tuple[dict, dict]:
    auroc, folds = {}, {}
    for u in units:
        for p, a in u["auroc"].items():
            auroc[(u["model"], u["repeat"], p)] = a
            folds[(u["repeat"], p)] = u["outer_fold"]
    return auroc, folds


def patient_deltas(units: list[dict], model: str = "b2_fc", ref: str = "b2") -> dict:
    """Delta_p (as the estimand code computes it) with its min and max over the 10 repeats."""
    auroc, folds = per_repeat(units)
    d = est.delta_summary(auroc, model, ref, folds)
    by_p = defaultdict(list)
    for (m, r, p), a in auroc.items():
        if m == model:
            by_p[p].append(a - auroc[(ref, r, p)])
    return {p: (d["delta_p"][p], min(v), max(v)) for p, v in by_p.items()}


def ied_counts(recount_csv: Path = cf.RECOUNT_CSV) -> dict:
    """Evaluable IED epochs per patient: IED epochs summed over the states in which the patient is evaluable."""
    out = {}
    for r in csv.DictReader(recount_csv.open()):
        states = [s for s in ("wake", "sleep") if s in r["evaluable_states_after"]]
        out[r["eeg_id"]] = sum(int(r[f"n_ied_{s}"]) for s in states)
    return out


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def forest_svg(deltas: dict, counts: dict, delta_hat: float, ci90, ci95, title: str,
               axis: str = "Δ within-patient AUROC (B2 + FC − B2)") -> str:
    rows = sorted(deltas, key=lambda p: (counts[p], p))
    lo = min(v[1] for v in deltas.values())
    hi = max(v[2] for v in deltas.values())
    step = 0.02
    xmin = min(-SESOI - step, np.floor(lo / step) * step)
    xmax = max(SESOI + step, np.ceil(hi / step) * step)
    left, right, top, rh = 120, 24, 84, 15
    plot_w = 520
    n = len(rows)
    summary_y = top + (n + 1.2) * rh
    height = int(summary_y + 78)
    width = left + plot_w + right

    def x(v: float) -> float:
        return left + (v - xmin) / (xmax - xmin) * plot_w

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" role="img" '
         f'aria-label="{_esc(title)}">',
         f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
         f'<text x="{left}" y="22" font-size="14" font-weight="600" fill="{INK}">{_esc(title)}</text>',
         f'<text x="{left}" y="40" font-size="11" fill="{INK2}">One row per evaluable IED-positive patient '
         f'(n = {n}), ordered by evaluable IED epochs.</text>',
         f'<text x="{left}" y="54" font-size="11" fill="{INK2}">Dot: mean over 10 repeats; line: repeat minimum '
         f'to maximum.</text>']
    ticks = np.round(np.arange(xmin, xmax + 1e-9, step), 4) + 0.0
    every = 1 if len(ticks) <= 9 else 2
    for j, t in enumerate(ticks):
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{summary_y + 12}" stroke="{GRID}" '
                 f'stroke-width="1"/>')
        if round(t / step) % every == 0:
            lab = "0" if abs(t) < 1e-9 else f"{t:+.2f}".replace("-", "−")
            s.append(f'<text x="{x(t):.1f}" y="{summary_y + 28}" font-size="11" fill="{INK2}" '
                     f'text-anchor="middle">{lab}</text>')
    for v in (-SESOI, SESOI):
        s.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 6}" y2="{summary_y + 12}" stroke="{INK2}" '
                 f'stroke-width="1" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{x(SESOI) + 4:.1f}" y="{top - 10}" font-size="11" fill="{INK2}">SESOI ±0.03</text>')
    s.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top - 6}" y2="{summary_y + 12}" stroke="{INK}" '
             f'stroke-width="1"/>')
    s.append(f'<text x="{left - 8}" y="{top - 10}" font-size="11" fill="{INK2}" text-anchor="end">IED epochs</text>')
    for i, p in enumerate(rows):
        y = top + i * rh + rh / 2
        d, a, b = deltas[p]
        s.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="{INK2}" '
                 f'text-anchor="end">{counts[p]}</text>')
        s.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{SERIES}" '
                 f'stroke-width="2" stroke-linecap="round" stroke-opacity="0.45"/>')
        s.append(f'<circle cx="{x(d):.1f}" cy="{y:.1f}" r="4" fill="{SERIES}" stroke="#ffffff" stroke-width="2">'
                 f'<title>{counts[p]} IED epochs: Δp {d:+.4f} (repeats {a:+.4f} to {b:+.4f})</title></circle>')
    y = summary_y
    s.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" font-weight="600" fill="{INK}" '
             f'text-anchor="end">Overall (M1)</text>')
    s.append(f'<line x1="{x(ci95[0]):.1f}" x2="{x(ci95[1]):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{INK}" '
             f'stroke-width="2"/>')
    s.append(f'<rect x="{x(ci90[0]):.1f}" y="{y - 4:.1f}" width="{x(ci90[1]) - x(ci90[0]):.1f}" height="8" '
             f'rx="2" fill="{INK}"/>')
    s.append(f'<circle cx="{x(delta_hat):.1f}" cy="{y:.1f}" r="5" fill="#ffffff" stroke="{INK}" stroke-width="2"/>')
    s.append(f'<text x="{left + plot_w / 2:.1f}" y="{summary_y + 46}" font-size="11" fill="{INK2}" '
             f'text-anchor="middle">{_esc(axis)}</text>')
    s.append(f'<text x="{left}" y="{summary_y + 66}" font-size="11" fill="{INK2}">Overall Δ = {delta_hat:+.4f}; bar: M1 '
             f'90% [{ci90[0]:+.4f}, {ci90[1]:+.4f}]; line: M1 95% [{ci95[0]:+.4f}, {ci95[1]:+.4f}].</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def write_tables(out: Path = OUT) -> None:
    (out / "tables").mkdir(parents=True, exist_ok=True)
    with (out / "tables" / "contrasts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["learner", "contrast", "role", "delta_hat", "n_e", "m1_90_lo", "m1_90_hi", "m1_95_lo", "m1_95_hi",
                    "label_m1", "boot_90_lo", "boot_90_hi", "label_bootstrap", "nb_90_lo", "nb_90_hi",
                    "label_nb_descriptive", "sd_delta_p"])
        for kind in ("hgb", "lr"):
            s = json.loads((out / f"summary_{kind}.json").read_text())
            nb = cf.summarise(out, kind, "nb", s["gates"]["pc3"])["contrasts"]
            for c, v in s["contrasts"].items():
                iv, ivn = v["intervals"], nb[c]["intervals"]
                w.writerow([kind, c, v["role"], v["delta_hat"], v["n_e"], *iv["0.90"], *iv["0.95"], v["label"],
                            *iv["bootstrap_0.90"], v["label_sensitivity_bootstrap"], *ivn["0.90"], nb[c]["label"],
                            v["heterogeneity"]["sd_delta_p"]])
    with (out / "tables" / "model_auroc.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["learner", "model", "mean_within_patient_auroc", "n_patients"])
        for kind in ("hgb", "lr"):
            auroc, _ = per_repeat(load_units(kind, out))
            by = defaultdict(lambda: defaultdict(list))
            for (m, _r, p), a in auroc.items():
                by[m][p].append(a)
            for m in cf.L.MODELS:
                if m in by:
                    w.writerow([kind, m, float(np.mean([np.mean(v) for v in by[m].values()])), len(by[m])])


ENGINEERING = [("b0", "b2_fc"), ("b0", "b2")]


def engineering_contrasts(kind: str, out: Path = OUT) -> dict:
    """B0 vs the multichannel models, paired within patient (Josue 2026-10-16: a secondary engineering finding
    that needs paired uncertainty before interpretation). Not pre-specified as a contrast: M1 and bootstrap
    intervals and R6 heterogeneity are reported; no taxonomy label is assigned."""
    auroc, folds = per_repeat(load_units(kind, out))
    res = {}
    for a, b in ENGINEERING:
        d = est.delta_summary(auroc, a, b, folds)
        res[f"{a}_vs_{b}"] = {"delta_hat": d["delta_hat"], "n_e": d["n_e"], "intervals": cf.intervals(d, "m1"),
                              "heterogeneity": cf.heterogeneity(d)}
    return res


def main() -> None:
    counts = ied_counts()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    for kind, name in (("hgb", "HGB, primary learner"), ("lr", "L2 logistic regression, secondary learner")):
        s = json.loads((OUT / f"summary_{kind}.json").read_text())["contrasts"]["b2_fc_vs_b2"]
        deltas = patient_deltas(load_units(kind))
        assert abs(np.mean([v[0] for v in deltas.values()]) - s["delta_hat"]) < 1e-12
        svg = forest_svg(deltas, counts, s["delta_hat"], s["intervals"]["0.90"], s["intervals"]["0.95"],
                         f"Per-patient gain from functional connectivity ({name})")
        (OUT / "figures" / f"forest_b2fc_vs_b2_{kind}.svg").write_text(svg)
        with (OUT / "tables" / f"patient_delta_{kind}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["eeg_id", "evaluable_ied_epochs", "delta_p", "repeat_min", "repeat_max"])
            for p in sorted(deltas, key=lambda q: (counts[q], q)):
                w.writerow([p, counts[p], *deltas[p]])
    write_tables()
    with (OUT / "tables" / "b0_paired.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["learner", "contrast", "status", "delta_hat", "n_e", "m1_90_lo", "m1_90_hi", "m1_95_lo",
                    "m1_95_hi", "boot_95_lo", "boot_95_hi", "sd_delta_p", "n_above_0.03", "n_below_-0.03"])
        for kind in ("hgb", "lr"):
            for c, v in engineering_contrasts(kind).items():
                iv, h = v["intervals"], v["heterogeneity"]
                w.writerow([kind, c, "secondary engineering finding, not pre-specified; no label", v["delta_hat"],
                            v["n_e"], *iv["0.90"], *iv["0.95"], *iv["bootstrap_0.95"], h["sd_delta_p"],
                            h["n_above_sesoi"], h["n_below_minus_sesoi"]])
    e = engineering_contrasts("hgb")["b0_vs_b2_fc"]
    deltas = patient_deltas(load_units("hgb"), "b0", "b2_fc")
    (OUT / "figures" / "forest_b0_vs_b2fc_hgb.svg").write_text(
        forest_svg(deltas, counts, e["delta_hat"], e["intervals"]["0.90"], e["intervals"]["0.95"],
                   "Per-patient difference, single-channel B0 vs B2 + FC (HGB; engineering finding)",
                   axis="Δ within-patient AUROC (B0 − (B2 + FC))"))
    print("wrote", *sorted(str(q.relative_to(ROOT)) for q in OUT.glob("*/*") if q.suffix in (".svg", ".csv")))


if __name__ == "__main__":
    main()
