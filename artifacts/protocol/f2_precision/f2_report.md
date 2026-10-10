# F-2 — Precision feasibility of the primary estimand (2026-10-10)

Workstream 03. Status: **WORKING evidence for a protocol decision.** Not a power guarantee, not a result about FC. Specification and decision rule were fixed in `06` v0.3 §6 before the run.

## Verdict (pre-specified rule)
**MARGINAL.** The ±0.03 equivalence margin is comfortably supportable if per-patient FC gains are moderately homogeneous, and is not reliably supportable if they are strongly heterogeneous. Issue 3 stays REVISED – CONDITIONAL.

| Rule (fixed before running) | Value | Threshold |
|---|---|---|
| Central scenario, worst cell: P(equivalence declared \| true Δ = 0) | 0.997 | ≥ 0.80 |
| Pessimistic scenario, worst cell | 0.249 | ≥ 0.50 |

Central passes; pessimistic fails, so the rule gives MARGINAL, not SUPPORTABLE and not NOT SUPPORTABLE.

## What was simulated
- Inputs: per-recording state × class counts in the committed `artifacts/dataset_audit/subject_state_matrix.csv` only. No signal, features or models.
- Estimand as decided in `06` issue 2: equal-weight mean over eligible IED-positive patients of within-patient, same-state ΔAUROC (IED vs same-state non-IED epochs, per state, combined by IED count). 46 patients eligible before removals.
- Scenario grid: baseline mean within-patient AUROC 0.85 / 0.90 / 0.95; patient-to-patient SD of the FC gain τ_g 0.10 / 0.25 / 0.50 (probit scale; at AUROC 0.90 these are roughly 0.012 / 0.03 / 0.06 on the AUROC scale); correlation of the two models' epoch scores r 0.80 / 0.90; CV refit noise σ_f 0 / 0.01 / 0.02 shared by fold; pilot 0 / 6 / 8 / 10 recordings (0 / 4 / 5 / 6 IED); extra QC loss 0 / 5 / 10 IED recordings; true Δ 0 or 0.03. 1,000 replicates per scenario (1,296 scenarios).
- "Honest" 90% CI = estimate ± 1.645 × the across-replicate SD (includes refit noise). The naive patient-level interval (what a patient bootstrap approximates) covered the true value in 0.71–0.93 of replicates across the grid, against a nominal 0.90; it is worst with fold-shared refit noise and near the 0.95 ceiling. So the honest interval is the one used for the verdict, and the real analysis needs the CV-variance correction already required in `06` issue 10.

## Main numbers (true Δ = 0; honest 90% CI)
| Scenario | Eligible patients | 90% CI half-width | P(equivalence) |
|---|---|---|---|
| Central (AUROC 0.90, τ_g 0.25, r 0.90, σ_f 0.01), pilot 6–8, QC loss 0–5 | 37–42 | 0.010–0.011 | 0.997–1.000 |
| τ_g 0.50, AUROC 0.90, r 0.90, σ_f 0.01, pilot 8, QC 5 | 37 | 0.018 | 0.72 |
| Pessimistic, AUROC 0.90, pilot 8, QC 10 | 33 | 0.020 | 0.57 |
| Pessimistic, AUROC 0.85, pilot 8, QC 10 | 33 | 0.025 | 0.25 |
| Pessimistic, AUROC 0.95, pilot 8, QC 10 | 33 | 0.013 | 0.96 |

If the true gain is 0.03, P(95% CI lower bound > 0) is ≥ 0.99 when τ_g ≤ 0.25 and 0.83–0.88 when τ_g = 0.50 (AUROC 0.90, pilot 8, QC 5).

**What drives precision:** the patient-to-patient spread of the FC gain, then QC loss, then baseline AUROC. Pilot size matters least.

**Model-light tipping point.** Equivalence at ±0.03 is declared with probability ≥ 0.80 when the SD of per-patient ΔAUROC (heterogeneity + sampling + refit noise) is ≤ about 0.062 with 37 patients (0.066 with 42); with probability ≥ 0.50 when it is ≤ about 0.079 (0.084). This SD is observable only once FC is run.

## Pilot size (issue 10)
- Central scenario, QC loss 0: P(equivalence) 1.000 (6), 0.999 (8), 0.999 (10). The pre-set rule therefore allows up to 10.
- Where precision is tight (pessimistic, AUROC 0.90, QC 0): 0.763 (6), 0.754 (8), 0.721 (10). Each extra IED recording moved into the pilot costs about 0.01 of equivalence probability.
- Recommendation: **8 recordings (5 IED, 3 IED-free)**, within Josue's 6–8 target. Gives a slightly better ceiling and debugging check than 6 at a cost of about 0.01. Not 10.

## Sensitivity added after the run (not part of the decision rule)
Within-patient epochs are temporally clustered, so the simulation's independent-epoch assumption is optimistic. Shrinking simulated epoch counts to one third (`clustering_sensitivity/`): central worst cell 0.964 (still passes); pessimistic worst cell 0.162. The verdict is unchanged (MARGINAL), but the heterogeneous-gain cells get worse, e.g. τ_g 0.50, AUROC 0.90, pilot 8, QC 5: P(equivalence) 0.47–0.65.

## Interpretation
- A bounded null at ±0.03 is a realistic outcome if FC's effect is fairly consistent across patients. If FC helps a lot in some patients and not at all in others (τ_g ≈ 0.5), the likely null outcome is **inconclusive**, not equivalent. The five-outcome taxonomy already reports that honestly.
- The biggest lever under our control is QC loss: the no-bad-channel rule (issue 12) costs precision for every IED recording it removes. F-4 tells us how many.
- This does not estimate the FC effect or its heterogeneity; those are unknown until the confirmatory run.

## Options for issue 3 (Josue's decision)
1. **Keep SESOI 0.03 (recommended).** State in the protocol that the bounded-null reading is credible only if gains are not strongly heterogeneous, and pre-register reporting of the per-patient Δ spread (with a prediction interval) so an inconclusive result is still informative. Keeps the SESOI justified by meaning, not by precision.
2. Widen SESOI to 0.04. Equivalence becomes likely even in most pessimistic cells, but the change would be driven by precision, which the lens review warned against.
3. Keep 0.03 and revisit after F-4: if QC removes ≤ 5 IED recordings the central picture holds; if it removes ~10, decide then.

## Provenance
- Code: `src/design/f2_precision_sim.py`, commit 0c477a6 on branch `claude/project-thread-0g1ch3` (repo `eng-rodriguez/dissertation-research`). Run in the cloud container: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, seed 20261010, 1,000 replicates.
- Outputs (repo): `artifacts/protocol/f2_precision/{f2_grid.csv, f2_summary.json}`; clustering sensitivity in `artifacts/protocol/f2_precision/clustering_sensitivity/` (`--count-scale 0.3333`).
- Reproduce: `python src/design/f2_precision_sim.py` and `python src/design/f2_precision_sim.py --count-scale 0.3333 --out artifacts/protocol/f2_precision/clustering_sensitivity` from the repo root.
