# EXP-003 run sheet: pilot-only pipeline debugging, baseline ceiling check and window-validity diagnostic

**Status:** DRAFT. It runs only after Josue approves the implementation defaults in `protocol/2026-10-12-pipeline-implementation-spec.md` §3. Registered in `07` as exploratory/debug: never a scientific result, and never evidence about H-1.

## Scope
- **Recordings:** the 8 frozen pilot recordings only (`artifacts/protocol/f3/pilot_subset.csv`, SHA-256 `4645df56…0bec29`). Four are evaluable after F-6: DA00100U (wake), DA00102U (sleep), DA00103A (sleep), DA00103T (wake and sleep).
- **Guards:** `src/features/extract.py` refuses non-pilot recordings without `--allow-non-pilot`, and `run_ladder.py pilot` reads pilot caches only. No confirmatory recording is touched.

## Steps, in order (on the Mac, from the repository root)
1. **Features.** Run `uv run python src/features/extract.py --root data/raw/vepiset/extracted/opensource-dataset --ids-from artifacts/protocol/f3/pilot_subset.csv --out data/interim/features/salience2s`.
   - It writes a per-recording cache under `data/interim/`, which is never committed.
   - The cache holds B1, B2 topography, ADJ, FC and FC_surr (k = 5), plus each window's peak salience and marker diagnostic.
2. **z\*.** Run `uv run python src/pipeline/run_ladder.py zstar`.
   - It sets the B2 salience threshold once, label-blind: the 95th percentile of the window-peak z values (spec §3).
   - It is written to `artifacts/pilot/zstar.json` and committed before step 3.
3. **Pilot ladder.** Run `uv run python src/pipeline/run_ladder.py pilot`.

## Fixed in advance
- **Split:** leave-one-recording-out over the 8 pilot recordings, so each fold trains on 7 and tests 1. The locked 4-fold inner-CV validity rule cannot be met on 8 recordings, so there is no nested tuning.
- **Learner:** HGB with the plan-A fixed configuration (learning_rate 0.05, max_leaf_nodes 15, min_samples_leaf 100, l2_regularization 1, max_iter 300, no early stopping) and R9 weights. Seed 20261010.
- **Models:** B0, B1, B2, B2 + ADJ, B2 + FC_surr, B2 + FC, PC-1 FC-only.

## Outputs (aggregate, pilot only; committed)
- `artifacts/pilot/pilot_auroc.csv`: the within-patient, same-state AUROC for each evaluable pilot patient and model.
- `artifacts/pilot/pilot_summary.json`:
  - the B2 level (ceiling check), mean AUROC per model, epoch counts and runtime;
  - the **window-validity share**: IED windows that contain an annotated marker. It is a diagnostic only and is never used to tune the window rule.
- Extraction runtime per recording, used to plan EXP-004 compute.

## What this run cannot do
- With 4 evaluable patients and 7-recording training sets, model differences here are debugging signals, not estimates.
- Nothing in the protocol changes because of these numbers without an RDR (`07` EXP-003).
