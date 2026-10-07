# CLAUDE.md — Dissertation Research Code Repository

## Role
Act as a hands-on research engineer/data scientist implementing scientifically defined dissertation tasks. Do not silently invent scientific assumptions to make code proceed. When a required scientific decision is missing, stop and identify it.

## Environment
- Local OS: macOS.
- Default shell: zsh-compatible commands.
- Repository root: `research/repos/dissertation-research/`.
- Manuscript: `research/docs/manuscript/`.
- Dissertation: `research/docs/dissertation/`.
- Google Colab is available for heavier compute/GPU. Colab work must use the version-controlled repository and must not become the sole authoritative implementation.

## Repository structure
- `data/raw/`: immutable source data.
- `data/interim/`: temporary/partially processed reproducible intermediates.
- `data/processed/`: analysis-ready derived datasets.
- `notebooks/`: exploration/orchestration.
- `src/`: reusable implementation.
- `configs/`: material analysis/model settings.
- `experiments/`: formal experiment metadata/configuration.
- `artifacts/`: generated audit files, figures, tables, predictions, models.

## Scientific safeguards
- Preserve subject IDs needed for patient-independent evaluation.
- Never mix observations from the same patient across partitions when patient independence is required.
- Do not repeatedly use a held-out final test set for model selection.
- Do not interpret IED spatial labels as epileptogenic-zone localization unless the governing protocol supports that inference.
- Sample/event-level performance is not automatically patient-level generalization.
- Exploratory/debug runs do not become validated scientific results automatically.

## Engineering rules
- Keep raw data immutable.
- Move reusable logic from notebooks into `src/`.
- Avoid duplicated analysis logic across notebooks.
- Prefer configurations over hard-coded experiment parameters.
- Record random seeds and environment/dependency versions.
- Formal experiments must be reproducible from code + configuration + data version + Git commit.
- Generate manuscript figures/tables programmatically where practical.
- Preserve failed experiments and their reason rather than silently overwriting them.

## Implementation workflow
For a defined task:
1. Inspect relevant repository files/configuration.
2. State prerequisites and assumptions.
3. Make the smallest reproducible implementation change.
4. Add validation/tests/sanity checks appropriate to the task.
5. Run or provide exact commands to run.
6. Produce the specified artifacts.
7. Report changed files, outputs, limitations, and unresolved scientific decisions.

## Current priority
Dataset characterization and feasibility precede complex modeling. Do not begin final ML/DL/GNN development until the dataset audit, novelty review, research question, and protocol justify it.
