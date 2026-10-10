"""Inner CV splits for nested tuning (06 §4.1, approved 2026-10-10) and their validity rule.

Inner folds are not stored in the frozen F-3 manifest. They are derived deterministically inside
each outer training set with the same stratified dealing as the outer folds (f3.deal_folds), with
seed = repeat seed * 10 + outer fold. This module derives them and applies the inner-fold validity
rule; `--check` runs the rule on the committed manifest before any model exists (counts only).

Validity rule (06 §4.1). An inner split of an outer training set is valid only if:
  1. every inner validation fold holds >= MIN_E_VALIDATION evaluable IED-positive patients (group E),
     so the estimand-aligned selection metric is defined in every fold;
  2. every inner training part holds >= 1 E patient evaluable in wake and >= 1 evaluable in sleep,
     so state-balanced training weights are defined;
  3. inner validation folds partition the outer training set (each recording exactly once) and
     contain no outer-test recording.
If 4 inner folds fail the rule for an outer fold, that outer fold uses 3; if 3 also fail, the run
stops and the failure is recorded as a protocol deviation for Josue. No other fallback.

Usage (from the repository root):
    python src/design/inner_folds.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3_pilot_and_folds as f3  # noqa: E402

N_INNER = 4
N_INNER_FALLBACK = 3
MIN_E_VALIDATION = 5


def inner_seed(repeat_seed: int, outer_fold: int) -> int:
    return repeat_seed * 10 + outer_fold


def violations(train_ids, inner: dict[str, int], n_inner: int, recs: dict[str, dict], test_ids=()) -> list[str]:
    out = []
    if set(inner) != set(train_ids) or set(inner) & set(test_ids):
        out.append("inner folds do not partition the outer training set")
    for j in range(n_inner):
        val = [i for i in train_ids if inner[i] == j]
        trn = [i for i in train_ids if inner[i] != j]
        n_e = sum(recs[i]["group"] == "E" for i in val)
        if n_e < MIN_E_VALIDATION:
            out.append(f"inner fold {j}: {n_e} evaluable patients in validation (< {MIN_E_VALIDATION})")
        for st in ("wake", "sleep"):
            if not any(st in recs[i]["evaluable_states"] for i in trn):
                out.append(f"inner fold {j}: no {st}-evaluable patient in training")
    return out


def derive(train_ids, recs: dict[str, dict], repeat_seed: int, outer_fold: int, test_ids=()) -> tuple[dict, int, list]:
    """Return (inner fold per recording, number of inner folds used, violations of the final attempt)."""
    strata = {i: f"{recs[i]['group']}|{recs[i]['coverage']}" for i in train_ids}
    seed = inner_seed(repeat_seed, outer_fold)
    for n in (N_INNER, N_INNER_FALLBACK):
        inner, _ = f3.deal_folds(train_ids, strata, n, "inner", seed)
        bad = violations(train_ids, inner, n, recs, test_ids)
        if not bad:
            return inner, n, []
    return inner, n, bad


def check(manifest: Path, qc_csv: Path, state_matrix: Path) -> dict:
    recs = f3.load_recordings(qc_csv, state_matrix)
    rows = [r for r in csv.DictReader(manifest.open()) if r["analysis_set"] == "primary"]
    by = defaultdict(list)
    for r in rows:
        by[(int(r["repeat"]), int(r["seed"]))].append(r)
    results, min_e, failures, fallbacks = [], None, [], 0
    for (rep, seed), rr in sorted(by.items()):
        for k in range(f3.N_FOLDS):
            test = [r["eeg_id"] for r in rr if int(r["outer_fold"]) == k]
            train = sorted(r["eeg_id"] for r in rr if int(r["outer_fold"]) != k)
            inner, n, bad = derive(train, recs, seed, k, test)
            e_counts = [sum(recs[i]["group"] == "E" for i in train if inner[i] == j) for j in range(n)]
            min_e = min(e_counts) if min_e is None else min(min_e, min(e_counts))
            fallbacks += n != N_INNER
            if bad:
                failures.append({"repeat": rep, "outer_fold": k, "violations": bad})
            results.append({"repeat": rep, "outer_fold": k, "n_inner": n, "e_per_inner_validation": e_counts})
    return {"rule": {"min_e_validation": MIN_E_VALIDATION, "n_inner": N_INNER, "fallback": N_INNER_FALLBACK},
            "n_outer_splits": len(results), "n_fallback_to_3": fallbacks, "n_failures": len(failures),
            "min_e_in_any_inner_validation_fold": min_e, "failures": failures, "splits": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="apply the validity rule to the committed manifest")
    ap.add_argument("--manifest", default="artifacts/protocol/f3/fold_manifest.csv")
    ap.add_argument("--qc", default="artifacts/dataset_audit/channel_qc.csv")
    ap.add_argument("--state-matrix", default="artifacts/dataset_audit/subject_state_matrix.csv")
    ap.add_argument("--out", default="artifacts/protocol/f3/inner_fold_check.json")
    a = ap.parse_args()
    if not a.check:
        ap.error("nothing to do; use --check")
    s = check(Path(a.manifest), Path(a.qc), Path(a.state_matrix))
    Path(a.out).write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps({k: s[k] for k in ("n_outer_splits", "n_fallback_to_3", "n_failures",
                                        "min_e_in_any_inner_validation_fold")}, indent=2))
    sys.exit(1 if s["n_failures"] else 0)


if __name__ == "__main__":
    main()
