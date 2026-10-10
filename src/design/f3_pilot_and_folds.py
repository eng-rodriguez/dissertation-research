"""F-3: draw the frozen pilot subset and the patient-grouped fold manifest (06 issue 10).

Reads only committed aggregate tables:
  artifacts/dataset_audit/channel_qc.csv          (F-4 primary QC eligibility, commit a458458)
  artifacts/dataset_audit/subject_state_matrix.csv (per-recording epoch counts by class and state)
No signal, feature, label-level detail or model result is used.

Unit: recording, treated as patient (one recording per patient is SOURCE_REPORTED in 05).

Groups:
  E  IED-positive and evaluable for the primary estimand (>=5 IED and >=5 non-IED epochs in a
     common wake/sleep state; 06 §2, issue 2)
  N  IED-positive, not evaluable
  F  IED-free
Coverage: for E, the states in which it is evaluable; otherwise the states with >=5 epochs
(wake, sleep, wake+sleep, none).

Pilot (06 issue 10, approved 2026-10-10): 5 recordings from E and 3 from F, drawn only from
recordings eligible under the F-4 primary QC rule. Within each group the count is allocated
to coverage strata by largest remainder, then recordings are taken in a seeded hash order.
The pilot can never return to the confirmatory cohort.

Fold manifest: 5-fold outer CV repeated 10 times, recording-grouped, stratified by group x
coverage. Primary-set recordings (QC-eligible, not pilot) are dealt first, so their folds do
not depend on the QC-ineligible recordings, which are then dealt into the same folds and
marked sensitivity_only (issue 12 interpolation sensitivity). Inner CV splits are not frozen
here; training code derives them inside each outer training set with deal_folds().

Randomness is SHA-256 ordering of "<purpose>:<seed>:<eeg_id>", so the draw does not depend on
the numpy or Python version. Seeds are fixed in this file before the run.

Usage (from the repository root):
    python src/design/f3_pilot_and_folds.py            # draw once; refuses to overwrite
    python src/design/f3_pilot_and_folds.py --verify   # recompute and compare with committed files

Writes artifacts/protocol/f3/{pilot_subset.csv, fold_manifest.csv, f3_summary.json, SHA256SUMS}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

MASTER_SEED = 20261010
PILOT_SEED = MASTER_SEED
REPEAT_SEEDS = tuple(MASTER_SEED * 100 + r for r in range(1, 11))  # 2026101001..2026101010
N_FOLDS = 5
PILOT_SIZE = {"E": 5, "F": 3}
MIN_EPOCHS = 5
STATES = ("wake", "sleep")
PILOT_FIELDS = ["eeg_id", "group", "coverage"]
MANIFEST_FIELDS = ["repeat", "seed", "eeg_id", "analysis_set", "group", "coverage", "outer_fold"]
OUTPUTS = ("pilot_subset.csv", "fold_manifest.csv")


def _hash_key(purpose: str, seed: int, eeg_id: str) -> str:
    return hashlib.sha256(f"{purpose}:{seed}:{eeg_id}".encode()).hexdigest()


def seeded_order(ids, purpose: str, seed: int) -> list[str]:
    return sorted(ids, key=lambda i: _hash_key(purpose, seed, i))


def load_recordings(qc_csv: Path, state_matrix: Path) -> dict[str, dict]:
    """Per recording: group, coverage, QC primary eligibility and epoch counts."""
    sm = {r["eeg_id"]: r for r in csv.DictReader(state_matrix.open())}
    recs = {}
    for q in csv.DictReader(qc_csv.open()):
        rid = q["eeg_id"]
        s = {k: int(v) for k, v in sm[rid].items() if k != "eeg_id"}
        n_ied = sum(s[f"ied_{k}"] for k in ("wake", "sleep", "mixed", "unlabelled"))
        evaluable = [st for st in STATES
                     if s[f"ied_{st}"] >= MIN_EPOCHS and s[f"all_{st}"] - s[f"ied_{st}"] >= MIN_EPOCHS]
        present = [st for st in STATES if s[f"all_{st}"] >= MIN_EPOCHS]
        group = "E" if evaluable else ("N" if n_ied else "F")
        # consistency with the F-4 outputs (same definitions, computed independently)
        if (group == "E") != (q["pre_qc_eligible_ied"] == "True") or (n_ied > 0) != (q["ied_positive"] == "True"):
            raise ValueError(f"{rid}: group disagrees with channel_qc.csv")
        recs[rid] = {
            "group": group,
            "coverage": "+".join(evaluable if evaluable else present) or "none",
            "qc_primary": q["primary_eligible"] == "True",
            "n_ied": n_ied,
            "n_epochs": sum(s[f"all_{k}"] for k in ("wake", "sleep", "mixed", "unlabelled")),
            "evaluable_states": evaluable,
        }
    missing = set(sm) - set(recs)
    if missing:
        raise ValueError(f"recordings in state matrix but not in channel_qc.csv: {sorted(missing)}")
    return recs


def largest_remainder(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Allocate n across strata proportionally to sizes; ties broken by stratum name."""
    total = sum(sizes.values())
    if n > total:
        raise ValueError(f"cannot draw {n} from {total}")
    quota = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def draw_pilot(recs: dict[str, dict], seed: int = PILOT_SEED) -> tuple[list[dict], dict]:
    pilot, allocation = [], {}
    for group, n in PILOT_SIZE.items():
        strata = defaultdict(list)
        for rid, r in recs.items():
            if r["qc_primary"] and r["group"] == group:
                strata[r["coverage"]].append(rid)
        alloc = largest_remainder({k: len(v) for k, v in strata.items()}, n)
        allocation[group] = {"available": {k: len(v) for k, v in sorted(strata.items())}, "drawn": alloc}
        for cov in sorted(strata):
            for rid in seeded_order(strata[cov], f"pilot:{group}:{cov}", seed)[: alloc[cov]]:
                pilot.append({"eeg_id": rid, "group": group, "coverage": cov})
    return sorted(pilot, key=lambda r: r["eeg_id"]), allocation


def deal_folds(ids, strata: dict[str, str], n_folds: int, purpose: str, seed: int, start: int = 0) -> tuple[dict, int]:
    """Stratified grouped dealing: strata in name order, seeded order within, one shared pointer.

    Each fold gets floor or ceil of every stratum's count, and overall fold sizes differ by at
    most one. Returns (assignment, next pointer) so a later set can continue the same deal.
    """
    by = defaultdict(list)
    for i in ids:
        by[strata[i]].append(i)
    out, p = {}, start
    for s in sorted(by):
        for i in seeded_order(by[s], f"{purpose}:{s}", seed):
            out[i] = p % n_folds
            p += 1
    return out, p


def build_manifest(recs: dict[str, dict], pilot_ids: set[str]) -> list[dict]:
    primary = [i for i, r in recs.items() if r["qc_primary"] and i not in pilot_ids]
    sens_only = [i for i, r in recs.items() if not r["qc_primary"]]
    if pilot_ids & set(sens_only):
        raise ValueError("pilot contains a QC-ineligible recording")
    strata = {i: f"{r['group']}|{r['coverage']}" for i, r in recs.items()}
    rows = []
    for rep, seed in enumerate(REPEAT_SEEDS, start=1):
        fold_p, p = deal_folds(primary, strata, N_FOLDS, "outer", seed)
        fold_s, _ = deal_folds(sens_only, strata, N_FOLDS, "outer", seed, start=p)
        for ids, folds, aset in ((primary, fold_p, "primary"), (sens_only, fold_s, "sensitivity_only")):
            for i in sorted(ids):
                rows.append({"repeat": rep, "seed": seed, "eeg_id": i, "analysis_set": aset,
                             "group": recs[i]["group"], "coverage": recs[i]["coverage"], "outer_fold": folds[i]})
    return rows


def to_csv(rows: list[dict], fields: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def balance(rows: list[dict], recs: dict[str, dict]) -> dict:
    """Per repeat and fold: recordings by group, evaluable patients by state, IED epochs (primary set)."""
    out = {}
    for rep in sorted({r["repeat"] for r in rows}):
        folds = []
        for k in range(N_FOLDS):
            ids = [r["eeg_id"] for r in rows if r["repeat"] == rep and r["outer_fold"] == k and r["analysis_set"] == "primary"]
            g = Counter(recs[i]["group"] for i in ids)
            folds.append({
                "n": len(ids), "E": g["E"], "N": g["N"], "F": g["F"],
                "E_evaluable_wake": sum("wake" in recs[i]["evaluable_states"] for i in ids),
                "E_evaluable_sleep": sum("sleep" in recs[i]["evaluable_states"] for i in ids),
                "ied_epochs": sum(recs[i]["n_ied"] for i in ids),
                "sensitivity_only_added": sum(1 for r in rows if r["repeat"] == rep and r["outer_fold"] == k
                                              and r["analysis_set"] == "sensitivity_only"),
            })
        out[str(rep)] = folds
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(qc_csv: Path, state_matrix: Path, out: Path, verify: bool = False) -> dict:
    recs = load_recordings(qc_csv, state_matrix)
    pilot, allocation = draw_pilot(recs)
    pilot_ids = {r["eeg_id"] for r in pilot}
    manifest = build_manifest(recs, pilot_ids)
    files = {"pilot_subset.csv": to_csv(pilot, PILOT_FIELDS), "fold_manifest.csv": to_csv(manifest, MANIFEST_FIELDS)}
    hashes = {k: sha256(v) for k, v in files.items()}

    if verify:
        found = {k: sha256((out / k).read_bytes()) for k in OUTPUTS}
        ok = found == hashes
        print(json.dumps({"verify": "MATCH" if ok else "MISMATCH", "recomputed": hashes, "committed": found}, indent=2))
        return {"verify_ok": ok, "sha256": hashes}

    if any((out / k).exists() for k in OUTPUTS):
        raise FileExistsError(f"{out} already holds a frozen pilot/manifest; use --verify, never redraw")

    primary = [r for r in manifest if r["repeat"] == 1 and r["analysis_set"] == "primary"]
    pc = Counter(recs[r["eeg_id"]]["group"] for r in primary)
    summary = {
        "status": "FROZEN at creation. The pilot never returns to the confirmatory cohort (06 issue 10).",
        "pilot": {"n": len(pilot), "by_group": dict(Counter(r["group"] for r in pilot)),
                  "allocation": allocation, "ids": sorted(pilot_ids)},
        "primary_set": {"n": len(primary), "E_evaluable": pc["E"], "N_ied_not_evaluable": pc["N"], "F_ied_free": pc["F"],
                        "E_evaluable_wake": sum("wake" in recs[r["eeg_id"]]["evaluable_states"] for r in primary),
                        "E_evaluable_sleep": sum("sleep" in recs[r["eeg_id"]]["evaluable_states"] for r in primary)},
        "sensitivity_only": sorted(i for i, r in recs.items() if not r["qc_primary"]),
        "folds": {"n_folds": N_FOLDS, "n_repeats": len(REPEAT_SEEDS), "repeat_seeds": list(REPEAT_SEEDS),
                  "stratified_by": "group (E/N/F) x coverage", "unit": "recording (= patient)",
                  "inner_cv": "not frozen here; derived inside each outer training set with deal_folds()"},
        "balance": balance(manifest, recs),
        "definitions": {"E": f">=1 of wake/sleep with >={MIN_EPOCHS} IED and >={MIN_EPOCHS} non-IED epochs",
                        "N": "IED-positive, not E", "F": "IED-free",
                        "coverage": f"E: evaluable states; N/F: states with >={MIN_EPOCHS} epochs",
                        "randomness": "SHA-256 order of '<purpose>:<seed>:<eeg_id>'"},
        "sha256": hashes,
        "inputs_sha256": {str(qc_csv): sha256(qc_csv.read_bytes()), str(state_matrix): sha256(state_matrix.read_bytes())},
        "provenance": {"script": "src/design/f3_pilot_and_folds.py", "git_commit": git_commit(),
                       "pilot_seed": PILOT_SEED, "python": platform.python_version()},
    }
    out.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        (out / k).write_bytes(v)
    (out / "SHA256SUMS").write_text("".join(f"{h}  {k}\n" for k, h in hashes.items()))
    (out / "f3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qc", default="artifacts/dataset_audit/channel_qc.csv")
    ap.add_argument("--state-matrix", default="artifacts/dataset_audit/subject_state_matrix.csv")
    ap.add_argument("--out", default="artifacts/protocol/f3")
    ap.add_argument("--verify", action="store_true", help="recompute and compare with the committed files")
    args = ap.parse_args()
    s = run(Path(args.qc), Path(args.state_matrix), Path(args.out), args.verify)
    if args.verify:
        sys.exit(0 if s["verify_ok"] else 1)
    print(json.dumps({k: s[k] for k in ("pilot", "primary_set", "sensitivity_only", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
