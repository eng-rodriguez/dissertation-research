"""vEpiSet cohort characterization: aggregate-only age and sex summary.

Reads the released metadata table only (MAT_Files/base_info.csv: file_name, sex,
age) and the committed, frozen fold manifest (the primary set is the recordings
with analysis_set == "primary"), and writes aggregate counts and
quantiles. No per-recording row is written or printed, so the output cannot be
linked back to an individual; raw EEG is never opened. Sex is reported exactly
as released, without recoding.

Approved by Josue on 2026-10-24 (decision D3 of the 2026-10-23 review synthesis)
as dataset characterization, not a model analysis.

Usage (from the repository root):
    python src/data/vepiset_cohort.py \
        --base-info data/raw/vepiset/extracted/opensource-dataset/MAT_Files/base_info.csv \
        --fold-manifest artifacts/protocol/f3/fold_manifest.csv \
        --out artifacts/dataset_audit
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

QUANTILE_METHOD = "linear interpolation between order statistics (numpy.percentile default)"


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def script_clean() -> bool:
    """True if this script is committed unchanged at HEAD."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", __file__], text=True,
            stderr=subprocess.DEVNULL)
        return out.strip() == ""
    except Exception:
        return False


def percentile(xs: list[float], q: float) -> float:
    """numpy.percentile's default (linear) method, without the dependency."""
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(rows: list[dict], label: str) -> dict:
    sex = collections.Counter(r["sex"].strip() for r in rows)
    ages: list[float] = []
    n_missing = 0
    for r in rows:
        raw = (r.get("age") or "").strip()
        try:
            ages.append(float(raw))
        except ValueError:
            n_missing += 1
    out = {
        "set": label,
        "n_recordings": len(rows),
        "sex_as_released": dict(sorted(sex.items())),
        "sex_percent": {k: round(100.0 * v / len(rows), 1) for k, v in sorted(sex.items())},
        "age_years": {
            "n_reported": len(ages),
            "n_missing": n_missing,
        },
    }
    if ages:
        non_integer = sorted({a for a in ages if a != int(a)})
        out["age_years"].update({
            "median": round(percentile(ages, 50), 1),
            "q1": round(percentile(ages, 25), 1),
            "q3": round(percentile(ages, 75), 1),
            "min": round(min(ages), 1),
            "max": round(max(ages), 1),
            "mean": round(sum(ages) / len(ages), 1),
            "n_under_18": sum(1 for a in ages if a < 18),
            "n_18_or_over": sum(1 for a in ages if a >= 18),
            "n_non_integer_values": len(non_integer),
            "quantile_method": QUANTILE_METHOD,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-info", required=True, type=Path)
    ap.add_argument("--fold-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    with open(args.base_info, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["eeg_id"] = r["file_name"].replace(".mat", "").strip()
    with open(args.fold_manifest, newline="") as f:
        primary = sorted({r["eeg_id"].strip() for r in csv.DictReader(f)
                          if r["analysis_set"] == "primary"})

    ids = {r["eeg_id"] for r in rows}
    missing = [i for i in primary if i not in ids]
    if missing:
        raise SystemExit(f"primary ids absent from base_info.csv: {missing}")
    primary_rows = [r for r in rows if r["eeg_id"] in set(primary)]
    if len(primary_rows) != len(primary):
        raise SystemExit("duplicate ids in base_info.csv")

    summary = {
        "provenance": {
            "git_commit": git_commit(),
            "script_committed_unchanged": script_clean(),
            "python": platform.python_version(),
            "base_info_sha256": sha256(args.base_info),
            "base_info_rows": len(rows),
            "fold_manifest": str(args.fold_manifest),
            "fold_manifest_sha256": sha256(args.fold_manifest),
            "primary_n": len(primary),
        },
        "scope": "aggregate only; no per-recording age or sex is written or printed",
        "sets": [summarize(rows, "all released recordings"),
                 summarize(primary_rows, "primary analysis set")],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    json.dump(summary, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
