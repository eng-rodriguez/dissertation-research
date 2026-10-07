"""vEpiSet audit, derived outputs: channel completeness and confounding report.

Reads only the aggregate tables written by `vepiset_audit.py audit` in
artifacts/dataset_audit/ (no raw data needed) and writes, next to them:
    channel_completeness.csv   structural channel count per recording
    confounding_report.md      patient/class/state confounding and split feasibility

Output is deterministic given the input tables, so it can be regenerated and
diffed. `vepiset_audit.py audit` calls this at the end of its run.

Usage (from the repository root):
    python src/data/vepiset_audit_report.py --audit-dir artifacts/dataset_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

EXPECTED_CHANNELS = 29  # paper: 19 10-20 + T1/T2 + A1/A2 + 2 ECG + 4 EMG
IED_CLASSES = ["generalized", "frontal", "temporal", "centro-parietal", "occipital"]
FOCAL = ["frontal", "temporal", "centro-parietal", "occipital"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_int(x: str) -> int:
    return int(float(x)) if x not in ("", None) else 0


def inverse_simpson(counts: list[int]) -> float:
    n = sum(counts)
    return 1.0 / sum((c / n) ** 2 for c in counts if c) if n else 0.0


def p_spans_split(n: int, test_frac: float = 0.2) -> float:
    """P(a recording with n epochs has epochs on both sides of a random split)."""
    return 1.0 - (1 - test_frac) ** n - test_frac ** n


def p_in_all_folds(n: int, k: int = 5) -> float:
    """P(a recording with n epochs, assigned uniformly to k folds, hits every fold)."""
    return sum((-1) ** j * math.comb(k, j) * (1 - j / k) ** n for j in range(k + 1))


def write_channel_completeness(recs: list[dict], out: Path) -> list[dict]:
    rows = []
    for r in recs:
        n_ch = to_int(r["n_channels"])
        rows.append({
            "eeg_id": r["eeg_id"],
            "n_channels_stored": n_ch,
            "n_channels_expected": EXPECTED_CHANNELS,
            "channel_count_complete": n_ch == EXPECTED_CHANNELS,
            "n_samples": to_int(r["n_samples"]),
            "duration_s": r["duration_s"],
            "channel_names_in_file": False,
            "scope": "structural: eeg_data array shape only; no bad-channel or signal-quality assessment",
        })
    fields = list(rows[0])
    with open(out / "channel_completeness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def fmt(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}"


def pct(x: str) -> str:
    """Share column from class_summary.csv as a percentage; empty for absent classes."""
    return f"{100 * float(x):.0f}%" if x else "–"


def build_report(audit_dir: Path, recs: list[dict], ch_rows: list[dict]) -> str:
    summary = json.loads((audit_dir / "audit_summary.json").read_text())
    classes = {r["class"]: r for r in read_csv(audit_dir / "class_summary.csv")}
    cs = {r["class"]: r for r in read_csv(audit_dir / "class_state_matrix.csv")}

    ied = [r for r in recs if to_int(r["n_ied_epochs"])]
    non = [r for r in recs if not to_int(r["n_ied_epochs"])]
    ied_counts = [to_int(r["n_ied_epochs"]) for r in ied]
    all_counts = [to_int(r["n_epochs"]) for r in recs]

    def sleep_share(group):
        w = sum(to_int(r["ep_wake"]) for r in group)
        s = sum(to_int(r["ep_sleep"]) for r in group)
        return s / (w + s) if w + s else float("nan")

    def coverage(r, prefix):
        w, s = to_int(r[prefix + "wake"]), to_int(r[prefix + "sleep"])
        return "both" if w and s else "wake only" if w else "sleep only" if s else "neither"

    def tally(group, prefix):
        out = {k: 0 for k in ["both", "wake only", "sleep only", "neither"]}
        for r in group:
            out[coverage(r, prefix)] += 1
        return out

    gen_only = sum(1 for r in ied if to_int(r["ep_generalized"])
                   and not any(to_int(r[f"ep_{c}"]) for c in FOCAL))
    focal_only = sum(1 for r in ied if not to_int(r["ep_generalized"]))
    both_gf = len(ied) - gen_only - focal_only
    g_counts = [to_int(r["ep_generalized"]) for r in recs if to_int(r["ep_generalized"])]
    f_counts = [sum(to_int(r[f"ep_{c}"]) for c in FOCAL) for r in recs]
    f_counts = [c for c in f_counts if c]

    # Clustering: Kish mean cluster size for IED epochs and illustrative design effects.
    m_kish = sum(c * c for c in ied_counts) / sum(ied_counts)
    top5 = sum(sorted(ied_counts, reverse=True)[:5]) / sum(ied_counts)
    q = statistics.quantiles(ied_counts, n=4) if len(ied_counts) > 1 else ied_counts * 3

    span_ied = sum(p_spans_split(c) for c in ied_counts)
    span_all = sum(p_spans_split(c) for c in all_counts)
    folds_ied = sum(p_in_all_folds(c) for c in ied_counts)
    folds_all = sum(p_in_all_folds(c) for c in all_counts)

    n_complete = sum(1 for r in ch_rows if r["channel_count_complete"])
    durations = [float(r["duration_s"]) for r in recs]

    prov = summary.get("provenance", {})
    L: list[str] = []
    a = L.append
    a("# vEpiSet confounding and split-feasibility report")
    a("")
    a("Generated by `src/data/vepiset_audit_report.py` from the aggregate tables in "
      "`artifacts/dataset_audit/`. Those tables were produced by `vepiset_audit.py audit` "
      f"at commit `{prov.get('git_commit', 'unknown')[:7]}`. Status: WORKING evidence for "
      "study design. Nothing here locks a split, target or research question.")
    a("")
    a("Unit: one recording (MAT file). The dataset paper states one recording per patient; "
      "the pseudonymous IDs cannot confirm this, so \"patient\" below means recording. "
      "Spatial class is the dataset authors' label on each 4-s epoch.")
    a("")

    a("## 1. Structural completeness")
    a(f"- {n_complete} of {len(ch_rows)} recordings store {EXPECTED_CHANNELS} channels "
      f"(`channel_completeness.csv`). Durations range {fmt(min(durations), 0)}–"
      f"{fmt(max(durations), 0)} s.")
    a("- Channel names, order and reference are not stored in the MAT files.")
    a("- This is a count of array rows only. No bad-channel, flat-line, artifact or "
      "signal-quality assessment has been performed.")
    a("")

    a("## 2. Patient concentration by spatial class")
    a("Effective N = inverse Simpson index over per-recording epoch counts: the number of "
      "equally contributing patients the class is worth.")
    a("")
    a("| Class | IED epochs | Recordings | Recordings ≥10 epochs | Top recording | Top 3 | Effective N |")
    a("|---|---|---|---|---|---|---|")
    for c in IED_CLASSES:
        r = classes[c]
        a(f"| {c} | {r['epochs']} | {r['recordings']} | {r['recordings_ge10_epochs']} | "
          f"{pct(r['top_recording_share'])} | {pct(r['top3_recordings_share'])} | "
          f"{fmt(float(r['effective_n_recordings']))} |")
    a("")

    cp, oc = classes["centro-parietal"], classes["occipital"]
    a("## 3. Very low effective N: centro-parietal and occipital")
    a(f"- Centro-parietal: {cp['epochs']} epochs from {cp['recordings']} recordings "
      f"(per recording: {cp['per_recording_epochs_desc']}); effective N "
      f"{fmt(float(cp['effective_n_recordings']))}.")
    a(f"- Occipital: {oc['epochs']} epochs from {oc['recordings']} recordings "
      f"(per recording: {oc['per_recording_epochs_desc']}); effective N "
      f"{fmt(float(oc['effective_n_recordings']))}.")
    a("- A patient-independent classifier for either class learns from 2–3 patients. Under "
      "leave-one-patient-out, holding out the dominant patient removes most of the class. "
      "A fixed held-out test set cannot contain every class with more than one patient.")
    a("")

    a("## 4. State and IED-status confounding")
    t_ied, t_non = tally(ied, "ep_"), tally(non, "ep_")
    t_iedep = tally(ied, "ied_ep_")
    a(f"- Sleep share of all epochs: {fmt(100 * sleep_share(ied), 0)}% in the {len(ied)} IED "
      f"recordings vs {fmt(100 * sleep_share(non), 0)}% in the {len(non)} recordings without "
      "IEDs.")
    a(f"- State coverage of recordings without IEDs: wake only {t_non['wake only']}, sleep only "
      f"{t_non['sleep only']}, both {t_non['both']}. IED recordings: wake only "
      f"{t_ied['wake only']}, sleep only {t_ied['sleep only']}, both {t_ied['both']}.")
    a(f"- Where each IED patient's IEDs fall: sleep only {t_iedep['sleep only']}, wake only "
      f"{t_iedep['wake only']}, both states {t_iedep['both']}. Within-patient wake-vs-sleep "
      f"IED comparisons are possible in {t_iedep['both']} patients.")
    a("")
    a("| Class | Wake epochs | Sleep epochs | Mixed | Recordings awake / asleep |")
    a("|---|---|---|---|---|")
    for c in ["non-IED"] + IED_CLASSES:
        r = cs[c]
        a(f"| {c} | {r['epochs_wake']} | {r['epochs_sleep']} | {r['epochs_mixed']} | "
          f"{r['recordings_wake']} / {r['recordings_sleep']} |")
    a("")
    a("- Consequence: a detector can score well by recognising sleep or by recognising "
      "patients, because IED patients are more often asleep. Spatial classes also differ in "
      "state mix, largely through a few patients.")
    a("")

    a("## 5. Generalized / focal patient overlap")
    a(f"- IED recordings: generalized only {gen_only}, focal only {focal_only}, both "
      f"{both_gf}.")
    a(f"- Generalized: {len(g_counts)} recordings, effective N "
      f"{fmt(inverse_simpson(g_counts))}. Focal (pooled): {len(f_counts)} recordings, "
      f"effective N {fmt(inverse_simpson(f_counts))}.")
    a(f"- The {both_gf} mixed recordings sit on both sides of a generalized-vs-focal contrast, "
      "so the patient-level label is ambiguous for them and they must be kept whole within "
      "one fold.")
    a("")

    a("## 6. Within-patient epoch clustering")
    a(f"- IED epochs per IED recording: median {fmt(statistics.median(ied_counts), 0)}, "
      f"IQR {fmt(q[0], 0)}–{fmt(q[2], 0)}, max {max(ied_counts)}. The top 5 recordings hold "
      f"{fmt(100 * top5, 0)}% of all IED epochs.")
    a(f"- Effective N of IED epochs across recordings: "
      f"{fmt(inverse_simpson(ied_counts))} of {len(ied_counts)} recordings.")
    a(f"- Kish mean cluster size for IED epochs: {fmt(m_kish)}. Illustrative design effect "
      f"1 + (m − 1)ρ: ρ = 0.05 → {fmt(1 + (m_kish - 1) * 0.05)}; ρ = 0.2 → "
      f"{fmt(1 + (m_kish - 1) * 0.2)}. ρ (within-patient correlation of features or errors) "
      "is unknown until features exist; these values only show the scale.")
    a("- Epochs from one patient are not independent observations. Epoch-level standard "
      "errors and p-values overstate precision by roughly the square root of the design "
      "effect.")
    a("")

    a("## 7. Consequences of epoch-level splitting")
    a("Computed from per-recording epoch counts, assuming epochs are assigned at random as in "
      "the dataset authors' code (random 20% test split; 5-fold StratifiedKFold without "
      "grouping).")
    a(f"- Random 80/20 split: expected {fmt(span_all)} of {len(all_counts)} recordings have "
      f"epochs in both train and test; for IED epochs alone, {fmt(span_ied)} of "
      f"{len(ied_counts)} IED recordings.")
    a(f"- 5-fold CV: expected {fmt(folds_all)} of {len(all_counts)} recordings appear in "
      f"every fold; for IED epochs, {fmt(folds_ied)} of {len(ied_counts)}.")
    a("- So almost every test epoch has same-patient (and often adjacent-in-time) epochs in "
      "training. Published epoch-level scores on vEpiSet measure within-patient "
      "recognition and are not a patient-generalization benchmark (RDR-004).")
    a("")

    a("## 8. Requirements for valid patient-independent evaluation")
    a("- Group every split by recording; no recording contributes epochs to more than one of "
      "train / validation / test.")
    a("- Stratify folds at the patient level by class presence and state coverage, not by "
      "epoch counts.")
    a("- Handle state explicitly: state-matched negatives, state-stratified metrics, or "
      "state as a covariate. Report wake and sleep results separately where N allows.")
    a("- Report patient-level results (per-patient metrics and their distribution) with "
      "patient-cluster bootstrap confidence intervals; epoch-level pooled metrics are "
      "secondary.")
    a("- Keep model selection inside the training patients (nested CV). Any fixed held-out "
      "test set is touched once.")
    a("- Fit normalization, feature selection and thresholds on training patients only.")
    a("- Do not treat centro-parietal or occipital as patient-generalizable classes with "
      "this dataset alone; merge, drop, or analyse them descriptively.")
    a("- Keep mixed-class recordings whole within a fold and decide explicitly how they are "
      "labelled at patient level.")
    a("")

    a("## 9. What this report does not establish")
    a("- Nothing about signal quality, bad channels, filtering, or channel order.")
    a("- Nothing about whether any representation (waveform, spectral, connectivity) "
      "separates classes or generalizes across patients.")
    a("- Spatial labels are reader judgements of IED field per epoch, not "
      "epileptogenic-zone localization.")
    a("")
    return "\n".join(L)


def run(audit_dir: Path) -> None:
    recs = read_csv(audit_dir / "subject_summary.csv")
    ch_rows = write_channel_completeness(recs, audit_dir)
    (audit_dir / "confounding_report.md").write_text(build_report(audit_dir, recs, ch_rows))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit-dir", default="artifacts/dataset_audit")
    args = ap.parse_args(argv)
    run(Path(args.audit_dir))
    print(f"wrote {args.audit_dir}/channel_completeness.csv and confounding_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
