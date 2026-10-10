# 06 — EXPERIMENTAL PROTOCOL
Status: **v1.0 — LOCKED (RDR-013, 2026-10-10)** Owner: Workstream 03 (Research Question & Protocol). Last updated: 2026-10-10.

Lock rule: `06` locks only with Josue's explicit approval and an RDR in `11-DECISION-LOG.md`, after all 14 mandatory issues below are DECIDED and the pre-lock checks in §6 are done. After lock, material changes follow §9.

Governing decisions: RDR-004 (patient independence), RDR-006 (minimum sufficient novelty), RDR-011 (central target CAND-007, LOCKED at the question-family level only; exact RQ, hypothesis and this protocol NOT YET LOCKED). Integrity requirement (RDR-011, `02`) applies to every section.

Labels used in this file:
- **DECIDED (date)**: approved by Josue. Still not LOCKED until `06` locks.
- **REVISED – CONDITIONAL**: Josue's revision is applied; the issue becomes DECIDED only when the named blocking check resolves and Josue confirms.
- **PROPOSED**: Workstream 03's recommendation, not yet reviewed.
- **VERIFIED / SOURCE_REPORTED / TO VERIFY**: dataset-fact labels as in `05`.
- **[VERIFY]** after a citation: cited from memory, not yet checked against the source. Must be checked before any manuscript use.

Version history:
- v0.1 (2026-10-07): generic template for the earlier spatial target. Nothing decided.
- v0.2 (2026-10-10): proposed defaults for the 14 mandatory issues.
- v0.3 (2026-10-10): Josue's decisions of 2026-10-10 applied (thread "Research question and protocol"). Issues 2, 5, 6, 7, 8, 9, 11, 13 DECIDED; issues 1, 3, 4, 10, 12, 14 REVISED – CONDITIONAL. F-2 criteria fixed before F-2 was run.
- v0.3 addendum (2026-10-10): F-2 run; pre-specified verdict MARGINAL (`protocol/2026-10-10-F2-precision-feasibility.md`). Josue kept SESOI 0.03; issue 3 DECIDED.
- v0.3 addendum 2 (2026-10-10): Josue approved issue 3 in full, issue 4 (DECIDED), the pilot size of 8 (issue 10 stays conditional on F-4 and F-3), and recorded F-2 as MARGINAL but acceptable for continuing. F-4 stop rule fixed before the QC run (issue 12).
- v0.3 addendum 3 (2026-10-10): F-4 run (CONTINUE, 41 of 46). Josue confirmed the issue-12 QC rule as run; issue 12 DECIDED. F-3 done: pilot and fold manifest drawn and frozen (code 5cc0058, outputs 93a2c49); issue 10 awaits Josue's confirmation and the biostatistics review of the CV-variance method.
- v0.3 addendum 4 (2026-10-10): Josue accepted F-4 under the pre-specified rule and accepted F-3 as generated. F-3 and F-4 COMPLETE. Issue 10 pilot and fold-manifest portion DECIDED; its inference / CV-variance portion stays conditional on the biostatistics review before lock. Precision consequence recorded (36 evaluable patients; no SESOI change).
- v0.3 addendum 5 (2026-10-10): F-1 COMPLETE (outer-only share 0.361 > 0.10). Josue adopted the whole-epoch salience search with clamped window centre; issue 1 DECIDED; terminology is now "label-blind salience-selected 2-s window".
- v0.3 addendum 6 (2026-10-10): Josue's lock-readiness decisions (`protocol/2026-10-10-lock-readiness.md`): D1 issue 8 count and per-band scheme APPROVED (issue 8 fully DECIDED); D2 §4.1 learner and tuning APPROVED with an inner-fold validity rule (added, checked on the frozen manifest: all 50 outer splits valid); D3 §4.2 APPROVED incl. distance-to-IED reporting; D4 §4.3 APPROVED, exact wPLI specification required before lock (drafted, PROPOSED); D5 §4.4, D6 §5, D7 RQ-1/H-1 wording APPROVED; D8 issue 14 DECIDED at policy level; D9 biostatistics deferred pending the lens review (received, `reviews/2026-10-10-biostatistics-lens-review.md`); D10 partial F-5 accepted for lock with a mandatory pre-submission recheck.
- v0.3 addendum 8 (2026-10-10): F-6 COMPLETE: post-exclusion evaluable patients 35 (from 36); recorded in `11`; no RDR required; inner-fold rule passes. All lock preconditions met except the lock RDR, which awaits Josue's explicit approval.
- v0.3 addendum 7 (2026-10-10): Josue's final pre-lock decisions. Biostatistics-lens R1–R10 adopted (issues 2, 3, 7, 9, 10, §4.4); Q1–Q5 decided; Nadeau–Bengio corrected repeated-CV interval is the PRIMARY interval; issue 10 inference DECIDED; exact wPLI specification APPROVED (incl. first/last-epoch exclusion and the PC-3 null-coupling comparison); pre-confirmatory engineering validation EV-1 added (§6.1). All 14 issues DECIDED.
- v1.0 (2026-10-10): locked by RDR-013.

---

## 0. Decision register (all DECIDED; LOCKED with v1.0)

| # | Issue | Current text (one line) | State | Blocking check |
|---|---|---|---|---|
| 1 | Analysis window | Label-blind salience-selected 2-s window inside each 4-s epoch: salience peak t_raw searched over the whole epoch, window centre clip(t_raw, 1.0 s, 3.0 s), window [centre − 1.0 s, centre + 1.0 s]; identical for all classes and representations. FC primary bands θ/α/β; δ exploratory on 2 s, evaluated mainly in the 4-s sensitivity analysis. Native 4-s epoch = sensitivity | DECIDED (2026-10-10; F-1 COMPLETE) | — |
| 2 | Primary estimand & metric | Procedure-level, within-patient, same-state ΔAUROC (B2 + FC vs B2), equal weight per patient, for unseen IED-positive patients evaluable after all exclusions (≥ 5 IED and ≥ 5 non-IED epochs in a common state); per-repeat Δ averaged over repeats, never pooled. Not the all-patient deployment estimand; deployment metrics secondary | DECIDED (2026-10-10) | — |
| 3 | SESOI & bounded null | SESOI 0.03 ΔAUROC (not widened to 0.04); ordered, mutually exclusive seven-rule taxonomy (incl. "positive, below SESOI" and "negative, within SESOI"); only the primary interval assigns the label; per-patient Δ SD, prediction interval and range reported; strong heterogeneity may yield an inconclusive result | DECIDED (2026-10-10) | — |
| 4 | Positive controls | PC-1 and PC-3 are validity gates for a bounded-null reading; PC-2 is a biological sanity check reported regardless of outcome, not a gate | DECIDED (2026-10-10) | — |
| 5 | Waveform/spectral baseline | Nested ladder B0 single-channel max → B1 19-channel waveform + spectral (incl. state-proxy bands) | DECIDED (2026-10-10) | — |
| 6 | Multichannel non-FC spatial baseline | B2 = B1 + amplitude/power topography and spatial-extent summaries; **B2 is the reference model for the primary estimand** | DECIDED (2026-10-10) | — |
| 7 | Anatomical-adjacency comparator | B2 + fixed 10–20 distance-graph features, no FC | DECIDED (2026-10-10) | — |
| 8 | Surrogate-FC control | B2 + FC on per-channel IAAFT surrogates; deterministic recorded seeds; IAAFT generated per band on the band-passed window; k = 5 realisations for the 2-s primary θ/α/β, k = 3 for the 4-s sensitivity and δ; mean wPLI per edge across realisations | DECIDED (2026-10-10; count and scheme D1) | — |
| 9 | State control | State-matched within-patient evaluation; state-balanced training weights; state-proxy features; state-stratified reporting | DECIDED (2026-10-10) | — |
| 10 | Grouped evaluation & inference | Repeated patient-grouped nested CV (frozen F-3 manifest, 5 folds × 10 repeats; pilot of 8 frozen and never returned); no fixed vEpiSet test set. **Primary interval: Nadeau–Bengio corrected repeated-CV**; patient-cluster percentile bootstrap = sensitivity only. AI-simulated biostatistics review accepted as the internal pre-lock review (not human sign-off); human review of issue 10 required before submission | DECIDED (2026-10-10) | — |
| 11 | Montage / re-reference | CAR primary; longitudinal bipolar sensitivity | DECIDED (2026-10-10) | — |
| 12 | 19-channel QC / eligibility | Primary: no bad scalp channels and satisfactory C3/C4, **no interpolation**. Interpolation of ≤ 2 non-adjacent bad channels only as a sensitivity analysis. If the rule removes too many recordings, return for a protocol decision | DECIDED (2026-10-10; F-4: CONTINUE, 41 of 46) | — |
| 13 | ECE engineering contribution | Open, leakage-controlled incremental-value evaluation framework, including a signal-provenance and reference-normalization layer that identifies and removes the undocumented common reference | DECIDED (2026-10-10) | — |
| 14 | External validation role | Secondary, outside the Draft v1 confirmatory claim. External corroboration of the incremental-value conclusion with the frozen representation/comparator framework and a dataset-appropriate patient-grouped estimand. BDSP preferred if grouping and clip structure allow valid evaluation; TUEV fallback; frozen-model transfer tertiary | DECIDED at policy level (2026-10-10, D8); dataset adoption by separate RDR | — (not a lock blocker) |

Further items `06` fixes beyond the mandatory list: learner family (§4.1), exclusions (§4.2), FC estimator details (§4.3), multiplicity (§4.4), reproducibility manifest (§4.5). All APPROVED 2026-10-10 (D2–D5; exact wPLI specification approved as written).

---

## 1. Research question and hypothesis (wording v0.3 APPROVED by Josue 2026-10-10, D7; LOCKED with v1.0)

Question family (LOCKED, RDR-011): does functional-connectivity information add discriminative value beyond waveform and spectral information for patient-independent IED detection, with state confounding controlled?

**RQ-1 (primary), wording v0.3 (approved 2026-10-10).** In IED-positive patients not seen during training, does adding θ-, α- and β-band inter-channel functional-connectivity features, computed on a label-blind salience-selected 2-s window, to a multichannel waveform, spectral and spatial-topography representation change the discrimination between the patient's IED-containing and IED-free scalp EEG epochs recorded in the same wake/sleep state?

**H-1, wording v0.3 (approved 2026-10-10).** In unseen IED-positive patients with both IED and same-state non-IED epochs, the equal-patient-weight mean within-patient, same-state ΔAUROC (B2 + FC minus B2) differs from zero. The result is read with the issue-3 taxonomy against SESOI = 0.03. A bounded null is interpretable only if PC-1 and PC-3 pass; strong patient-to-patient heterogeneity of the gain may yield an inconclusive result instead.

Notes on the wording:
- "IED-locked" is dropped: a detector does not know where the IED is, so the window comes from a label-blind rule applied identically to every epoch (issue 1).
- "Beyond waveform and spectral" is operationalised as beyond B2 (issue 6), so a gain cannot be explained by field extent or topography alone.
- The population is IED-positive patients only. The primary estimand says nothing about false positives in IED-free patients; that is the deployment question, answered by secondary metrics (issue 2).

Prior expectation (stated, not tested): the FIU lineage (`13`, Mohammed) predicts a positive increment. Mohammed 2023 FC-GNN ≈ CA-GNN, Malkov 2026 (LIT-010) and Nhu 2023 (single-channel max ≥ multichannel) predict a small or zero increment.

Secondary questions (wording also NOT LOCKED):
- **RQ-2 (evaluation validity):** how much do epoch-level splits and the state imbalance inflate detection performance on vEpiSet? Feeds the engineering contribution (issue 13).
- **RQ-3 (external):** is the incremental-value conclusion corroborated in an independent dataset? (issue 14). Not required for Draft v1.
- **RQ-4 (spatial, secondary, explicitly underpowered):** does the FC increment differ between generalized and focal IEDs? Descriptive estimates with CIs only, plus the within-patient check in the 7 usable mixed recordings (audit §10). No five-class spatial classifier as a claim.

---

## 2. Data, unit and analysis sets

- Dataset: vEpiSet, Figshare v2 (MD5 3b46ecbc…, VERIFIED). Signal source: the continuous MAT `eeg_data`, rows 1–19 (scalp; VERIFIED). Non-scalp rows (PG1/PG2, A1/A2, ECG, EMG) are never model inputs (shortcut evidence LIT-011, LIT-045); they may be used only as QC references.
- Signal is already 50 Hz notched and high-cut near 70 Hz (VERIFIED). No re-application of the authors' 0.1–70 Hz or notch filters. 0.1 Hz high-pass remains TO VERIFY and does not affect bands ≥ 1 Hz.
- Label unit: the authors' 4-s epoch (folder label; VERIFIED). Positive = any IED class 1–5; negative = class 0.
- Independence unit: recording, treated as patient (one recording per patient is SOURCE_REPORTED).
- **Primary analysis set:** epochs with a single state label (wake or sleep), full length, passing epoch QC, not in the exclusions of §4.2, from recordings eligible under the primary QC rule (issue 12) and outside the frozen pilot subset (issue 10).
- **Primary evaluation population (issue 2):** IED-positive recordings with ≥ 5 IED epochs and ≥ 5 non-IED epochs in at least one common state. From committed `subject_state_matrix.csv` (checked 2026-10-10, before QC and pilot removal): 46 of 52 IED recordings qualify at ≥ 5, holding 2,482 of 2,503 state-labelled IED epochs (≥ 3: 49; ≥ 10: 39). The 32 recordings without IEDs contribute to training and to the deployment-style secondary metrics only. **After F-4 and F-3 (2026-10-10, frozen):** 41 of the 46 pass QC; 5 go to the pilot; the confirmatory primary set is 69 recordings with **36 evaluable IED-positive patients** (14 evaluable in wake, 30 in sleep), 6 other IED-positive and 27 IED-free recordings. **After all §4.2 exclusions and epoch QC (F-6, 2026-10-10): 35 evaluable** (13 in wake, 28 in sleep).

---

## 3. The 14 mandatory issues

### Issue 1 — Symmetric, label-blind analysis window — DECIDED (2026-10-10)
**Final rule (Josue's decision after F-1, 2026-10-10).** The window is called a **label-blind salience-selected 2-s window**. It is not necessarily centred on the salience peak, because edge peaks have their window centre clamped.
1. Within each 4-s epoch, compute the label-blind salience trace on the CAR signal over the **entire epoch**: per channel, the smoothed Teager–Kaiser energy (20-ms smoothing), z-scored with that channel's recording-wide median and MAD (computed over the whole recording, so label-blind and available offline). Salience s(t) = maximum over the 19 channels.
2. t_raw = argmax s(t) over the full epoch [0, 4) s.
3. t_center = clip(t_raw, 1.0 s, 3.0 s).
4. Window = **[t_center − 1.0 s, t_center + 1.0 s]** (2 s, 1,000 samples). It always lies inside the epoch and always contains t_raw.
5. Applied identically to IED and non-IED epochs and to every representation (B0–B2, adjacency, surrogate FC, FC). Non-IED windows therefore hold the most salient transient in that epoch, a natural hard negative.
6. Band-pass filtering for FC is applied to the continuous recording before windows are extracted, to avoid filter edge effects.
7. **FC bands on the 2-s salience-selected window:** θ 4–8, α 8–13, β 13–30 Hz are primary. δ 1–4 Hz is exploratory on the 2-s window, because short windows make low-frequency phase estimates unstable (2–8 cycles).
8. **Pre-specified sensitivity analysis:** the native 4-s whole epoch, no salience selection, all representations. δ-band FC is evaluated mainly here.

F-1 is COMPLETE; no further F-1 rerun is required.

**Why.** Both lens reviews' M1 (`reviews/`): centring positives on the marker and taking negatives anywhere else lets window placement carry the label, and whole-epoch FC is dominated by background and state. The 2-s window contains the discharge and its after-going slow wave with more cycles for phase estimation than the v0.2 1-s window.

**History.** Draft v0.3 searched the salience peak only within [1.0, 3.0] s, so a discharge in the outer 1.0 s could never select the window. F-1 (below) showed this affects a large, state-independent share of IED epochs, and Josue adopted the pre-recorded alternative (whole-epoch search, clamped centre) on 2026-10-10. Consequence: a discharge near an epoch edge sits off-centre in its window; this applies identically to both classes.

**F-1 result (2026-10-10, Josue's Mac; outputs commit 8a6f0b8; `protocol/2026-10-10-F1-marker-position-run.md`).** Pre-specified rule: **RETURN TO JOSUE.** Patient-weighted OUTER-ONLY share over the 36 confirmatory evaluable patients = **0.361** (threshold 0.10); epoch-weighted 0.300 (511 of 1,703 marked IED epochs). Every one of the 36 patients is above 0.10 (range 0.105–0.55, median 0.40). Point markers are spread evenly across the 4-s epoch (each 0.25-s bin holds 5.3–7.0% of 3,113 markers), as expected from the fixed 4-s grid. Wake 0.30 and sleep 0.28 (all recordings), so the loss is not state-dependent. 12 IED epochs have no marker (already excluded, §4.2). Decision 2026-10-10 (Josue): the restricted [1.0, 3.0] s search is not acceptable; the pre-recorded whole-epoch search with clamped centre is adopted (rule above).

**Diagnostic only.** Window validity: share of IED epochs whose salience-selected window contains an annotated `!` marker, computed on the pilot subset. It may be reported, but it must not be used to tune the rule on the basis of model performance.

### Issue 2 — Primary estimand and metric — DECIDED (2026-10-10)
**Text.**
- **Population:** unseen IED-positive patients that contain both IED epochs and same-state non-IED epochs (eligibility in §2). **This is not the all-patient deployment estimand.**
- **Estimand:** Δ = mean over eligible IED-positive patients of [AUROC_p(B2 + FC) − AUROC_p(B2)].
- AUROC_p is the patient's own IED epochs versus the same patient's non-IED epochs **in the same state**, computed per state and combined within the patient weighted by IED epoch count. Scores are out-of-fold predictions from models that never saw that patient.
- Equal weight per patient (the top 5 recordings hold 42% of IED epochs).
- Both models use the same folds, learner, tuning budget and seeds, so Δ is paired within patient.
- **Estimand scope (R1, adopted 2026-10-10).** Δ is a procedure-level estimand: the expected within-patient, same-state ΔAUROC of the frozen pipeline (features, learner, tuning, weights) trained on about 4/5 of the primary set (≈ 55 recordings) and applied to a new eligible IED-positive patient from the same source population. It is not the performance of a single final model.
- **Computation (R2, adopted).** For each repeat r, AUROC_{p,s,r} is the Mann–Whitney AUROC (ties = ½) of patient p's IED vs non-IED epochs in state s, scored by the single outer-fold model that held p out. Only states in which p is evaluable (≥ 5 IED and ≥ 5 non-IED epochs after all §4.2 and epoch-QC exclusions) enter. AUROC_{p,r} = Σ_s w_{ps} AUROC_{p,s,r} with w_{ps} = n^IED_{ps} / Σ_{s′} n^IED_{ps′} over evaluable states. Δ_{p,r} = AUROC_{p,r}(B2 + FC) − AUROC_{p,r}(B2); Δ_p = mean_r Δ_{p,r}; Δ̂ = mean of Δ_p over the evaluable patients (equal weight, including patients with 5–9 IED epochs). Scores are never pooled or averaged across repeats.
- **Eligibility after exclusions (R3 and Q4, adopted).** Before any model output exists, repo code recomputes evaluability from labels and counts only, on the final primary analysis set after all §4.2 exclusions (including the first/last-epoch exclusion) and epoch QC. Patients who lose evaluability stay in their frozen folds as training-only (N) recordings; **the fold manifest is not redrawn**. The resulting E count and any change from 36 are recorded in `11` before the confirmatory run. **F-6 result (2026-10-10): E = 35** (13 evaluable in wake, 28 in sleep); DA00103R lost evaluability, DA00102T is now evaluable in wake only and DA00102W in sleep only; recorded in `11` (LOG-2026-10-10-F6); no RDR required. An RDR is required if E falls below 33, or if the change arises from a substantive protocol alteration rather than the already pre-specified exclusions. The inner-fold validity rule (§4.1) is rechecked on the recomputed E set.

**Why.** Within-patient, same-state comparison removes the between-patient channel where the state imbalance (64% vs 36% sleep) and cohort differences live (Adjouadi-lens M2/M3). Training remains patient-independent. AUROC is prevalence-free, so it is comparable across patients and strata.

**Secondary metrics (estimation, not confirmatory), including IED-free patients:**
- Pooled state-matched ΔAUROC and ΔAUPRC over all held-out epochs, including the 32 IED-free patients (deployment-like). AUPRC only within one analysis set, never compared across strata.
- Δ sensitivity at a fixed false-positive rate per minute, thresholds set inside training folds, FP rate measured in IED-free patients.
- Δ log-loss after within-training-fold calibration (Pepe et al. 2013 [VERIFY]).
- Calibration summary.

**Ceiling risk (Cabrerizo-lens M2).** Mitigated by the salience-selected window (harder negatives) and by looking at the baseline level only on the pilot subset before lock.

### Issue 3 — SESOI and bounded-null logic — DECIDED (2026-10-10, after F-2)
**Text.** SESOI = 0.03 on the ΔAUROC scale of issue 2. Not widened to 0.04.

**F-2 (complete, 2026-10-10): MARGINAL under the pre-specified rule, accepted by Josue as sufficient to continue.** F-2 is precision-feasibility evidence, not a power guarantee. Central scenario P(equivalence | Δ = 0) ≥ 0.997 (90% CI half-width ≈ 0.010–0.011, 37–42 patients); pessimistic worst cell 0.249. Precision is driven mainly by how much the FC gain varies between patients, then by QC loss. If per-patient gains are strongly heterogeneous, the likely null outcome is "inconclusive", not "equivalent". Details: `protocol/2026-10-10-F2-precision-feasibility.md`.

**Decision (Josue, 2026-10-10): SESOI = 0.03, DECIDED.** The margin is justified by its engineering meaning, not fitted to precision.
- **Pre-registered reporting:** the per-patient ΔAUROC distribution, its SD, and a 95% prediction interval for a new patient's Δ, so that any outcome, including an inconclusive one, states how variable the FC gain is across patients.
- **Heterogeneity caveat (stated in advance):** strong patient-to-patient heterogeneity of the FC gain may yield an inconclusive result rather than a bounded null. An inconclusive result is reported as such.

**Result taxonomy (R4 and Q1, adopted 2026-10-10; replaces the five-outcome table).** Bounds come from the primary interval method (issue 10) and are compared unrounded. L95/U95 and L90/U90 are the bounds of the 95% and 90% intervals. The rules are evaluated in this order and the first match applies:

| # | Outcome | Rule |
|---|---|---|
| 1 | Meaningfully superior | L95 > +0.03 |
| 2 | Positive, below SESOI | L95 > 0 and U90 < +0.03 |
| 3 | Positive, meaningful magnitude not established | L95 > 0 |
| 4 | Negative, within SESOI | U95 < 0 and L90 > −0.03 |
| 5 | Inferior | U95 < 0 |
| 6 | Equivalent / bounded null | L90 > −0.03 and U90 < +0.03 (two one-sided tests at α = 0.05 each; Lakens 2017 [VERIFY]) |
| 7 | Inconclusive | otherwise; reported as such, never as "no effect" |

- **Meaning of outcome 2 (Q1, Josue).** "Positive, below SESOI" is a statistically supported positive increment whose magnitude does not justify FC under the pre-specified engineering threshold. Because it also meets the equivalence bounds, both facts are stated. It is never called "no effect", and "practically equivalent" is not used in a way that hides the detected direction. Outcome 4 is read symmetrically.
- **Validity gate.** If PC-1 or PC-3 fails, outcome 6 is reported as "Inconclusive (validity gate failed)".
- **Governing interval (R5, adopted).** Only the primary interval assigns the outcome label. The sensitivity interval's label is reported beside it; a discordance is reported and discussed, never used to relabel. PC-1's 95% lower bound uses the same primary method.
- **Error rates (stated in advance).** At Δ = ±0.03 the combined probability of a false claim can reach about 0.075 (0.05 false equivalence + 0.025 false "meaningfully superior"); elsewhere it is ≤ 0.05. The 90% equivalence interval is retained (Q3).
- **Heterogeneity reporting (R6, adopted).** Report the SD s_Δ of the repeat-averaged Δ_p; a 95% prediction interval for a new patient's observed Δ, Δ̂ ± t_{n−1, 0.975} · s_Δ · √(1 + 1/n) (descriptive, since the Δ_p share training data, and wider than the spread of true gains because it includes epoch-sampling noise); the distribution-free range [min Δ_p, max Δ_p]; the counts with Δ_p > +0.03 and Δ_p < −0.03; and a forest plot of Δ_p ordered by IED count with across-repeat min–max. No per-patient p-values.

A bounded null is interpretable only if PC-1 and PC-3 pass (issue 4).

**Justification for 0.03 (accepted 2026-10-10).**
- Engineering cost–benefit: FC adds 513 primary features (171 pairs × 3 bands), a reference-dependent preprocessing chain, and band-limited estimation on short windows. An increment below 0.03 AUROC would not justify adding that chain to a detector. At a baseline of about 0.90 this equals removing about 30% of the residual (1 − AUROC).
- Scale of observed spatial-context gains: Wang 2026 anatomical graph +0.069 AUROC on TUEV (LIT-046); V2IED +2.4% accuracy (LIT-048).

### Issue 4 — Positive controls — DECIDED (2026-10-10)
**Text.**
- **PC-1 (signal is there; validity gate).** FC-only model (no B2 features), same folds and learner. Pass: within-patient AUROC 95% CI lower bound > 0.5, using the primary (Nadeau–Bengio) interval method (R5).
- **PC-3 (pipeline works; validity gate).** Synthetic test in the repo test suite: inject transients with a known inter-channel phase lag into real or simulated background; the FC pipeline must recover the lag structure, and IAAFT surrogates must destroy it. **Also (approved with the wPLI specification, 2026-10-10):** on synthetic data with no inter-channel coupling, the real (continuous-record Hilbert) and surrogate (window Hilbert) pipelines must give matching wPLI distributions with no systematic offset. Engineering check, not a scientific result.
- **PC-2 (biological sanity check; not a gate).** Per eligible IED patient, the median across same-state windows of global mean wPLI per primary band (θ, α, β), IED minus non-IED windows; patient-level sign/Wilcoxon test, Holm-adjusted across bands. Direction expected from LIT-017 and LIT-023. Reported whatever its outcome; failure is discussed, not disqualifying.
- Optional estimator sanity check: FC discriminates wake from sleep in non-IED epochs.

**Gate.** A bounded null (issue 3) is interpretable only if PC-1 and PC-3 pass.

### Issue 5 — Waveform/spectral baseline — DECIDED (2026-10-10)
A nested ladder on the issue-1 window.
- **B0 (single-channel max).** Per-channel features, model applied per channel, epoch score = maximum over channels (precedent LIT-052; Nhu 2023).
- **B1 (19-channel waveform + spectral).** Per channel: peak-to-peak amplitude, maximum absolute slope, line length, Teager energy, kurtosis, skewness, sharpness of the dominant peak (half-wave durations), zero-crossing rate, Hjorth activity, mobility and complexity; absolute and relative band power δ 1–4, θ 4–8, α 8–13, σ 11–16, β 13–30, low-γ 30–45 Hz. σ and δ power double as state proxies (issue 9). Template: Ao 2026 (LIT-047). About 19 × 20 features.

### Issue 6 — Multichannel non-FC spatial baseline — DECIDED (2026-10-10)
**B2 = B1 + spatial-extent and topography summaries** from per-channel amplitude and power only: number of channels whose peak salience exceeds a fixed z threshold; max/median ratio across channels of peak-to-peak and Teager energy; amplitude-weighted 10–20 centroid (x, y) and its spread; left–right and anterior–posterior power asymmetry per band.

**B2 is the reference model for the primary estimand.** Primary contrast: B2 + FC vs B2. Inter-channel timing features are left out of B2 because they are a form of coupling; optional harder-baseline sensitivity.

### Issue 7 — Anatomical-adjacency comparator — DECIDED (2026-10-10)
**B2 + ADJ.** Fixed graph over standard 10–20 coordinates, Gaussian weights w_ij = exp(−d_ij² / σ²), σ = median nearest-neighbour distance, no data-driven edges. For every B1 feature: the neighbour-weighted mean and the channel-minus-neighbour contrast. No phase, correlation or coupling information.

Key secondary contrast: (B2 + FC) vs (B2 + ADJ). It may carry an issue-3 taxonomy label marked "secondary, not multiplicity-controlled" (R10) and is never presented as confirmatory.

Attribution: our classical-feature operationalisation of the anatomical-prior idea in Wang 2026 (LIT-046), not a reimplementation of SPID-Net.

### Issue 8 — Surrogate-FC control — DECIDED (2026-10-10; realisation count and generation scheme D1, 2026-10-10)
**B2 + FC_surr.** For each window, each channel is replaced by an independent IAAFT surrogate (Schreiber & Schmitz 1996 [VERIFY]), preserving that channel's power spectrum and amplitude distribution while destroying inter-channel phase relations. FC_surr uses exactly the FC pipeline and has the same dimension.
- **Seeds:** deterministic and recorded; each seed derived from (global seed, recording ID, epoch index, realisation index), so any surrogate can be regenerated.
- **Realisations:** a small fixed number per window, target 3–5, aggregated (mean FC per edge across realisations) rather than one random draw. If this materially increases runtime, the cost is documented before the final number is fixed (before lock).
- **Runtime documented 2026-10-10** (`protocol/2026-10-10-surrogate-runtime-note.md`; synthetic noise only): IAAFT 0.32 s per 2-s window and 0.81 s per 4-s window per realisation, single core. 2-s primary θ/α/β with per-band surrogates: about 20 / 27 / 34 CPU hours for k = 3 / 4 / 5. Recommendation: k = 5 per band for the 2-s primary, k = 3 for the 4-s sensitivity and δ. Open detail: per-band vs broadband surrogate generation (per band recommended, so filtering matches the real FC pipeline). **Decision D1 (Josue, 2026-10-10): APPROVED.** Surrogates are generated per band (IAAFT on each band-passed window cut from the continuously filtered signal); k = 5 for the 2-s primary θ/α/β analysis; k = 3 for the 4-s sensitivity analysis and exploratory δ; IAAFT stops at 200 iterations or when the change in relative spectral error falls below 1e-8. Exact pipeline in §4.3.

Readings: (B2 + FC_surr) vs B2 isolates capacity, dimensionality and amplitude effects; (B2 + FC) vs (B2 + FC_surr) is the gain attributable to coupling. The amplitude control is the combination of wPLI (amplitude-insensitive), amplitude features already in B1/B2, and this surrogate.

### Issue 9 — State-control strategy — DECIDED (2026-10-10)
1. Primary estimand is within-patient and state-matched (issue 2).
2. Mixed-state and unlabelled epochs are excluded from the primary analysis set.
3. Training sample weights equalise, within each training fold, the state distribution of negatives to that of positives. **Formula (R9, adopted 2026-10-10):** within each outer training set, each patient's epochs first get weight 1/n_p (n_p = that patient's epochs in the training set); negatives are then multiplied by π⁺_s / π⁻_s, the patient-weighted state shares of positives and negatives in that training set, so patient totals become approximate. Weights depend on labels and state only and are identical for every model in a contrast. Inner-CV weights are recomputed inside each inner training set.
4. State-proxy features (δ and σ power) are in every baseline.
5. The annotated state label is not a model input.
6. All primary and key secondary results are reported per state; wake is flagged as underpowered (effective N ≈ 7.4 IED patients).
7. Confound quantification (RQ-2): a state-only classifier (label, and EEG state proxies) on patient-level IED status.
8. Residual sleep-depth confound: sensitivity analysis restricting within-patient negatives to the δ-power range of that patient's IED windows.

### Issue 10 — Patient-grouped evaluation and cluster-aware inference — DECIDED (2026-10-10; pilot and fold manifest accepted after F-3; inference after the biostatistics-lens review)
**Josue's decision (2026-10-10).** F-3 is COMPLETE and accepted as generated. The pilot is frozen at 8 recordings (5 evaluable IED-positive, 3 IED-free); drawing the IED-positive pilot recordings from the evaluable group (E) is approved because the pilot must support the baseline-ceiling check for the primary estimand. The confirmatory primary set is 69 recordings, including 36 evaluable IED-positive patients. The frozen outer design is 5 patient-grouped folds × 10 repeats from the committed manifest and hashes. The pilot can never return to the confirmatory set. **Neither the pilot list nor the fold manifest may be redrawn, modified or regenerated unless a formal protocol-change RDR is approved.** The inference and CV-variance text below stays subject to the biostatistics review required before lock.
**Text (Josue's revision, 2026-10-10).**
- **Pilot subset (approved 2026-10-10): 8 recordings, 5 IED-positive and 3 IED-free.** Selected only after F-4, by seeded repo code, from the QC-eligible population (primary rule), stratified by IED status and state coverage. Once drawn, the list is frozen and SHA-256 hashed, and it **can never return to the confirmatory cohort**. Used only for pipeline debugging, anchor tuning and the baseline ceiling check. Basis: F-2 showed pilot size barely affects the central scenario; where precision is tight each extra IED recording in the pilot costs about 0.01 of equivalence probability.
- **F-3 COMPLETE 2026-10-10, accepted by Josue** (`protocol/2026-10-10-F3-pilot-and-folds.md`). Code `src/design/f3_pilot_and_folds.py` at 5cc0058 (seeds fixed in code before the run); outputs `artifacts/protocol/f3/` at 93a2c49. Inputs: the committed F-4 `channel_qc.csv` and `subject_state_matrix.csv` only.
  - Frozen outputs and SHA-256 (also in `artifacts/protocol/f3/SHA256SUMS`):
    - `pilot_subset.csv`: `4645df56844998a1fb1a81e5c23a4e2bacea7de69671998f73083f40710bec29`
    - `fold_manifest.csv`: `7f0f53fe83456386efb4a2ffd16db34ac169b7a7880f36d921bcf46fa2062c5d`
    - Pilot IDs: DA001008 (F), DA00100G (F), DA00100U (E), DA00102U (E), DA00103A (E), DA00103Q (E), DA00103T (E), DC11304C (F).
    - Repeat seeds 2026101001–2026101010; pilot seed 20261010. `python src/design/f3_pilot_and_folds.py --verify` must print MATCH.
  - Pilot: 5 evaluable IED-positive recordings (3 sleep-, 1 wake-, 1 wake+sleep-evaluable) and 3 IED-free (2 wake, 1 wake+sleep), allocated to state-coverage strata by largest remainder.
  - Primary analysis set: 69 recordings, of which 36 are IED-positive patients evaluable for the primary estimand (14 evaluable in wake, 30 in sleep), 6 IED-positive not evaluable, 27 IED-free.
  - Fold manifest: 5 folds × 10 repeats, recording-grouped, stratified by group × state coverage; at least 7 evaluable patients in every test fold. The 7 QC-ineligible recordings are dealt into the same folds after the primary set and marked `sensitivity_only`.
  - **Precision consequence (recorded 2026-10-10).** F-4 removed exactly 5 of the 46 pre-QC evaluable IED-positive patients (41 remain). After the 5 evaluable IED-positive pilot recordings are removed, the primary confirmatory estimand has **36 evaluable IED-positive patients**. This corresponds closely to the F-2 "pilot 8, QC loss 5" scenario (≈ 37 patients; P(equivalence | Δ = 0) ≈ 0.998 central, ≈ 0.67–0.72 with strongly heterogeneous gains). No SESOI change is triggered; SESOI stays 0.03 (issue 3).
  - Pilot IED-positive recordings come from the evaluable group (E): approved by Josue 2026-10-10. Inner CV splits are not frozen in the manifest; training code derives them inside each outer training set with the same seeded dealing (inner fold count still to fix in §4.1).
- **Fold manifest.** Patient-grouped, stratified (IED / non-IED; state coverage) 5-fold outer CV, repeated 10 times with recorded seeds. Created once by repo code, SHA-256 hashed, committed before any FC result exists. Every model uses it. Mixed-class and mixed-state recordings stay whole.
- **Nested tuning.** Patient-grouped inner CV inside each outer training set for every fitted choice. Equal tuning budget for every model in the ladder.
- **Training weights.** Each training patient contributes equal total weight.
- **Inference (R7, R8, Q2; adopted 2026-10-10).**
  - **Primary interval: Nadeau–Bengio corrected repeated-CV interval** (Nadeau & Bengio 2003 [VERIFY]; applied to repeated k-fold as in Bouckaert & Frank 2004 [VERIFY]). For each of the J = 50 repeat × outer-fold cells j, Δ̄_j = mean of Δ_{p,r} over the evaluable patients in that test fold. S² = sample variance of the 50 Δ̄_j; σ̂² = (1/50 + 1/4) S², where 1/4 = n_test / n_train for 5 folds. CI_{1−α} = Δ̂ ± t_{n_E − 1, 1−α/2} · σ̂, with the 90% and 95% intervals from the same σ̂ and n_E the number of evaluable patients (35 after F-6). Δ̂ as in issue 2 (R2).
  - **Sensitivity interval only: patient-cluster percentile bootstrap.** Resample the n_E repeat-averaged Δ_p with replacement, 10,000 resamples, seed 20261010; 90% and 95% percentile intervals. The repeats enter only through Δ_p. It omits training-set variance and is expected to be narrower than the primary. Its label is reported beside the primary label, never used to relabel (issue 3, R5).
  - **Why.** In the AI-simulated biostatistics-lens synthetic refit check (frozen 36/6/27 design, 5 × 10), the patient-cluster bootstrap (percentile or BCa) covered 0.72–0.80 at nominal 0.90, because the missing variance is the training-set component shared by all 50 fold-repeats; the Nadeau–Bengio interval covered 0.90–0.93. No epoch-level DeLong tests or epoch-level p-values (design effect 7–25; confounding report §6).
- **Review status (Q5, Josue 2026-10-10).** The AI-simulated biostatistics-lens review (`reviews/2026-10-10-biostatistics-lens-review.md`), with R1–R10 incorporated, satisfies the **internal** pre-lock review requirement. **It is not a human biostatistician's sign-off.** A human statistical/methodological review of issue 10 is **required before manuscript submission**, and preferably before final interpretation of the confirmatory result.
- **Pre-confirmatory engineering validation EV-1** (§6.1) re-runs the coverage check with the actual learner and nested tuning before the confirmatory contrast is unblinded.
- **No fixed held-out vEpiSet test set.** Confirmatory status is protected by the frozen protocol, the pre-committed manifest, the frozen pilot exclusion, and running the confirmatory CV once.

### Issue 11 — Montage / re-reference — DECIDED (2026-10-10)
**CAR over the 19 scalp rows** primary, computed on QC-eligible recordings.
- All rows share one common reference (VERIFIED), so x_i − mean_j(x_j) removes it exactly. Because the reference is (C3 + C4)/2, the stored C3 and C4 rows are near mirrors (C3 = −C4 to 3 dp in 35 recordings); CAR recovers both, and the stored referential rows are never used for FC.
- Lineage alignment: Mohammed's FIU work used average reference with wPLI (`13`).
- **Sensitivity:** longitudinal bipolar (double banana, 18 derivations), adjacency graph on derivation midpoints.

### Issue 12 — 19-channel QC / bad-channel eligibility — DECIDED (2026-10-10)
**Josue confirmed the rule as run (2026-10-10), with the F-4 results and the frontal extreme-amplitude flags in view. No criterion was changed.**
**F-4 result (2026-10-10, run on Josue's Mac at commit 8c5894a; outputs commit a458458):** stop rule **CONTINUE**, exactly at threshold: 41 of 46 pre-QC-eligible IED-positive recordings pass the primary rule. 77 of 84 recordings are primary-eligible (47 IED-positive, 30 IED-free); all 84 are sensitivity-eligible (7 recordings have 1–2 non-adjacent bad channels). DA00100Y passes (no bad channel; C3/C4 satisfactory, r = −0.78). Epoch flags: 33 non-IED epochs, 0 IED epochs. Details: `protocol/2026-10-10-F4-channel-qc-run.md`.

**Text (Josue's revision, 2026-10-10).** Label-blind, thresholds fixed in code before the run, run once over all 84 recordings on Josue's Mac; aggregate outputs only.
- **Recording-level criteria, per scalp channel (PREP-style; Bigdely-Shamlo et al. 2015 [VERIFY]):** flat (robust SD < 1 µV, or > 5% of samples with zero first difference); extreme amplitude (robust z of log-SD across channels > 5); low neighbour correlation (95th-percentile correlation with its anatomical neighbours, CAR, 1–30 Hz, < 0.4 in > 50% of 4-s windows); clipping (> 0.1% of samples at the channel's extreme value).
- **C3/C4 check (needed for DA00100Y).** C3/C4 correlation, var(C3 + C4)/var(C3 − C4), and each of C3 and C4's neighbour correlation after CAR. Satisfactory C3/C4 behaviour = both pass the neighbour-correlation criterion and neither is flagged by the other criteria.
- **Epoch level:** flag only flat, clipped or disconnected segments. No amplitude-threshold epoch rejection (IEDs are high-amplitude; amplitude rejection would remove positives asymmetrically). Exclusion rates reported by class.
- **Primary eligibility (confirmatory FC analysis):** a recording is eligible only if it has **no bad scalp channels** and satisfactory C3/C4 behaviour. **No interpolation in the primary analysis**, because interpolation manufactures spatial dependence and contaminates the FC quantity under study.
- **Sensitivity only:** recordings with ≤ 2 non-adjacent bad channels, spherical-spline interpolated, applied identically to all representations.
- **F-4 ran first (COMPLETE 2026-10-10)** and met the stop rule; Josue accepted the result under the pre-specified rule. The 7 QC-ineligible recordings remain sensitivity-only. No threshold may be changed after the results were seen.
- **F-4 stop rule (fixed 2026-10-10, before the run):** continue only if **≥ 41 of the 46 pre-QC-eligible IED-positive recordings** (§2) pass the primary rule, i.e. QC removes ≤ 5. This is the QC loss the F-2 central scenario covered. If more than 5 are removed, stop and return for a protocol decision. Total survivors and IED-free survivors are reported but do not trigger the stop.
- Implementation: `src/data/vepiset_channel_qc.py` (thresholds in its `CRITERIA`, fixed before the run). QC-only details fixed with the code: QC uses a per-sample median-across-channels reference so one bad channel cannot contaminate the others' metrics (the analysis montage stays CAR); flat and clipping checks use the stored referential rows; extreme-amplitude z uses a floor of 0.1 on the scaled MAD of log-SD; neighbours are 10–20 electrodes within 0.65 head radii; the interpolation sensitivity set has no C3/C4 condition beyond the ≤ 2 non-adjacent bad-channel rule.
- The same eligible set is used for baseline, adjacency, surrogate and FC models, so every comparison stays paired.

### Issue 13 — Explicit ECE engineering contribution — DECIDED (2026-10-10)
The engineering contribution is an **open, reproducible, leakage-controlled evaluation framework for testing the incremental value of network features in EEG event detection**, consisting of:
1. Label-blind salience-selected windowing that makes positives and negatives symmetric (issue 1).
2. A nested comparator ladder (B0 → B1 → B2 → +ADJ → +FC_surr → +FC) run on one hashed patient-fold manifest with equal tuning budgets (issues 5–8, 10).
3. State-matched within-patient evaluation and confound quantification, with measured inflation from epoch-level splits and state imbalance on vEpiSet (RQ-2).
4. A signal-provenance and reference-normalization layer that identifies and removes the undocumented common reference (`05`). Re-referencing itself is standard practice and is not claimed as a novel method.
5. Release as a Python package with configs and tests, reusable for any feature family, not only FC.

Not a new algorithm; it is the instrument that makes the scientific answer defensible. Dissertation chapter map: A evaluation validity (RQ-2); B incremental value of FC (RQ-1); C external corroboration (RQ-3); D spatial type, descriptive (RQ-4).

### Issue 14 — External-validation role of TUEV and/or BDSP — DECIDED at policy level (2026-10-10, D8)
**Decision D8 (Josue, 2026-10-10).** The role, the BDSP → TUEV → frozen-model-transfer order and the rule that a corroboration criterion is fixed before any external data are analysed are DECIDED. Adopting a specific external dataset needs its own RDR once access exists; this is not a blocker for locking `06`.
**Text (Josue's revision, 2026-10-10).**
- External data are **secondary and outside the Journal Draft v1 confirmatory claim**.
- **Goal: external corroboration of the incremental-value conclusion**, using the frozen representation and comparator framework (window rule, B0–B2, ADJ, FC_surr, FC, learner, fold discipline) with a **dataset-appropriate patient-grouped estimand**. The identical vEpiSet primary estimand is not required, because external labels and state structure differ.
- Corroboration criterion: fixed per dataset before its data are analysed, CI-based, and stated in the issue-3 taxonomy; not "same sign".
- **BDSP "Measuring Expertise in Identifying IEDs" is preferred** if its patient IDs or grouping and its clip structure support valid patient-grouped evaluation (to check on access). Its 128 Hz sampling caps analysis at 64 Hz, which the θ/α/β bands fit.
- **TUEV is the fallback.** Pre-specified IED definition = SPSW only; GPED and PLED (periodic) excluded, unlike Wang's grouping. No sleep labels, so state control is unavailable there and must be stated. Known asymmetric transfer (LIT-034).
- **Tertiary:** frozen-model transfer (vEpiSet-trained models scored on the external set), threshold-free metrics plus pre-specified threshold handling.
- Access for both is pending. Adoption still needs a recorded decision (`05` External Dataset Policy).

---

## 4. Other elements fixed before lock (§4.1–4.4 APPROVED 2026-10-10, D2–D5; exact wPLI specification APPROVED 2026-10-10)

### 4.1 Learner — APPROVED (2026-10-10, D2, with inner-fold validity rule)
Primary: gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`), so the baseline is as strong as reasonably possible. Secondary: L2 logistic regression on standardised features. Same folds, budget and seeds. A credible published DL model is context only, never the comparator for H-1; dissertation scope unless time allows.
- **Inner CV:** 4 recording-grouped folds inside each outer training set, stratified by the F-3 group × coverage strata, dealt with the frozen F-3 dealing function, seed = repeat seed × 10 + outer fold (`src/design/inner_folds.py`).
- **Inner-fold validity rule.** An inner split is valid only if (1) every inner validation fold holds ≥ 5 evaluable IED-positive patients (group E), so the selection metric is defined in every fold; (2) every inner training part holds ≥ 1 wake-evaluable and ≥ 1 sleep-evaluable E patient, so the state-balanced weights are defined; (3) inner validation folds partition the outer training set and contain no outer-test recording. If 4 folds fail for an outer fold, that outer fold uses 3; if 3 also fail, the run stops and the failure is recorded as a protocol deviation for Josue. No other fallback. **Checked 2026-10-10 on the frozen manifest (counts only; `artifacts/protocol/f3/inner_fold_check.json`): all 50 outer splits valid with 4 inner folds; every inner validation fold holds 7–8 E patients.** If R3 of the biostatistics review is adopted, the check is rerun on the post-exclusion evaluable set before any model output exists.
- **Tuning grid**, fixed and identical for every ladder model (B0, B1, B2, +ADJ, +FC_surr, +FC, PC-1 FC-only): HGB 16 configurations (learning_rate ∈ {0.05, 0.1}; max_leaf_nodes ∈ {15, 31}; min_samples_leaf ∈ {20, 100}; l2_regularization ∈ {0, 1}; max_iter 300). Logistic regression: C ∈ 8 log-spaced values from 1e-3 to 1e2.
- **Selection metric:** mean within-patient, same-state AUROC over the E patients of the inner validation folds, equal weight per patient; ties go to the simpler configuration.
- **Training:** equal total weight per patient with the issue-9 state balancing; feature scaling fitted on training data only. **Seeds:** learner random_state = repeat seed + outer fold, recorded.

### 4.2 Exclusions — APPROVED (2026-10-10, D3)
- The 12 IED epochs with no reconstructable marker and the 50 non-IED epochs containing a marker (VERIFIED, `05`): excluded; sensitivity includes them.
- The 65 short final epochs: excluded.
- The first and last full 4-s epoch of every recording: excluded from all analyses and all representations, label-blind, because filter and Hilbert transients are largest at the recording ends (wPLI specification step 4, approved 2026-10-10). The number of IED epochs removed is reported.
- Non-IED epochs adjacent to an IED epoch: excluded from training and evaluation negatives; sensitivity includes them.
- APPROVED (D3, 2026-10-10; from F-5): record each non-IED epoch's temporal distance to the nearest IED marker and report results by distance bin, since the dataset owners treat peri-IED negatives as distinct (LIT-039). Descriptive only.
- Mixed and unlabelled state epochs: excluded from the primary set.

### 4.3 FC representation — APPROVED (2026-10-10, D4; exact estimator specification APPROVED as written 2026-10-10)
- Estimator: weighted phase lag index (wPLI; Vinck et al. 2011 [VERIFY]), from the analytic signal of band-filtered CAR data (filtered on the continuous recording), averaged over the samples of the issue-1 window.
- **Primary bands on the 2-s salience-selected window: θ 4–8, α 8–13, β 13–30 Hz.**
- **δ 1–4 Hz:** exploratory on the 2-s window; evaluated mainly in the 4-s native-epoch sensitivity analysis.
- Low-γ 30–45 Hz: exploratory (muscle-artefact risk; below the ~70 Hz high-cut).
- Primary FC features: the 171 upper-triangle edges per primary band (513). Secondary: node strength per channel and band (57) plus global mean per band.
- FC_surr (issue 8) uses exactly this pipeline.

**Exact wPLI specification (APPROVED as written by Josue, 2026-10-10).**
1. **Input.** The continuous MAT `eeg_data` rows 1–19 of each primary-eligible recording, float64, re-referenced to CAR over those 19 rows (issue 11).
2. **Band-pass.** For each band (δ 1–4, θ 4–8, α 8–13, β 13–30, low-γ 30–45 Hz), a Butterworth band-pass of order 4, `scipy.signal.butter(4, [lo, hi], btype="bandpass", fs=500, output="sos")`, applied zero-phase with `scipy.signal.sosfiltfilt` (default odd padding) to the **whole continuous recording**, per channel.
3. **Analytic signal.** `scipy.signal.hilbert` on the whole continuous band-passed recording, per channel, giving z_c(t).
4. **Recording edges.** Filter and Hilbert transients are largest at the recording ends, so the first and last full 4-s epoch of every recording are excluded from all analyses and all representations (label-blind; the number of IED epochs this removes is reported). This adds one item to §4.2.
5. **Window.** The issue-1 salience-selected 2-s window (1,000 samples); for the 4-s sensitivity, the whole epoch (2,000 samples).
6. **wPLI per window, band and pair i < j.** X_ij(t) = Im(z_i(t) · conj(z_j(t))) over the window's samples; wPLI_ij = |Σ_t X_ij(t)| / Σ_t |X_ij(t)|, in [0, 1]. If Σ_t |X_ij(t)| = 0 the value is 0 and the event is counted. This is the within-window, sample-averaged form of Vinck et al.'s wPLI (2011 [VERIFY]), which was defined over trials; samples within a window are autocorrelated, so it is a per-window connectivity descriptor and its finite-sample bias is not corrected (no debiased wPLI). The bias depends on window length and band, which FC and FC_surr share.
7. **Features.** 171 edges per band in the stored channel order (Fp1 … Pz), (i, j) row-major over the upper triangle; primary = θ, α, β (513). Secondary: node strength s_i = mean over j ≠ i of wPLI_ij (19 per band) and the global mean per band. δ and low-γ are exploratory.
8. **FC_surr (D1).** For each window, band and realisation: take the band-passed real-valued window from step 2 (before the Hilbert transform), replace each channel by an independent IAAFT surrogate (≤ 200 iterations, stop when the change in relative spectral error is < 1e-8), compute the surrogate's analytic signal with `scipy.signal.hilbert` on the window itself, and apply step 6. Average wPLI per edge across the k realisations (k = 5 primary, 3 for 4-s and δ). Seed = the integer from the first 8 bytes of SHA-256 of "fcsurr:20261010:<eeg_id>:<epoch start sample>:<band>:<realisation>:<channel>".
9. **Known asymmetry and its check.** The real analytic signal comes from the continuous recording, while the surrogate's comes from the window, because a surrogate exists only for the window. PC-3 therefore also checks, on synthetic data with no inter-channel coupling, that the real and surrogate pipelines give matching wPLI distributions (no systematic offset), in addition to recovering an injected lag and destroying it in surrogates.
10. **Implementation.** numpy and scipy at the pinned versions (`pyproject.toml`); no other FC library. Written and tested only after lock.

### 4.4 Multiplicity — APPROVED (2026-10-10, D5)
One confirmatory test (H-1). Key secondary contrasts (B2 + FC vs B2 + ADJ; B2 + FC vs B2 + FC_surr; deployment metrics) are estimates with CIs, Holm-adjusted if any is described as a test. Everything else (δ and low-γ, node strength, montage, 4-s window, state strata, learners, interpolation, RQ-4) is descriptive.
- **R10 (adopted 2026-10-10).** The taxonomy applied to the primary contrast is the single confirmatory decision. Secondary contrasts may carry taxonomy labels marked "secondary, not multiplicity-controlled" and may not appear as confirmatory in the abstract or conclusions. State strata, including wake (n = 13 after F-6), are reported as estimate + percentile-bootstrap CI without outcome labels. The primary and sensitivity intervals are not two chances at a label.

### 4.5 Reproducibility before the first registered experiment (Adjouadi-lens R8)
- Configs: **done 2026-10-10** (`configs/protocol_v1/`, commit 26ec921): common design, wPLI/IAAFT specification, one config per ladder model; the outcome taxonomy and both intervals are fixed in code (`src/evaluation/taxonomy.py`) with tests. Learner libraries (scikit-learn) are pinned when the implementation starts after lock.
- Pinned dependencies in `pyproject.toml` with a lockfile: **done 2026-10-10 on the 03 branch** (commit 3ddb389: numpy 2.5.3, scipy 1.18.1, pytest 9.1.1; `uv.lock` and hashed `requirements-lock.txt`; full suite passes in a clean environment built from the lock). Analysis dependencies are added, pinned, when the locked protocol needs them.
- Configs in `configs/` for every model in the ladder; fold manifest and pilot list generated by code, hashed, committed.
- Locked `06` text committed to the repo under `experiments/protocol/` with a timestamp before any FC result is computed.
- Experiments registered in `07` before they run.

---

## 5. Robustness and error analysis (descriptive; APPROVED 2026-10-10, D6)
- Robustness: native 4-s window (incl. δ FC); bipolar montage; logistic-regression learner; inclusion of excluded epochs; interpolated recordings added (issue 12 sensitivity); sleep-depth-restricted negatives; epoch-weighted Δ; ≥ 3 and ≥ 10 eligibility thresholds.
- Error analysis: per-patient Δ distribution (forest plot), patients where FC helps or hurts, by state and by IED type (generalized vs focal), false positives in IED-free patients by state.

## 6. Pre-lock checks (small, each needed for a specific decision)
| ID | Check | Why it is needed | Where it runs | Touches the FC contrast? |
|---|---|---|---|---|
| F-1 | Count IED markers by position within the 4-s epoch (outer 1.0 s at each end) — **COMPLETE 2026-10-10** (0.361 > 0.10, returned to Josue, who adopted the whole-epoch search with clamped centre; issue 1 DECIDED; no rerun required; run on Josue's Mac, outputs 8a6f0b8; code b18d632; rule fixed before run: patient-weighted outer-only share over the 36 confirmatory evaluable patients ≤ 0.10 keeps the anchor rule, > 0.10 returns to Josue; `protocol/2026-10-10-F1-marker-position-run.md`) | Decides whether the [1.0, 3.0] s anchor search loses too many IEDs | Josue's Mac, existing interim event tables | No |
| F-2 | Precision-feasibility simulation of the issue-2 estimand — **DONE 2026-10-10, MARGINAL** | Decides whether SESOI ±0.03 is plausibly supportable (issue 3) and the pilot size (issue 10) | Cloud, committed aggregate counts only | No |
| F-3 | Draw the frozen pilot subset and the fold manifest after QC — **COMPLETE 2026-10-10, accepted by Josue** (code 5cc0058, outputs 93a2c49; `protocol/2026-10-10-F3-pilot-and-folds.md`) | Fixes the confirmatory set before any model runs | Repo code on committed aggregate tables (cloud); reproducible on the Mac with `--verify` | No |
| F-4 | Run the issue-12 QC over 84 recordings — **COMPLETE 2026-10-10: CONTINUE, 41 of 46; accepted by Josue under the pre-specified rule** (code 8c5894a, outputs a458458; `protocol/2026-10-10-F4-channel-qc-run.md`) | Decides whether the no-bad-channel rule is affordable; DA00100Y | Repo code, Josue's Mac | No |
| F-5 | Re-run the vEpiSet citing-works search; check `vepiset/peri-ied-eeg-dynamics` for a paper — **RUN 2026-10-10, PARTIAL: no new pre-emption found**; citing-works listings still unreadable (registry counts unchanged at 16 = the known 16); LIT-039 still unpublished, WATCH; PMID 40850814 abstract unread (`novelty/2026-10-10-F5-novelty-recheck.md`). **Accepted as partial for lock by Josue (D10, 2026-10-10); a full citing-works and LIT-039 recheck is mandatory before manuscript submission** | Novelty-gate condition before lock | Web | No |
| F-6 | R3 recount: evaluable patients after all §4.2 exclusions (incl. first/last epoch) and epoch QC, labels and counts only; then the inner-fold validity rule on the recounted set — **COMPLETE 2026-10-10: E = 35 (36 → 35; DA00103R lost evaluability), rule result RECORD CHANGE IN 11, no RDR; inner-fold rule passes on all 50 outer splits** (code b25c9dc; outputs 8a29603; `11` LOG-2026-10-10-F6; `protocol/2026-10-10-F6-evaluable-recount-run.md`) | Fixes the final confirmatory evaluable set before any model output (issue 2, Q4) | Josue's Mac (needs signal for the F-4 epoch flags) | No |

### F-2 specification (fixed 2026-10-10, before running)
F-2 is a **precision-feasibility analysis, not a power guarantee**. It uses only the per-recording state × class counts in the committed `subject_state_matrix.csv` and assumed score distributions; no signal, feature or model is involved. Model-specific quantities (baseline AUROC, how much FC changes scores, CV refit noise) are unknown and enter as scenario grids, so the output is a range, not a prediction.

Simulation model:
- Eligible IED-positive patients as in §2 (≥ 5 IED and ≥ 5 non-IED same-state epochs), using each patient's real per-state epoch counts.
- Removal scenarios: pilot 0 / 6 / 8 / 10 recordings (IED share by stratification ≈ 52/84) × extra QC loss 0 / 5 / 10 IED recordings, removed at random per replicate.
- Within-patient scores: binormal. Baseline separation d_p ~ N(μ_d, 0.5) on the probit scale, μ_d calibrated so the mean within-patient AUROC is 0.85, 0.90 or 0.95. FC model: d_p + g_p, g_p ~ N(γ, τ_g), τ_g ∈ {0.10, 0.25, 0.50}; epoch-level correlation between the two models' scores r ∈ {0.80, 0.90}. γ calibrated so the population-mean Δ is 0 (null) or 0.03.
- CV refit noise: 5 folds × 10 repeats with patients randomly assigned per repeat; each fold-repeat adds a shared shift in Δ ~ N(0, σ_f), σ_f ∈ {0, 0.01, 0.02}.
- 1,000 replicates per scenario. Per replicate: per-patient AUROCs by state (Mann–Whitney), combined per issue 2, equal-weight mean Δ, and a 90% CI two ways: (a) "naive" t-interval over patients (what a patient bootstrap approximates); (b) "honest" using the across-replicate SD of the estimate.

Reported per scenario: number of eligible patients; median 90% CI half-width (naive and honest); naive-CI coverage; P(equivalence declared | true Δ = 0) using the honest interval; P(95% CI lower bound > 0 | true Δ = 0.03).

Pre-specified reading:
- **Central scenario:** mean baseline AUROC 0.90, τ_g = 0.25, r = 0.90, σ_f = 0.01, pilot 6–8, QC loss 0–5.
- **Pessimistic scenario:** τ_g = 0.50, r = 0.80, σ_f = 0.02, pilot 8, QC loss 10, baseline AUROC 0.85 and 0.90.
- **Supportable:** P(equivalence | Δ = 0) ≥ 0.80 in the central scenario and ≥ 0.50 in the pessimistic scenario.
- **Marginal:** central ≥ 0.50 but below the supportable rule. Report to Josue with options; do not mark issue 3 DECIDED.
- **Not realistically supportable:** central < 0.50. Stop and report (Josue's instruction, 2026-10-10).
- Pilot size: the largest of 6 / 8 / 10 whose central P(equivalence | Δ = 0) is within 0.05 of the 6-recording value, capped at 8 unless that rule supports more.

### 6.1 Pre-confirmatory engineering validations (do not block lock; must pass before the confirmatory FC contrast is unblinded)
- **EV-1 (Josue, 2026-10-10).** Port and re-run the synthetic CI-coverage check of the biostatistics-lens review using the actual HGB learner and the nested-tuning structure (§4.1), if computationally feasible, on synthetic data only. It must run before the confirmatory FC contrast is unblinded. If it shows materially inadequate coverage for the locked Nadeau–Bengio method, stop before confirmatory interpretation and return for an RDR. "Materially inadequate" is fixed before EV-1 runs, in its run sheet, and approved by Josue.
- **PC-3** (issue 4) runs in the test suite with the FC implementation, before any confirmatory run.
- **F-6** (§6) must be complete before any model output exists.

## 7. Lock preconditions (checklist)
- [x] Dataset audit sufficient (structural audit + signal provenance, `05`)
- [x] Novelty review sufficient (CAND-007 PASS WITH CONDITIONS; Wang 2026 does not pre-empt)
- [x] Advisor approval of the central direction (ADVISOR-FEEDBACK-2026-10-10; RDR-011)
- [x] All 14 mandatory issues DECIDED (2026-10-10; issue 14 at policy level; issue 10 inference after the biostatistics-lens review, Q2)
- [x] Issue 8 realisation count fixed with runtime cost documented (D1, 2026-10-10)
- [x] F-1 to F-5 done and recorded (F-2 COMPLETE: MARGINAL, accepted; F-3 COMPLETE, accepted; F-4 COMPLETE: CONTINUE, accepted; F-1 COMPLETE, issue 1 decided; F-5 partial accepted for lock by Josue (D10) with a **mandatory pre-submission recheck** of citing works and LIT-039)
- [x] Exact RQ-1 and H-1 wording approved (D7, 2026-10-10)
- [x] Internal biostatistics review of issues 2, 3 and 10: AI-simulated lens review with R1–R10 incorporated (Q5, 2026-10-10). **Not human sign-off; human review of issue 10 required before manuscript submission.**
- [x] Exact wPLI specification approved (§4.3, 2026-10-10)
- [x] Reproducibility items in §4.5 ready for lock (pinned environment, configs, frozen hashed pilot and manifest; locked text committed to the repo at lock)
- [x] F-6 recount run on the Mac and recorded in `11` (Q4): E = 35, no RDR required (LOG-2026-10-10-F6)
- [x] Lock RDR written (RDR-013 in `11`, 2026-10-10, on Josue's explicit lock approval)

## 8. What this protocol will not permit us to conclude
- Nothing about epileptogenic-zone, seizure-onset-zone or irritative-zone localisation (spatial labels are reader judgements of IED field).
- The primary estimand concerns IED-positive patients only; it does not establish deployment performance or false-positive behaviour in IED-free patients.
- A positive Δ shows θ/α/β wPLI adds discriminative information for this dataset, montage, window and learner; it does not show FC is the best representation, or that a learned multichannel DL model lacks the same information.
- A bounded null bounds the increment from this FC representation at this SESOI; it does not show that all network information is redundant, and says little about δ-band coupling, which is exploratory here.
- Within-state results in wake rest on about 7 effective IED patients.

## 9. Protocol change control
After LOCK, material changes require an RDR with reason, evidence, impact on confirmatory status, and affected experiments. Changes triggered by confirmatory results invalidate confirmatory interpretation of the affected analysis.
