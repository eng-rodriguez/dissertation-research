"""PC-3 (06 issue 4) for EXP-001: synthetic test of the FC pipeline (wpli.py, surrogates.py).

Three checks on synthetic 19-channel data at 500 Hz, per primary band:
  lag      - transients with a known inter-channel phase lag are injected into independent 1/f background;
             the real FC pipeline must give the coupled edges high wPLI, rank them on top, and recover the
             lag direction;
  surrogate - per-channel IAAFT surrogates (FC_surr) must make the coupled edges indistinguishable from
             uncoupled ones;
  null     - with no coupling, real wPLI (continuous-record Hilbert) and surrogate wPLI (window Hilbert)
             must match, with no systematic offset (06 §4.3 step 9).
Design, seeds and pass criteria: experiments/EXP-001-pc3/config.json (committed before this code).
EXP-005 (experiments/EXP-005-pc3/config.json; 11 LOG-2026-10-10-PC3) re-tests on fresh seeds with the
surrogate check amended to: lag direction at chance in the surrogates, and surrogate wPLI equal to a
shared-amplitude-spectrum null (each channel keeps its window amplitude spectrum, phases independent).
EXP-006 (experiments/EXP-006-pc3/config.json; 11 LOG-2026-10-10-PC3-EXP006) judges surrogate destruction
against the same-channel surrogate reference: wPLI between the FC_surr surrogate of a channel and a second,
independent surrogate of the same channel (single-channel properties only, no coupling by construction).
Engineering check on synthetic data only; no vEpiSet data is read.

Usage (from the repository root):
    python src/connectivity/pc3.py                                                 # EXP-001
    python src/connectivity/pc3.py --config experiments/EXP-005-pc3/config.json   # EXP-005
    python src/connectivity/pc3.py --config experiments/EXP-006-pc3/config.json   # EXP-006
Writes artifacts/protocol/pc3/pc3_report.json (EXP-001), exp005_report.json or exp006_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surrogates as sg  # noqa: E402
import wpli as wp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "EXP-001-pc3" / "config.json"
BAND_CENTRE = {"theta": 6.0, "alpha": 10.5, "beta": 21.5}
ID_PREFIX = {"EXP-001": "SYN", "EXP-005": "SYN5", "EXP-006": "SYN6", "EXP-007": "SYN7"}  # synthetic eeg_id prefix used in the surrogate seeds
REPORT_NAME = {"EXP-001": "pc3_report.json", "EXP-005": "exp005_report.json", "EXP-006": "exp006_report.json",
               "EXP-007": "exp007_report.json"}


def background(rng: np.random.Generator, n_ch: int, n: int, fs: int) -> np.ndarray:
    """Independent Gaussian channels with a 1/f power spectrum, unit variance."""
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = rng.normal(size=(n_ch, f.size)) + 1j * rng.normal(size=(n_ch, f.size))
    spec[:, 0] = 0
    spec[:, 1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(spec, n=n, axis=-1)
    return x / x.std(axis=-1, keepdims=True)


def window_starts(n_windows: int, win_n: int, fs: int) -> list[int]:
    return [int((6 + 3 * k) * fs) for k in range(n_windows)]


def inject_bursts(x: np.ndarray, band: str, starts, win_n: int, pairs, signs, snr: float, fs: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Add lagged band-limited bursts (1.6-s Hann taper centred in each window) to each coupled pair."""
    x = x.copy()
    bg_rms = np.sqrt(np.mean(wp.bandpass(x, band, fs) ** 2, axis=-1))
    d = int(round(fs / (4 * BAND_CENTRE[band])))
    taper_n = int(1.6 * fs)
    taper = np.hanning(taper_n)
    for (i, j), sign in zip(pairs, signs):
        lead, lag = (i, j) if sign > 0 else (j, i)
        src = np.zeros(x.shape[1])
        for s in starts:
            c0 = s + win_n // 2 - taper_n // 2
            noise = wp.bandpass(rng.normal(size=taper_n + 2 * fs), band, fs)[fs:fs + taper_n]
            burst = taper * noise
            src[c0:c0 + taper_n] = burst * snr * bg_rms[lead] / np.sqrt(np.mean(burst ** 2))
        x[lead] += src
        x[lag] += np.roll(src, d)
    return x


def shared_spectrum_null(win: np.ndarray, seeds) -> np.ndarray:
    """Each row keeps its Fourier amplitude spectrum with independent uniform random phases (DC, Nyquist 0)."""
    n = win.shape[-1]
    mag = np.abs(np.fft.rfft(win, axis=-1))
    ph = np.vstack([np.random.default_rng(sd).uniform(0, 2 * np.pi, mag.shape[1]) for sd in seeds])
    ph[:, 0] = 0
    if n % 2 == 0:
        ph[:, -1] = 0
    return np.fft.irfft(mag * np.exp(1j * ph), n=n, axis=-1)


def ftnull_seed(eeg_id: str, start: int, realisation: int, channel: int) -> int:
    key = f"ftnull:{sg.GLOBAL_SEED}:{eeg_id}:{start}:{realisation}:{channel}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _lag_block(band: str, design: dict, eeg_id: str, rng_bg, rng_burst, same_channel_ref: bool) -> dict:
    """One synthetic lag-test recording: per-window real and surrogate quantities."""
    fs, n_ch, win_n = design["fs"], design["n_channels"], int(design["window_s"] * design["fs"])
    lt, k = design["lag_test"], design["surrogate_realisations"]
    starts = window_starts(lt["n_windows"], win_n, fs)
    x = background(rng_bg, n_ch, int(lt["duration_s"] * fs), fs)
    x = inject_bursts(x, band, starts, win_n, lt["coupled_pairs"], lt["lag_sign"], lt["snr"], fs, rng_burst)
    ei, ej = wp.edge_index(n_ch)
    edge_of = {(a, b): e for e, (a, b) in enumerate(zip(ei, ej))}
    pairs = [tuple(p) for p in lt["coupled_pairs"]]
    coupled = [edge_of[p] for p in pairs]

    z = wp.analytic(wp.bandpass(wp.car(x), band, fs))
    real = np.vstack([wp.wpli(z[:, s:s + win_n])[0] for s in starts])
    signed = np.vstack([wp.signed_imag(z[:, s:s + win_n]) for s in starts])

    # Surrogates exactly as surrogates.surrogate_fc (same seeds), keeping the signed sum for the lag direction,
    # the shared-amplitude-spectrum null and (EXP-006 on) the same-channel reference on the same windows.
    xb = wp.bandpass(wp.car(x), band, fs)
    surr = np.zeros((len(starts), len(ei)))
    hits = np.zeros(len(coupled))
    null = np.zeros((len(starts), len(coupled)))
    ref = np.zeros((len(starts), len(coupled), 2))  # same-channel reference for channels a and b of each pair
    chans = sorted({c for p in pairs for c in p})
    row = {c: i for i, c in enumerate(chans)}
    for w, s in enumerate(starts):
        win = xb[:, s:s + win_n]
        for r in range(k):
            sv, _ = sg.iaaft_rows(win, [sg.surrogate_seed(eeg_id, s, band, r, c) for c in range(n_ch)])
            if same_channel_ref:
                sv2, _ = sg.iaaft_rows(win[chans], [sg.surrogate_seed(f"{eeg_id}-REF", s, band, r, c) for c in chans])
                for q, pair in enumerate(pairs):
                    for h, c in enumerate(pair):
                        ref[w, q, h] += wp.wpli(wp.analytic(np.vstack([sv[c], sv2[row[c]]])))[0][0] / k
            zs = wp.analytic(sv)
            surr[w] += wp.wpli(zs)[0] / k
            si = wp.signed_imag(zs)
            hits += [np.sign(si[e]) == sgn for e, sgn in zip(coupled, lt["lag_sign"])]
            for q, (a, b) in enumerate(pairs):
                zn = wp.analytic(shared_spectrum_null(win[[a, b]], [ftnull_seed(eeg_id, s, r, a), ftnull_seed(eeg_id, s, r, b)]))
                null[w, q] += wp.wpli(zn)[0][0] / k
    return {"real": real, "signed": signed, "surr": surr, "hits": hits, "draws": len(starts) * k, "null": null, "ref": ref}


def _excess(blocks: list[dict], coupled) -> tuple[np.ndarray, np.ndarray]:
    """Per coupled edge and channel: mean and Monte Carlo SE over windows of FC_surr minus the same-channel reference."""
    surr = np.vstack([b["surr"] for b in blocks])[:, coupled]
    ref = np.concatenate([b["ref"] for b in blocks])
    d = surr[:, :, None] - ref
    return d.mean(axis=0), d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])


def lag_and_surrogate(band: str, design: dict, prefix: str = "SYN", same_channel_ref: bool = False,
                      max_se: float | None = None) -> dict:
    lt = design["lag_test"]
    b_idx = design["bands"].index(band)
    seeds = design["seeds"]
    n_ch = design["n_channels"]
    ei, ej = wp.edge_index(n_ch)
    edge_of = {(a, b): e for e, (a, b) in enumerate(zip(ei, ej))}
    coupled = [edge_of[tuple(p)] for p in lt["coupled_pairs"]]
    uncoupled = np.setdiff1d(np.arange(len(ei)), coupled)

    def block(i):
        return _lag_block(band, design, f"{prefix}-LAG-{band}-B{i:03d}",
                          np.random.default_rng([seeds["background_lag"], b_idx, i]),
                          np.random.default_rng([seeds["bursts"], b_idx, i]), same_channel_ref)

    if "blocks" in lt:  # EXP-007: pool fresh blocks until every excess SE <= max_se (looks at SEs only)
        bl = lt["blocks"]
        blocks = [block(i) for i in range(bl["initial"])]
        while _excess(blocks, coupled)[1].max() > max_se and len(blocks) < bl["max"]:
            blocks += [block(i) for i in range(len(blocks), min(len(blocks) + bl["increment"], bl["max"]))]
    else:  # EXP-001, EXP-005, EXP-006: one recording, original seeding
        blocks = [_lag_block(band, design, f"{prefix}-LAG-{band}", np.random.default_rng([seeds["background_lag"], b_idx]),
                             np.random.default_rng([seeds["bursts"], b_idx]), same_channel_ref)]

    real = np.vstack([b["real"] for b in blocks])
    signed = np.vstack([b["signed"] for b in blocks])
    m = real.mean(axis=0)
    fs = design["fs"]
    lag = {"mean_wpli_coupled": m[coupled].tolist(),
           "max_mean_wpli_uncoupled": float(m[uncoupled].max()),
           "coupled_are_top_edges": set(np.argsort(m)[-len(coupled):].tolist()) == set(coupled),
           "share_correct_direction": [float(np.mean(np.sign(signed[:, e]) == sg_)) for e, sg_ in
                                       zip(coupled, lt["lag_sign"])],
           "delay_samples": int(round(fs / (4 * BAND_CENTRE[band])))}
    ms = np.vstack([b["surr"] for b in blocks]).mean(axis=0)
    null = np.vstack([b["null"] for b in blocks]).mean(axis=0)
    hits = sum(b["hits"] for b in blocks) / sum(b["draws"] for b in blocks)
    surrogate = {"mean_wpli_coupled": ms[coupled].tolist(),
                 "uncoupled_mean_range": [float(ms[uncoupled].min()), float(ms[uncoupled].max())],
                 "diff_mean_coupled_vs_uncoupled": float(ms[coupled].mean() - ms[uncoupled].mean()),
                 "share_injected_direction": hits.tolist(),
                 "mean_wpli_shared_spectrum_null": null.tolist(),
                 "diff_vs_shared_spectrum_null": (ms[coupled] - null).tolist()}
    if same_channel_ref:
        surrogate["same_channel_reference"] = np.concatenate([b["ref"] for b in blocks]).mean(axis=0).tolist()
        exc, se = _excess(blocks, coupled)
        surrogate["excess_over_same_channel_reference"] = exc.tolist()
        surrogate["excess_se"] = se.tolist()
    out = {"lag": lag, "surrogate": surrogate}
    if "blocks" in lt:
        out["n_blocks"] = len(blocks)
        out["n_windows"] = int(real.shape[0])
    return out


def null_match(band: str, design: dict, prefix: str = "SYN") -> dict:
    fs, n_ch, win_n = design["fs"], design["n_channels"], int(design["window_s"] * design["fs"])
    nt, k = design["null_test"], design["surrogate_realisations"]
    starts = window_starts(nt["n_windows"], win_n, fs)
    x = background(np.random.default_rng(design["seeds"]["background_null"]), n_ch, int(nt["duration_s"] * fs), fs)
    real = wp.real_fc(x, starts, win_n, bands=(band,), fs=fs)[band]["wpli"]
    s = sg.surrogate_fc(x, f"{prefix}-NULL-{band}", starts, win_n, k, bands=(band,), fs=fs, keep_realisations=True)[band]
    d = real.mean(axis=1) - s["wpli"].mean(axis=1)
    half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
    return {"mean_real": float(real.mean()), "mean_surrogate_kavg": float(s["wpli"].mean()),
            "offset_mean": float(d.mean()), "offset_ci95": [float(d.mean() - half), float(d.mean() + half)],
            "ks_distance": float(stats.ks_2samp(real.ravel(), s["realisations"][:, 0, :].ravel()).statistic),
            "sd_real": float(real.std()), "sd_surrogate_single": float(s["realisations"][:, 0, :].std()),
            "sd_surrogate_kavg": float(s["wpli"].std())}


def _band(args):
    band, design, prefix, ref, max_se = args
    return band, {**lag_and_surrogate(band, design, prefix, ref, max_se), "null": null_match(band, design, prefix)}


def run(cfg: dict, workers: int | None = None) -> dict:
    design = cfg["synthetic_design"]
    ref = "max_excess_over_same_channel_reference" in cfg["criteria"]["surrogate_destruction"]
    max_se = cfg["criteria"]["surrogate_destruction"].get("precision", {}).get("max_se_per_excess")
    jobs = [(b, design, ID_PREFIX[cfg["experiment"]], ref, max_se) for b in design["bands"]]
    if workers == 1:
        return dict(map(_band, jobs))
    with ProcessPoolExecutor(max_workers=workers or min(len(jobs), 3)) as ex:
        return dict(ex.map(_band, jobs))


def evaluate(report: dict, crit: dict) -> dict:
    """Per band and criterion: True/False. Mirrors the gate tests in tests/test_pc3_*.py."""
    out = {}
    for b, r in report.items():
        lc, sc, nc = crit["lag_recovery"], crit["surrogate_destruction"], crit["null_coupling_match"]
        out[b] = {
            "lag_min_wpli": min(r["lag"]["mean_wpli_coupled"]) >= lc["min_mean_wpli_coupled_edge"],
            "lag_top_edges": r["lag"]["coupled_are_top_edges"] is lc["coupled_edges_are_the_top_edges"],
            "lag_direction": min(r["lag"]["share_correct_direction"]) >= lc["min_share_windows_correct_lag_direction"],
            "null_offset": max(abs(v) for v in r["null"]["offset_ci95"]) <= nc["max_abs_offset_ci_bound"],
            "null_ks": r["null"]["ks_distance"] <= nc["max_ks_distance_real_vs_single_realisation"],
        }
        if "direction_share_range" in sc:  # EXP-005, EXP-006
            lo, hi = sc["direction_share_range"]
            out[b]["surr_direction_at_chance"] = all(lo <= v <= hi for v in r["surrogate"]["share_injected_direction"])
        if "max_excess_over_same_channel_reference" in sc:  # EXP-006
            out[b]["surr_vs_same_channel_reference"] = max(
                v for pair in r["surrogate"]["excess_over_same_channel_reference"] for v in pair) \
                <= sc["max_excess_over_same_channel_reference"]
            if "precision" in sc:  # EXP-007
                out[b]["excess_se_ok"] = max(v for pair in r["surrogate"]["excess_se"] for v in pair) \
                    <= sc["precision"]["max_se_per_excess"]
        elif "max_abs_diff_surrogate_vs_shared_spectrum_null" in sc:  # EXP-005
            out[b]["surr_vs_shared_spectrum_null"] = max(abs(v) for v in r["surrogate"]["diff_vs_shared_spectrum_null"]) \
                <= sc["max_abs_diff_surrogate_vs_shared_spectrum_null"]
        else:  # EXP-001
            lo, hi = r["surrogate"]["uncoupled_mean_range"]
            out[b]["surr_within_range"] = all(lo <= m <= hi for m in r["surrogate"]["mean_wpli_coupled"])
            out[b]["surr_diff"] = abs(r["surrogate"]["diff_mean_coupled_vs_uncoupled"]) \
                <= sc["max_abs_diff_mean_surrogate_coupled_vs_uncoupled"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out", default="artifacts/protocol/pc3")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    t = time.perf_counter()
    report = run(cfg, a.workers)
    checks = evaluate(report, cfg["criteria"])
    result = "PASS" if all(all(c.values()) for c in checks.values()) else "FAIL"
    if result == "FAIL" and "result_rule" in cfg["criteria"]:  # EXP-007: only the precision check failing
        if all(v for c in checks.values() for name, v in c.items() if name != "excess_se_ok"):
            result = "INDETERMINATE"
    out = {"experiment": cfg["experiment"], "result": result, "checks": checks, "report": report,
           "config": a.config, "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), "runtime_s": round(time.perf_counter() - t, 1),
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "machine": platform.platform()}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / REPORT_NAME[cfg["experiment"]]).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"result": result, "checks": checks}, indent=2))
    sys.exit(0 if result == "PASS" else 1)


if __name__ == "__main__":
    main()
