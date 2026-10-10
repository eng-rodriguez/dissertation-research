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
Engineering check on synthetic data only; no vEpiSet data is read.

Usage (from the repository root):
    python src/connectivity/pc3.py
Writes artifacts/protocol/pc3/pc3_report.json.
"""

from __future__ import annotations

import argparse
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


def lag_and_surrogate(band: str, design: dict) -> dict:
    fs, n_ch, win_n = design["fs"], design["n_channels"], int(design["window_s"] * design["fs"])
    lt, k = design["lag_test"], design["surrogate_realisations"]
    b_idx = design["bands"].index(band)
    seeds = design["seeds"]
    starts = window_starts(lt["n_windows"], win_n, fs)
    x = background(np.random.default_rng([seeds["background_lag"], b_idx]), n_ch, int(lt["duration_s"] * fs), fs)
    x = inject_bursts(x, band, starts, win_n, lt["coupled_pairs"], lt["lag_sign"], lt["snr"], fs,
                      np.random.default_rng([seeds["bursts"], b_idx]))
    ei, ej = wp.edge_index(n_ch)
    edge_of = {(a, b): e for e, (a, b) in enumerate(zip(ei, ej))}
    coupled = [edge_of[tuple(p)] for p in lt["coupled_pairs"]]
    uncoupled = np.setdiff1d(np.arange(len(ei)), coupled)

    z = wp.analytic(wp.bandpass(wp.car(x), band, fs))
    real = np.vstack([wp.wpli(z[:, s:s + win_n])[0] for s in starts])
    signed = np.vstack([wp.signed_imag(z[:, s:s + win_n]) for s in starts])
    m = real.mean(axis=0)
    lag = {"mean_wpli_coupled": m[coupled].tolist(),
           "max_mean_wpli_uncoupled": float(m[uncoupled].max()),
           "coupled_are_top_edges": set(np.argsort(m)[-len(coupled):].tolist()) == set(coupled),
           "share_correct_direction": [float(np.mean(np.sign(signed[:, e]) == sg_)) for e, sg_ in
                                       zip(coupled, lt["lag_sign"])],
           "delay_samples": int(round(fs / (4 * BAND_CENTRE[band])))}

    surr = sg.surrogate_fc(x, f"SYN-LAG-{band}", starts, win_n, k, bands=(band,), fs=fs)[band]["wpli"]
    ms = surr.mean(axis=0)
    surrogate = {"mean_wpli_coupled": ms[coupled].tolist(),
                 "uncoupled_mean_range": [float(ms[uncoupled].min()), float(ms[uncoupled].max())],
                 "diff_mean_coupled_vs_uncoupled": float(ms[coupled].mean() - ms[uncoupled].mean())}
    return {"lag": lag, "surrogate": surrogate}


def null_match(band: str, design: dict) -> dict:
    fs, n_ch, win_n = design["fs"], design["n_channels"], int(design["window_s"] * design["fs"])
    nt, k = design["null_test"], design["surrogate_realisations"]
    starts = window_starts(nt["n_windows"], win_n, fs)
    x = background(np.random.default_rng(design["seeds"]["background_null"]), n_ch, int(nt["duration_s"] * fs), fs)
    real = wp.real_fc(x, starts, win_n, bands=(band,), fs=fs)[band]["wpli"]
    s = sg.surrogate_fc(x, f"SYN-NULL-{band}", starts, win_n, k, bands=(band,), fs=fs, keep_realisations=True)[band]
    d = real.mean(axis=1) - s["wpli"].mean(axis=1)
    half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
    return {"mean_real": float(real.mean()), "mean_surrogate_kavg": float(s["wpli"].mean()),
            "offset_mean": float(d.mean()), "offset_ci95": [float(d.mean() - half), float(d.mean() + half)],
            "ks_distance": float(stats.ks_2samp(real.ravel(), s["realisations"][:, 0, :].ravel()).statistic),
            "sd_real": float(real.std()), "sd_surrogate_single": float(s["realisations"][:, 0, :].std()),
            "sd_surrogate_kavg": float(s["wpli"].std())}


def _band(args):
    band, design = args
    return band, {**lag_and_surrogate(band, design), "null": null_match(band, design)}


def run(cfg: dict, workers: int | None = None) -> dict:
    design = cfg["synthetic_design"]
    jobs = [(b, design) for b in design["bands"]]
    if workers == 1:
        return dict(map(_band, jobs))
    with ProcessPoolExecutor(max_workers=workers or min(len(jobs), 3)) as ex:
        return dict(ex.map(_band, jobs))


def evaluate(report: dict, crit: dict) -> dict:
    """Per band and criterion: True/False. Mirrors tests/test_pc3_fc_pipeline.py, which is the gate."""
    out = {}
    for b, r in report.items():
        lc, sc, nc = crit["lag_recovery"], crit["surrogate_destruction"], crit["null_coupling_match"]
        lo, hi = r["surrogate"]["uncoupled_mean_range"]
        out[b] = {
            "lag_min_wpli": min(r["lag"]["mean_wpli_coupled"]) >= lc["min_mean_wpli_coupled_edge"],
            "lag_top_edges": r["lag"]["coupled_are_top_edges"] is lc["coupled_edges_are_the_top_edges"],
            "lag_direction": min(r["lag"]["share_correct_direction"]) >= lc["min_share_windows_correct_lag_direction"],
            "surr_within_range": all(lo <= m <= hi for m in r["surrogate"]["mean_wpli_coupled"]),
            "surr_diff": abs(r["surrogate"]["diff_mean_coupled_vs_uncoupled"])
            <= sc["max_abs_diff_mean_surrogate_coupled_vs_uncoupled"],
            "null_offset": max(abs(v) for v in r["null"]["offset_ci95"]) <= nc["max_abs_offset_ci_bound"],
            "null_ks": r["null"]["ks_distance"] <= nc["max_ks_distance_real_vs_single_realisation"],
        }
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
    out = {"experiment": cfg["experiment"], "result": result, "checks": checks, "report": report,
           "config": a.config, "runtime_s": round(time.perf_counter() - t, 1),
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "machine": platform.platform()}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "pc3_report.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"result": result, "checks": checks}, indent=2))
    sys.exit(0 if result == "PASS" else 1)


if __name__ == "__main__":
    main()
