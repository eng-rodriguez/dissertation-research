"""Runtime of IAAFT surrogate generation for the issue-8 control (06), on synthetic noise only.

Issue 8 asks for 3-5 IAAFT realisations per analysis window, with the runtime cost documented
before the realisation count is fixed. This times surrogate generation for 19-channel windows of
2 s and 4 s at 500 Hz, using coloured Gaussian noise. No vEpiSet data, FC estimator, feature or
model is involved; the FC cost on surrogates is k times the FC cost on the real windows and is
reported as that multiple.

IAAFT (iterative amplitude-adjusted Fourier transform; Schreiber & Schmitz 1996 [VERIFY]) is
implemented here only for timing; the analysis implementation follows protocol lock.

Usage (from the repository root):
    python src/design/surrogate_runtime_bench.py
Writes artifacts/protocol/surrogate_runtime/runtime.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

FS = 500
N_CH = 19
N_EPOCHS = 25_449  # authors' 4-s epochs in vEpiSet (05, VERIFIED)


def iaaft(x: np.ndarray, rng: np.random.Generator, max_iter: int = 200, tol: float = 1e-8) -> tuple[np.ndarray, int]:
    """Per-channel IAAFT of x (channels x samples). Returns (surrogate, iterations used)."""
    amp = np.abs(np.fft.rfft(x, axis=-1))
    sorted_x = np.sort(x, axis=-1)
    s = rng.permuted(x, axis=-1)
    prev = np.inf
    for it in range(1, max_iter + 1):
        phase = np.angle(np.fft.rfft(s, axis=-1))
        s = np.fft.irfft(amp * np.exp(1j * phase), n=x.shape[-1], axis=-1)
        ranks = np.argsort(np.argsort(s, axis=-1), axis=-1)
        s = np.take_along_axis(sorted_x, ranks, axis=-1)
        err = np.mean((np.abs(np.fft.rfft(s, axis=-1)) - amp) ** 2) / np.mean(amp ** 2)
        if abs(prev - err) < tol:
            break
        prev = err
    return s, it


def bench(seconds: float, n_windows: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = np.cumsum(rng.normal(size=(n_windows, N_CH, n)), axis=-1)  # 1/f^2-like background
    x -= x.mean(axis=-1, keepdims=True)
    iaaft(x[0], rng)  # warm-up
    t = time.perf_counter()
    iters = [iaaft(w, rng)[1] for w in x]
    dt = (time.perf_counter() - t) / n_windows
    return {"window_s": seconds, "n_windows_timed": n_windows, "sec_per_window_per_realisation": round(dt, 5),
            "median_iterations": int(np.median(iters))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-windows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", default="artifacts/protocol/surrogate_runtime")
    a = ap.parse_args()
    rows = [bench(s, a.n_windows, a.seed) for s in (2.0, 4.0)]
    proj = {}
    for r in rows:
        for k in (3, 4, 5):
            # one surrogate set per window; generated once and cached, independent of CV repeats
            proj[f"{r['window_s']:.0f}s_k{k}_cpu_hours_all_epochs"] = round(
                r["sec_per_window_per_realisation"] * k * N_EPOCHS / 3600, 2)
    out = {"benchmark": rows, "projection_single_core": proj,
           "fc_cost_multiplier_vs_real": "k (FC is computed once per realisation per window; CV repeats reuse cached features)",
           "n_epochs": N_EPOCHS, "seed": a.seed, "machine": platform.platform(), "processor": platform.processor(),
           "python": platform.python_version(), "numpy": np.__version__}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "runtime.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
