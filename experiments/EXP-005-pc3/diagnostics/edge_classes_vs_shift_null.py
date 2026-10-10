# Diagnostic after the EXP-005 FAIL (not a PC-3 criterion). Is the surrogate excess specific to coupled edges?
# Per edge class, mean wPLI of IAAFT surrogates and of independent random circular shifts of each channel
# (shift keeps each channel's spectrum and amplitude distribution exactly and has no phase relation by lag).
# Usage from the repository root: python experiments/EXP-005-pc3/diagnostics/edge_classes_vs_shift_null.py theta
import json, sys, numpy as np
sys.path.insert(0, "src/connectivity")
import pc3, wpli as wp, surrogates as sg
band = sys.argv[1]
cfg = json.load(open("experiments/EXP-005-pc3/config.json")); d = cfg["synthetic_design"]; lt = d["lag_test"]
b_idx = d["bands"].index(band); fs, n, k = 500, 1000, 5
starts = pc3.window_starts(lt["n_windows"], n, fs)
x = pc3.background(np.random.default_rng([d["seeds"]["background_lag"], b_idx]), 19, lt["duration_s"] * fs, fs)
x = pc3.inject_bursts(x, band, starts, n, lt["coupled_pairs"], lt["lag_sign"], lt["snr"], fs,
                      np.random.default_rng([d["seeds"]["bursts"], b_idx]))
xb = wp.bandpass(wp.car(x), band, fs); eid = f"SYN5-LAG-{band}"
ei, ej = wp.edge_index(19); bc = {c for p in lt["coupled_pairs"] for c in p}; cp = {tuple(p) for p in lt["coupled_pairs"]}
kind = np.array(["coupled" if (a, b) in cp else "burst-burst" if a in bc and b in bc else "burst-bg" if (a in bc or b in bc)
                 else "bg-bg" for a, b in zip(ei, ej)])
rng = np.random.default_rng(99); W_i, W_s = np.zeros(len(ei)), np.zeros(len(ei))
for s in starts:
    win = xb[:, s:s + n]
    for r in range(k):
        sv, _ = sg.iaaft_rows(win, [sg.surrogate_seed(eid, s, band, r, c) for c in range(19)])
        W_i += wp.wpli(wp.analytic(sv))[0]
        W_s += wp.wpli(wp.analytic(np.vstack([np.roll(win[c], rng.integers(n)) for c in range(19)])))[0]
W_i /= len(starts) * k; W_s /= len(starts) * k
for c in ("coupled", "burst-burst", "burst-bg", "bg-bg"):
    m = kind == c
    print(band, f"{c:11s} n={m.sum():3d}  IAAFT {W_i[m].mean():.3f}  shift {W_s[m].mean():.3f}  excess {W_i[m].mean() - W_s[m].mean():+.3f}")
