# Mechanism diagnostic after the EXP-005 FAIL (not a PC-3 criterion). Does a per-channel IAAFT surrogate keep the
# timing of the original window's amplitude envelope? Zero-lag Pearson correlation between |hilbert| of the original
# band-passed window and of its surrogate: IAAFT vs independent random circular shift vs random-phase (FT) surrogate.
# Usage from the repository root: python experiments/EXP-005-pc3/diagnostics/envelope_timing.py theta
import json, sys, numpy as np
sys.path.insert(0, "src/connectivity")
import pc3, wpli as wp, surrogates as sg
band = sys.argv[1]
cfg = json.load(open("experiments/EXP-005-pc3/config.json")); d = cfg["synthetic_design"]; lt = d["lag_test"]
b_idx = d["bands"].index(band); fs, n = 500, 1000
starts = pc3.window_starts(lt["n_windows"], n, fs)
x = pc3.background(np.random.default_rng([d["seeds"]["background_lag"], b_idx]), 19, lt["duration_s"] * fs, fs)
x = pc3.inject_bursts(x, band, starts, n, lt["coupled_pairs"], lt["lag_sign"], lt["snr"], fs,
                      np.random.default_rng([d["seeds"]["bursts"], b_idx]))
xb = wp.bandpass(wp.car(x), band, fs); eid = f"SYN5-LAG-{band}"
bc = sorted({c for p in lt["coupled_pairs"] for c in p}); bg = [c for c in range(19) if c not in bc]
rng = np.random.default_rng(5)
env = lambda v: np.abs(wp.analytic(v))
r = {k: {"burst": [], "background": []} for k in ("iaaft", "shift", "random_phase")}
for s in starts:
    win = xb[:, s:s + n]; e0 = env(win)
    sv, _ = sg.iaaft_rows(win, [sg.surrogate_seed(eid, s, band, 0, c) for c in range(19)])
    sh = np.vstack([np.roll(win[c], rng.integers(n)) for c in range(19)])
    rp = pc3.shared_spectrum_null(win, rng.integers(2**63, size=19))
    for k, sur in (("iaaft", sv), ("shift", sh), ("random_phase", rp)):
        es = env(sur)
        for grp, chans in (("burst", bc), ("background", bg)):
            r[k][grp] += [np.corrcoef(e0[c], es[c])[0, 1] for c in chans]
for k, v in r.items():
    print(band, f"{k:12s} envelope corr with original: burst channels {np.mean(v['burst']):+.2f}, background {np.mean(v['background']):+.2f}")
