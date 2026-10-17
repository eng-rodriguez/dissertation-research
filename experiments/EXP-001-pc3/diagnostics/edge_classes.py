# Diagnostic after the EXP-001 FAIL (not a PC-3 criterion). Surrogate vs real wPLI by edge class.
# Usage from the repository root: python experiments/EXP-001-pc3/diagnostics/edge_classes.py theta
import json, sys, numpy as np
sys.path.insert(0, "src/connectivity")
import pc3, wpli as wp, surrogates as sg
cfg = json.load(open("experiments/EXP-001-pc3/config.json")); d = cfg["synthetic_design"]
band = sys.argv[1]; b_idx = d["bands"].index(band); lt = d["lag_test"]; fs=500; win=1000
starts = pc3.window_starts(lt["n_windows"], win, fs)
x = pc3.background(np.random.default_rng([d["seeds"]["background_lag"], b_idx]), 19, lt["duration_s"]*fs, fs)
x = pc3.inject_bursts(x, band, starts, win, lt["coupled_pairs"], lt["lag_sign"], lt["snr"], fs, np.random.default_rng([d["seeds"]["bursts"], b_idx]))
ei, ej = wp.edge_index(19); burst_ch = {c for p in lt["coupled_pairs"] for c in p}
coupled = {tuple(p) for p in lt["coupled_pairs"]}
kind = np.array(["coupled" if (a,b) in coupled else "burst-burst" if a in burst_ch and b in burst_ch else "burst-bg" if (a in burst_ch or b in burst_ch) else "bg-bg" for a,b in zip(ei,ej)])
z = wp.analytic(wp.bandpass(wp.car(x), band, fs))
real = np.vstack([wp.wpli(z[:, s:s+win])[0] for s in starts]).mean(0)
surr = sg.surrogate_fc(x, f"SYN-LAG-{band}", starts, win, 5, bands=(band,))[band]["wpli"].mean(0)
for k in ["coupled","burst-burst","burst-bg","bg-bg"]:
    m = kind==k; print(band, k, m.sum(), "real %.3f" % real[m].mean(), "surr mean %.3f range [%.3f, %.3f]" % (surr[m].mean(), surr[m].min(), surr[m].max()))
