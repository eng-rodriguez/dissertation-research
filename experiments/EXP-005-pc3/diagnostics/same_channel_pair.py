# Mechanism diagnostic after the EXP-005 FAIL (not a PC-3 criterion). Two independent IAAFT surrogates of the SAME
# channel share every single-channel property and have no coupling by construction. If their wPLI equals the wPLI
# between the surrogates of a coupled pair (i, j), the coupled-edge excess comes from the pair's matched
# single-channel properties (spectrum and amplitude distribution), not from retained coupling.
# Usage from the repository root: python experiments/EXP-005-pc3/diagnostics/same_channel_pair.py theta
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
pairs = [tuple(p) for p in lt["coupled_pairs"]]
ij, ii, jj, rp_ij, rp_ii = ([] for _ in range(5))
for s in starts:
    win = xb[:, s:s + n]
    for r in range(k):
        for a, b in pairs:
            sa, _ = sg.iaaft_rows(win[[a]], [sg.surrogate_seed(eid, s, band, r, a)])
            sb, _ = sg.iaaft_rows(win[[b]], [sg.surrogate_seed(eid, s, band, r, b)])
            sa2, _ = sg.iaaft_rows(win[[a]], [sg.surrogate_seed(eid + "-dup", s, band, r, a)])
            sb2, _ = sg.iaaft_rows(win[[b]], [sg.surrogate_seed(eid + "-dup", s, band, r, b)])
            w = lambda u, v: wp.wpli(wp.analytic(np.vstack([u, v])))[0][0]
            ij.append(w(sa[0], sb[0])); ii.append(w(sa[0], sa2[0])); jj.append(w(sb[0], sb2[0]))
            seeds = [sg.surrogate_seed(eid + "-ft", s, band, r, c) for c in (a, b, a + 100)]
            rp = pc3.shared_spectrum_null(win[[a, b, a]], seeds)
            rp_ij.append(w(rp[0], rp[1])); rp_ii.append(w(rp[0], rp[2]))
print(band, "coupled pairs, mean wPLI: IAAFT(i) vs IAAFT(j) %.3f | IAAFT(i) vs IAAFT'(i) %.3f | IAAFT(j) vs IAAFT'(j) %.3f"
      % (np.mean(ij), np.mean(ii), np.mean(jj)))
print(band, "                          random-phase(i) vs (j) %.3f | random-phase(i) vs (i)' %.3f" % (np.mean(rp_ij), np.mean(rp_ii)))
