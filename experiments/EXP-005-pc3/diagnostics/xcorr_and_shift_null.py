# Diagnostic after the EXP-005 FAIL (not a PC-3 criterion). Are IAAFT surrogates of bursty band-passed windows
# close to circularly shifted copies of the original? Compares (a) max circular cross-correlation between each
# channel's window and its surrogate, burst vs background channels; (b) coupled-edge wPLI of IAAFT surrogates vs
# independent random circular shifts of each channel (which keep spectrum and amplitude distribution exactly).
# Usage from the repository root: python experiments/EXP-005-pc3/diagnostics/xcorr_and_shift_null.py theta
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
burst_ch = sorted({c for p in lt["coupled_pairs"] for c in p}); bg_ch = [c for c in range(19) if c not in burst_ch]
def maxcorr(a, b):
    c = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=a.size)
    return np.max(np.abs(c)) / (np.linalg.norm(a) * np.linalg.norm(b))
cc_b, cc_g, iaaft_w, shift_w = [], [], [], []
rng = np.random.default_rng(99)
for s in starts:
    win = xb[:, s:s + n]
    for r in range(k):
        sv, _ = sg.iaaft_rows(win, [sg.surrogate_seed(eid, s, band, r, c) for c in range(19)])
        cc_b += [maxcorr(win[c], sv[c]) for c in burst_ch]; cc_g += [maxcorr(win[c], sv[c]) for c in bg_ch]
        sh = np.vstack([np.roll(win[c], rng.integers(n)) for c in range(19)])
        wi, ws = wp.wpli(wp.analytic(sv))[0], wp.wpli(wp.analytic(sh))[0]
        ei, ej = wp.edge_index(19); eo = {(a, b): e for e, (a, b) in enumerate(zip(ei, ej))}
        iaaft_w.append([wi[eo[tuple(p)]] for p in lt["coupled_pairs"]]); shift_w.append([ws[eo[tuple(p)]] for p in lt["coupled_pairs"]])
print(band, "max circular xcorr original vs IAAFT surrogate: burst channels median %.2f, background channels median %.2f"
      % (np.median(cc_b), np.median(cc_g)))
print(band, "coupled-edge mean wPLI: IAAFT surrogates %s | independent random circular shifts %s"
      % (np.round(np.mean(iaaft_w, 0), 3), np.round(np.mean(shift_w, 0), 3)))
