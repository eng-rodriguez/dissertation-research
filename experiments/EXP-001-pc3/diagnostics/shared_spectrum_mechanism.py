# Mechanism check (diagnostic, not a PC-3 criterion): do two signals with independent uniform random phases
# have higher wPLI when they share the same amplitude spectrum? And do IAAFT surrogates keep the lag direction?
import json, sys, numpy as np
sys.path.insert(0, "src/connectivity")
import pc3, wpli as wp, surrogates as sg
rng = np.random.default_rng(7); fs=500; n=1000; taper=np.hanning(800)
def burst():
    b = np.zeros(n); b[100:900] = taper * wp.bandpass(rng.normal(size=1800), "theta")[500:1300]; return b
def randphase(mag):
    ph = rng.uniform(0, 2*np.pi, mag.size); ph[0]=0; ph[-1]=0
    return np.fft.irfft(mag*np.exp(1j*ph), n=n)
same, diff = [], []
for _ in range(400):
    m1 = np.abs(np.fft.rfft(burst())); m2 = np.abs(np.fft.rfft(burst()))
    for mags, out in (((m1, m1), same), ((m1, m2), diff)):
        z = wp.analytic(np.vstack([randphase(m) for m in mags])); out.append(wp.wpli(z)[0][0])
print("independent random phases, SAME amplitude spectrum: mean wPLI %.3f" % np.mean(same))
print("independent random phases, DIFFERENT amplitude spectra: mean wPLI %.3f" % np.mean(diff))
# lag direction in IAAFT surrogates of the PC-3 lag recording (theta)
cfg = json.load(open("experiments/EXP-001-pc3/config.json")); d = cfg["synthetic_design"]; lt = d["lag_test"]
starts = pc3.window_starts(lt["n_windows"], n, fs)
x = pc3.background(np.random.default_rng([d["seeds"]["background_lag"], 0]), 19, lt["duration_s"]*fs, fs)
x = pc3.inject_bursts(x, "theta", starts, n, lt["coupled_pairs"], lt["lag_sign"], lt["snr"], fs, np.random.default_rng([d["seeds"]["bursts"], 0]))
xb = wp.bandpass(wp.car(x), "theta"); ei, ej = wp.edge_index(19)
edge_of = {(a, b): e for e, (a, b) in enumerate(zip(ei, ej))}
agree = {tuple(p): [] for p in lt["coupled_pairs"]}
for s in starts:
    for r in range(5):
        seeds = [sg.surrogate_seed("SYN-LAG-theta", s, "theta", r, c) for c in range(19)]
        sur, _ = sg.iaaft_rows(xb[:, s:s+n], seeds); si = wp.signed_imag(wp.analytic(sur))
        for p, sgn in zip(lt["coupled_pairs"], lt["lag_sign"]): agree[tuple(p)].append(np.sign(si[edge_of[tuple(p)]]) == sgn)
print("IAAFT surrogates, share of windows x realisations with the injected lag direction (real: 1.00):",
      {str(k): round(float(np.mean(v)), 2) for k, v in agree.items()})
