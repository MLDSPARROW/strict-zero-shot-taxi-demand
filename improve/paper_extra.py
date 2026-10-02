"""Paper v2 extras (2026-10-02): (1) helper-only leave-one-city-out table for the LEVEL unit; (2) day-level paired tests of the
FINAL method vs the strongest simple baseline (share by jobs) and vs the base joint allocator, FULLY STRICT trip errors."""
import numpy as np, io, contextlib
from scipy.stats import ttest_rel
src = open("idea_avg_rhythm.py", encoding="utf-8").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(src[:src.rindex("for tgt, (srcs, S) in st.CFG.items():")], ns)
st, strict_T, U, trips, TAXI = ns["st"], ns["strict_T"], ns["U"], ns["trips"], ns["TAXI"]
UN = ["jobs", "residents", "population", "taxi commuters", "transit commuters", "car-free households"]
for tgt in ["chicago", "nyc"]:
    H = [c for c in TAXI if c != tgt]
    print(f"=== LEVEL unit selection for target {tgt} (helpers {H}): mean |error| of the held-out helper's yearly total")
    for u in UN:
        e = [abs(np.exp(np.mean([np.log(trips[c] / U[c][u]) for c in H if c != h])) * U[h][u] / trips[h] - 1) for h in H]
        print(f"   {u:20s} {100*np.mean(e):5.1f}%   (" + ", ".join(f"{h} {100*x:.0f}%" for h, x in zip(H, e)) + ")")
rng = np.random.default_rng(0)
for tgt in ["chicago", "nyc"]:
    srcs, S = st.CFG[tgt]; Y = st.Y[tgt]; tot = Y.sum(2); idx = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Y[di, si]; T = strict_T(tgt, S, di, si); n = len(di)
    E_ = np.load(f"data/extra_{tgt}.npz"); pf = np.load(f"out/FINAL_{tgt}_shares.npy")
    pja = np.mean([np.load(f"../multicity/multi_{tgt}_from_{srcs}_seed{s}.npy") for s in range(10)], 0); pja /= pja.sum(1, keepdims=True)
    pj = np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0)
    def daily(p):
        E = p * T[:, None] - Yc; c = np.bincount(di); return np.bincount(di, np.abs(E).mean(1)) / c, np.bincount(di, (E ** 2).mean(1)) / c
    fa, fs = daily(pf)
    for name, p in [("share by jobs", pj), ("base joint allocator", pja)]:
        a, s_ = daily(p); out = []
        for x, y, nm in [(fa, a, "MAE"), (fs, s_, "MSE")]:
            d = x - y; nb = len(d) // 7; bs = [d[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(d) - 7, nb)])].mean() for _ in range(2000)]
            out.append(f"{nm}: final better on {100*(x<y).mean():.0f}% of days, paired t p={ttest_rel(x, y)[1]:.2g}, 7-day block bootstrap 95% CI of daily difference [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
        print(f"{tgt:8s} FINAL vs {name:22s} " + " | ".join(out))
