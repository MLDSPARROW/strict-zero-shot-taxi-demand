"""Revision analyses (2026-10-02), Chicago and NYC (development targets), FULLY STRICT where trip errors are reported.
(1) 95% CIs of the final MAE/RMSE (7-day block bootstrap over days, 2000 draws).
(2) Allocation-only metrics of the final shares: mean absolute share error, mean Jensen-Shannon divergence per slot,
    Spearman correlation of annual region totals, top-10 recall of the annual busiest regions.
(3) Number of partitions in the ensemble: 1, 2, 5, 10 (first k partitions; both heads; strict trip errors).
(4) Rhythm sensitivity: month factor from same-year helpers (final) / none / pooled over all share helpers (any year)."""
import numpy as np, io, contextlib
from datetime import date, timedelta
from scipy.stats import spearmanr
src = open("idea_avg_rhythm.py", encoding="utf-8").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(src[:src.rindex("for tgt, (srcs, S) in st.CFG.items():")], ns)
st, strict_T, v2 = ns["st"], ns["strict_T"], ns["v2"]
SCALE = {"chicago": ["nyc_for_chicago", "sf_for_chicago"], "nyc": ["chicago", "sf_for_nyc"]}
rng = np.random.default_rng(0)
def files(tgt, n, rs): return [f"out/multi_{tgt}_from_{'+'.join(h if h == 'chicago' else f'{h}_p{r}' for h in SCALE[tgt])}_v{n}_seed{r-1}.npy" for r in rs]
def ens(fs): p = np.mean([np.load(f) for f in fs], 0); return p / p.sum(1, keepdims=True)
for tgt in ["chicago", "nyc"]:
    srcs, S = st.CFG[tgt]; Y = st.Y[tgt]; tot = Y.sum(2); idx = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Y[di, si]; Sh = Yc / tot[di, si][:, None]; T = strict_T(tgt, S, di, si)
    pf = np.load(f"out/FINAL_{tgt}_shares.npy"); E = pf * T[:, None] - Yc
    dA = np.bincount(di, np.abs(E).mean(1)) / np.bincount(di); dS = np.bincount(di, (E ** 2).mean(1)) / np.bincount(di); nb = len(dA) // 7
    bm, br = [], []
    for _ in range(2000):
        ii = np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(dA) - 7, nb)]); bm.append(dA[ii].mean()); br.append(np.sqrt(dS[ii].mean()))
    print(f"\n===== {tgt.upper()} =====\n(1) FINAL MAE {np.abs(E).mean():.2f} [95% CI {np.percentile(bm,2.5):.2f}, {np.percentile(bm,97.5):.2f}] | RMSE {np.sqrt((E**2).mean()):.2f} [{np.percentile(br,2.5):.2f}, {np.percentile(br,97.5):.2f}] (n = {len(dA)} days)")
    def alloc(p):
        q = np.clip(p, 1e-12, 1); s = np.clip(Sh, 1e-12, 1); mM = 0.5 * (q + s)
        js = 0.5 * (np.sum(q * np.log(q / mM), 1) + np.sum(np.where(Sh > 0, s * np.log(s / mM), 0), 1))
        pt, tt = (p * tot[di, si][:, None]).sum(0), Yc.sum(0); top = set(np.argsort(-tt)[:10])
        return np.abs(p - Sh).mean(), js.mean(), spearmanr(pt, tt)[0], len(top & set(np.argsort(-pt)[:10])) / 10
    pja = ens([f"../multicity/multi_{tgt}_from_{srcs}_seed{s}.npy" for s in range(10)])
    E_ = np.load(f"data/extra_{tgt}.npz"); pj = np.repeat((E_["jobs"] / E_["jobs"].sum())[None], len(di), 0)
    print("(2) allocation-only: mean abs share error | mean JS divergence | Spearman of annual totals | top-10 recall")
    for nm, p in [("jobs-proportional", pj), ("base joint allocator", pja), ("final", pf)]:
        a = alloc(p); print(f"    {nm:22s} {a[0]:.5f} | {a[1]:.4f} | {a[2]:.3f} | {a[3]:.1f}")
    print("(3) partitions k: MAE / RMSE (strict, final = average of heads)")
    for k in [1, 2, 5, 10]:
        p = 0.5 * ens(files(tgt, "eh", range(1, k + 1))) + 0.5 * ens(files(tgt, "eo", range(1, k + 1))); e = p * T[:, None] - Yc
        print(f"    k={k:2d}: {np.abs(e).mean():.2f} / {np.sqrt((e**2).mean()):.2f}")
    print("(4) rhythm month factor: per-slot city-total error | final MAE / RMSE")
    rh = np.mean([st.rhythm(c) for c in S], 0); w = st.dow[tgt][di]; nd = np.bincount(st.dow[tgt], minlength=7)
    base = rh[w, si] / nd[w]
    def mfac(cities):
        mf = []
        for c in cities:
            tc = st.Y[c].sum((1, 2)); mo = v2.month_of(c); m = np.array([tc[mo == k].mean() for k in range(1, 13)]); mf.append(m / m.mean())
        mf = np.mean(mf, 0); mt = v2.month_of(tgt); return mf[mt[di] - 1] / mf[mt - 1].mean()
    for nm, TT in [("same-year helper (final)", T), ("none", base), ("pooled over all share helpers", base * mfac(S))]:
        TT = TT / TT.sum() * T.sum(); e = pf * TT[:, None] - Yc
        print(f"    {nm:30s} {100*np.abs(TT-tot[di,si]).mean()/tot[di,si].mean():.1f}% | {np.abs(e).mean():.2f} / {np.sqrt((e**2).mean()):.2f}")
