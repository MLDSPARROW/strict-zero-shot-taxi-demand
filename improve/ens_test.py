"""Ensemble-level test (2026-10-02): 10-merging ensemble vs single-merging 10-run ensemble, FULLY STRICT trip errors.
Daily MAE and daily squared error compared day by day (365 days); paired t-test and a block bootstrap (7-day blocks, 2000 draws)."""
import numpy as np, importlib.util, io, contextlib
from scipy.stats import ttest_rel
src = open("idea_avg_rhythm.py").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(src[:src.rindex("for tgt, (srcs, S) in st.CFG.items():")], ns)
st, strict_T = ns["st"], ns["strict_T"]; rng = np.random.default_rng(0)
for tgt, H in [("chicago", ["nyc_for_chicago", "sf_for_chicago"]), ("nyc", ["chicago", "sf_for_nyc"])]:
    srcs, S = st.CFG[tgt]; Y = st.Y[tgt]; tot = Y.sum(2); idx = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Y[di, si]; T = strict_T(tgt, S, di, si)
    for n in ("eh", "eo"):
        A = np.mean([np.load(f"out/multi_{tgt}_from_{'+'.join(H)}_v{n}_seed{s}.npy") for s in range(10)], 0)
        Bm = np.mean([np.load(f"out/multi_{tgt}_from_{'+'.join(h if h == 'chicago' else f'{h}_p{r}' for h in H)}_v{n}_seed{r-1}.npy") for r in range(1, 11)], 0)
        ea = np.abs(A / A.sum(1, keepdims=True) * T[:, None] - Yc); eb = np.abs(Bm / Bm.sum(1, keepdims=True) * T[:, None] - Yc)
        da, db = np.bincount(di, ea.mean(1)) / np.bincount(di), np.bincount(di, eb.mean(1)) / np.bincount(di)
        sa, sb = np.bincount(di, (ea ** 2).mean(1)) / np.bincount(di), np.bincount(di, (eb ** 2).mean(1)) / np.bincount(di)
        out = []
        for x, y, nm in [(db, da, "MAE"), (sb, sa, "MSE")]:
            p = ttest_rel(x, y)[1]; d = x - y; nb = len(d) // 7
            bs = [d[np.concatenate([np.arange(s, s + 7) for s in rng.integers(0, len(d) - 7, nb)])].mean() for _ in range(2000)]
            out.append(f"{nm}: 10-merging better on {100*(x<y).mean():.0f}% of days, paired t p={p:.2g}, block-bootstrap 95% CI of difference [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
        print(f"{tgt:8s} {n}: " + " | ".join(out))
