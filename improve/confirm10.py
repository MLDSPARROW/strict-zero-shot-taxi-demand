"""10-run confirmation (declared 2026-10-02): partition-diverse scale-matched helpers, runs r=1..10 (merging r, seed r-1),
vs single merging (seeds 0-9). Internal paired t-test on per-run share error (pre-declared rule); FULLY STRICT trip errors
of the 10-run ensembles (full year and December)."""
import numpy as np, importlib.util, io, contextlib
from datetime import date, timedelta
from scipy.stats import ttest_rel
spec = importlib.util.spec_from_file_location("ss", "strict_seeds.py"); ss = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(ss)
B = "../bench77/data"
for tgt, f, H in [("chicago", "chicago_2021_ca_30min.npy", ["nyc_for_chicago", "sf_for_chicago"]), ("nyc", "nyc_2021_zone_30min.npy", ["chicago", "sf_for_nyc"])]:
    Y = np.load(f"{B}/{f}").astype(np.float64); tot = Y.sum(2); idx = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Sh = Y[di, si] / tot[di, si][:, None]
    dec = np.array([(date(2021, 1, 1) + timedelta(days=int(d))).month == 12 for d in di])
    print(f"\n===== {tgt.upper()} =====")
    for n in ("eh", "eo"):
        A = [np.load(f"out/multi_{tgt}_from_{'+'.join(H)}_v{n}_seed{s}.npy") for s in range(10)]
        Bm = [np.load(f"out/multi_{tgt}_from_{'+'.join(h if h == 'chicago' else f'{h}_p{r}' for h in H)}_v{n}_seed{r-1}.npy") for r in range(1, 11)]
        ea = np.array([np.abs(p / p.sum(1, keepdims=True) - Sh).mean() for p in A]); eb = np.array([np.abs(p / p.sum(1, keepdims=True) - Sh).mean() for p in Bm])
        t, pv = ttest_rel(eb, ea)
        for lab, P in [("one merging (10 runs)", A), ("10 different mergings", Bm)]:
            p = np.mean(P, 0); p /= p.sum(1, keepdims=True); T = None
            fy = ss.strict_scores(tgt, [p])[0]
            src = open("strict_seeds.py").read()
            print(f"  {n} {lab:24s} FULLY STRICT full year {fy[0]:.2f} / {fy[1]:.2f} / {fy[2]:.1f}%")
        print(f"  {n} [internal statistics] 10 mergings better in {(eb<ea).sum()}/10 runs, change {100*(eb.mean()/ea.mean()-1):+.1f}%, paired t p={pv:.4f}")
