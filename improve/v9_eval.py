"""PROTOCOL_V9.md evaluation on the development targets Chicago and NYC (demand-reading runs in out/; the target demand is used here for scoring
only). Final method = equal average of the two heads: v1 = eh and eo; variant k = ehk and eok (bike-share cities as extra teachers). Ensemble of runs
r = 1..n (helper partition r, model seed r - 1), scored on the observed slots: MAE_share (the pre-declared metric), allocation-only trip MAE and RMSE
(true slot totals), peak ratio, top-10 recall of the yearly shares, the MAE_share of the single runs and of each head.
Screen (n = 3): k passes if its MAE_share is >= 3% better than v1 in one development target and not > 3% worse in the other.
Confirmation (n = 10, --confirm): k is adopted if (a) its mean MAE_share over the two targets is lower than v1's, (b) the 7-day block bootstrap (2,000
draws, as coarse_eval.py) interval of the daily MAE_share difference v1 - k lies above zero in at least one target, and (c) in the other target it is not
> 3% worse and its interval is not entirely below zero (the rules of PROTOCOL_V4.md).
Usage (from improve/): python v9_eval.py [--confirm]"""
import os, sys, numpy as np
from v9_common import srcs
HERE = os.path.dirname(os.path.abspath(__file__)); B = f"{HERE}/../bench77/data"
src = open(f"{HERE}/coarse_eval.py").read(); ns = {}
exec(src[src.index("def boot"):src.index("norm = lambda")], {"np": np}, ns); boot = ns["boot"]
CONF = "--confirm" in sys.argv; N = 10 if CONF else 3
V1 = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
YF = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy"}
VAR = {"v1": ("", lambda T, r: V1[T](r)), "k": ("k", lambda T, r: "+".join(srcs(T, r)))}
norm = lambda p: p / p.sum(1, keepdims=True)
res = {}
for T in ("chicago", "nyc"):
    Y = np.load(YF[T]).astype(np.float64); tot = Y.sum(2); nd = Y.shape[0]
    idx = [(d, s) for d in range(nd) for s in range(48) if tot[d, s] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx])
    Sh = Y[di, si] / tot[di, si][:, None]; Tt = tot[di, si]; true_year = Y.sum((0, 1)) / Y.sum()
    print(f"\n===== {T}: {Y.shape[2]} regions, {len(idx)} observed slots, ensembles of runs 1..{N} =====")
    for name, (v, sf) in VAR.items():
        fs = {h: [f"{HERE}/out/multi_{T}_from_{sf(T, r)}_v{h}{v}_seed{r - 1}.npy" for r in range(1, N + 1)] for h in ("eh", "eo")}
        if not all(os.path.exists(f) for h in fs for f in fs[h]): print(f"   {name:3s} missing runs"); continue
        runs = {h: [norm(np.load(f)) for f in fs[h]] for h in fs}
        H = {h: np.mean(runs[h], 0) for h in runs}; P = H["eh"] / 2 + H["eo"] / 2
        a = np.abs(P - Sh); daily = np.bincount(di, weights=a.mean(1), minlength=nd) / np.maximum(np.bincount(di, minlength=nd), 1)
        e = P * Tt[:, None] - Y[di, si]; single = [np.abs((x + y) / 2 - Sh).mean() for x, y in zip(runs["eh"], runs["eo"])]
        yr = (P * Tt[:, None]).sum(0); yr /= yr.sum(); top = len(set(np.argsort(-yr)[:10]) & set(np.argsort(-true_year)[:10])) / 10
        res[(T, name)] = (a.mean(), daily)
        print(f"   {name:3s} MAE_share {a.mean():.5f} (single runs {np.mean(single):.5f} +- {np.std(single, ddof=1):.5f}; heads eh {np.abs(H['eh'] - Sh).mean():.5f}, "
              f"eo {np.abs(H['eo'] - Sh).mean():.5f}) | allocation-only MAE {np.abs(e).mean():.3f} RMSE {np.sqrt((e ** 2).mean()):.3f} | "
              f"peak ratio {P.max(1).mean() / Sh.max(1).mean():.2f} | top-10 recall {top:.1f}")
rng = np.random.default_rng(0); TS = ("chicago", "nyc")
print(f"\n===== {'confirmation rule (a)-(c)' if CONF else 'screen rule: >= 3% better in one development target, not > 3% worse in the other'} =====")
name = "k"
if all((T, x) in res for T in TS for x in ("v1", name)):
    g, ci = {}, {}
    for T in TS:
        g[T] = 100 * (res[(T, "v1")][0] - res[(T, name)][0]) / res[(T, "v1")][0]
        d = res[(T, "v1")][1] - res[(T, name)][1]; b = boot(d, rng); ci[T] = (np.percentile(b, 2.5), np.percentile(b, 97.5))
        print(f"   {name} {T:8s} change {g[T]:+.1f}% | v1 - {name} daily MAE_share {d.mean():+.6f} [{ci[T][0]:+.6f}, {ci[T][1]:+.6f}], {name} better on {100 * np.mean(d > 0):.0f}% of days")
    if not CONF: print(f"   {name} -> {'PASS' if max(g.values()) >= 3 and min(g.values()) >= -3 else 'fail'}")
    else:
        a_ = np.mean([res[(T, name)][0] for T in TS]) < np.mean([res[(T, "v1")][0] for T in TS])
        b_ = [T for T in TS if ci[T][0] > 0]
        c_ = any(all(g[o] >= -3 and ci[o][1] >= 0 for o in TS if o != T) for T in b_)
        ok = a_ and bool(b_) and c_
        print(f"   {name} (a) lower mean MAE_share: {a_} | (b) interval above zero in: {b_ or 'none'} | (c) other target acceptable: {c_} -> {'ADOPT' if ok else 'not adopted'}")
        print(f"   DECISION: {'adopt k' if ok else 'nothing adopted, v1 stays'}")
else: print("   missing runs")
