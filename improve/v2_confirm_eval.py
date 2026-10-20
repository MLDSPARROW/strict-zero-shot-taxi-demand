"""PROTOCOL_V2.md addendum 2: confirmation on the development targets. 10-run ensembles (r = 1..10) of eh, eo and ep;
v1 = eh/2 + eo/2, v2 = eh/2 + eo/4 + ep/4. Metric: MAE_share on the target's observed slots; daily series for a 7-day block
bootstrap (2,000 draws, as coarse_eval.py). Adoption rule: (a) mean MAE_share over both targets lower for v2; (b) interval
of v1 - v2 above zero in at least one target; (c) in the other, v2 not > 3% worse and its interval not entirely below zero.
Usage (from improve/): python v2_confirm_eval.py"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); B = f"{HERE}/../bench77/data"
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[src.index("def boot"):src.index("norm = lambda")], {**ns, "np": np}, ns); boot = ns["boot"]
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
YF = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy"}
norm = lambda p: p / p.sum(1, keepdims=True)
R = range(1, 11); res = {}
for T in ("chicago", "nyc"):
    Y = np.load(YF[T]).astype(np.float64); tot = Y.sum(2)
    idx = [(d, s) for d in range(Y.shape[0]) for s in range(48) if tot[d, s] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Sh = Y[di, si] / tot[di, si][:, None]; Tt = tot[di, si]
    H = {v: np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{v}_seed{r - 1}.npy")) for r in R], 0) for v in ("eh", "eo", "ep")}
    C = {"v1 = eh/2 + eo/2": H["eh"] / 2 + H["eo"] / 2, "v2 = eh/2 + eo/4 + ep/4": H["eh"] / 2 + H["eo"] / 4 + H["ep"] / 4, **{f"head {v}": H[v] for v in H}}
    print(f"\n===== {T}: 10-run ensembles, {len(idx)} observed slots =====")
    for k, P in C.items():
        a = np.abs(P - Sh); daily = np.bincount(di, weights=a.mean(1), minlength=Y.shape[0]) / np.maximum(np.bincount(di, minlength=Y.shape[0]), 1)
        e = P * Tt[:, None] - Y[di, si]; res[(T, k)] = (a.mean(), daily)
        print(f"   {k:24s} MAE_share {a.mean():.5f} | allocation-only MAE {np.abs(e).mean():.3f} RMSE {np.sqrt((e ** 2).mean()):.3f}")
v1, v2 = "v1 = eh/2 + eo/2", "v2 = eh/2 + eo/4 + ep/4"; rng = np.random.default_rng(0); ci, g = {}, {}
print("\n===== adoption rule =====")
for T in ("chicago", "nyc"):
    d = res[(T, v1)][1] - res[(T, v2)][1]; b = boot(d, rng); ci[T] = (np.percentile(b, 2.5), np.percentile(b, 97.5))
    g[T] = 100 * (res[(T, v1)][0] - res[(T, v2)][0]) / res[(T, v1)][0]
    print(f"   {T}: v1 - v2 daily MAE_share {d.mean():+.6f} [{ci[T][0]:+.6f}, {ci[T][1]:+.6f}], v2 better on {100 * np.mean(d > 0):.0f}% of days, change {g[T]:+.1f}%")
a_ = np.mean([res[(T, v2)][0] for T in ci]) < np.mean([res[(T, v1)][0] for T in ci])
b_ = [T for T in ci if ci[T][0] > 0]
c_ = any(all(g[o] >= -3 and ci[o][1] >= 0 for o in ci if o != T) for T in b_)
print(f"   (a) lower mean MAE_share: {a_} | (b) interval above zero in: {b_ or 'none'} | (c) other target acceptable: {c_}")
print(f"   DECISION: {'ADOPT v2' if a_ and b_ and c_ else 'do not adopt v2 (v1 stays)'}")
