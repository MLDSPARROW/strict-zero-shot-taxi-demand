"""DESCRIPTIVE (development targets only; nothing here is adopted): the Chicago-NYC Pareto frontier of fixed mixes of the trained heads.
P = a * eh + b * eo + c * ep (+ d * XGBoost scale-matched if out_xgb/ exists), weights on a simplex grid (step 0.05), 10-run ensembles per
head, the same mix for both cities. Strict trip MAE and RMSE with the city totals of the paper. Lists the mixes that beat the final method
v1 = (eh + eo)/2 on all four numbers (MAE and RMSE in both cities) and the mix with the best worst-case relative change.
Usage (from improve/): python v11_pareto.py [--xgb]"""
import os, sys, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE = ns["totals"], ns["YP"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
norm = lambda q: q / q.sum(1, keepdims=True)
XGB = "--xgb" in sys.argv; names = ["eh", "eo", "ep"] + (["xgb"] if XGB else []); step = 0.05 if not XGB else 0.1
grid = [w for w in itertools.product(*[np.round(np.arange(0, 1 + 1e-9, step), 3)] * len(names)) if abs(sum(w) - 1) < 1e-6]
res = {}
for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float64); di, si, Tt, unit, lev = totals(T, T); y = Y[di, si]
    H = [np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0).astype(np.float32) for h in names if h != "xgb"]
    if XGB: H.append(np.load(f"{HERE}/out_xgb/xgb_sm_{T}.npy").astype(np.float32))
    L = [(h * Tt[:, None]).astype(np.float32) for h in H]; y = y.astype(np.float32)
    for w in grid:
        e = sum(wk * Lk for wk, Lk in zip(w, L) if wk > 0) - y; res[(T, w)] = (float(np.abs(e).mean()), float(np.sqrt((e.astype(np.float64) ** 2).mean())))
    del H, L
v1 = tuple([0.5, 0.5] + [0.0] * (len(names) - 2)); b = {T: res[(T, v1)] for T in ("chicago", "nyc")}
rel = lambda w: [100 * (b[T][k] - res[(T, w)][k]) / b[T][k] for T in ("chicago", "nyc") for k in (0, 1)]   # positive = better than v1
print(f"heads {names}; v1 = {v1}: Chicago {b['chicago'][0]:.3f} / {b['chicago'][1]:.3f}, NYC {b['nyc'][0]:.3f} / {b['nyc'][1]:.3f}; {len(grid)} mixes")
dom = sorted([w for w in grid if min(rel(w)) > 0], key=lambda w: -min(rel(w)))
print(f"mixes better than v1 on all four numbers (Chicago MAE, RMSE, NYC MAE, RMSE): {len(dom)}")
for w in dom[:12]:
    r = rel(w); print(f"   {dict(zip(names, w))}: Chicago {res[('chicago', w)][0]:.3f} / {res[('chicago', w)][1]:.3f}, NYC {res[('nyc', w)][0]:.3f} / {res[('nyc', w)][1]:.3f} | changes {', '.join(f'{x:+.1f}%' for x in r)}")
best = max(grid, key=lambda w: min(rel(w))); r = rel(best)
print(f"best worst-case mix: {dict(zip(names, best))}: Chicago {res[('chicago', best)][0]:.3f} / {res[('chicago', best)][1]:.3f}, NYC {res[('nyc', best)][0]:.3f} / {res[('nyc', best)][1]:.3f} | changes {', '.join(f'{x:+.1f}%' for x in r)}")
for k, w in (("equal mix of all", tuple([round(1 / len(names), 3)] * len(names))),):
    if (("chicago", w) in res): r = rel(w); print(f"{k}: changes {', '.join(f'{x:+.1f}%' for x in r)}")
