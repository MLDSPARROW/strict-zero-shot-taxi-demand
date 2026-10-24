"""DESCRIPTIVE (development targets only; nothing here is adopted): hub-and-city factorisation of the allocation.
P[t, airport regions] = v1[t, airport regions] (the airport mass and split of the final method v1 = (eh + eo)/2, which has an airport mechanism);
P[t, other regions] = (1 - airport mass of v1 in slot t) x the city split of a mix sum_k w_k M_k over the models M = eh, eo, ep and the scale-matched
XGBoost baseline (out_xgb/), renormalised over the non-airport regions. The same weights for both cities, simplex grid with step 0.1.
Reports the plain mixes and the factorised mixes that beat v1 on all four numbers (strict MAE and RMSE in both cities) and the best worst case.
Usage (from improve/): python v11_hybrid.py"""
import os, sys, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE = ns["totals"], ns["YP"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
norm = lambda q: q / q.sum(1, keepdims=True); names = ["eh", "eo", "ep", "xgb"]
grid = [w for w in itertools.product(*[np.round(np.arange(0, 1.0001, 0.1), 2)] * 4) if abs(sum(w) - 1) < 1e-6]
res = {}
for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float64); di, si, Tt, unit, lev = totals(T, T); y = Y[di, si].astype(np.float32)
    air = np.load(f"{HERE}/data/enpl_{T}.npy") > 0; na = ~air
    M = [np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0).astype(np.float32) for h in names[:3]]
    M.append(norm(np.load(f"{HERE}/out_xgb/xgb_sm_{T}.npy").astype(np.float64)).astype(np.float32))
    v1 = 0.5 * M[0] + 0.5 * M[1]; am = v1[:, air].sum(1, keepdims=True)                       # airport mass of v1 per slot
    Cn = [m[:, na] / m[:, na].sum(1, keepdims=True) for m in M]                                  # city split of each model (non-airport regions)
    tt = Tt[:, None].astype(np.float32)
    for w in grid:
        mix = sum(wk * m for wk, m in zip(w, M) if wk > 0); e = mix * tt - y
        res[("plain", T, w)] = (float(np.abs(e).mean()), float(np.sqrt((e.astype(np.float64) ** 2).mean())))
        cs = sum(wk * c for wk, c in zip(w, Cn) if wk > 0); P = np.empty_like(v1); P[:, air] = v1[:, air]; P[:, na] = (1 - am) * cs / cs.sum(1, keepdims=True)
        e = P * tt - y; res[("hub+city", T, w)] = (float(np.abs(e).mean()), float(np.sqrt((e.astype(np.float64) ** 2).mean())))
    del M, Cn
w1 = (0.5, 0.5, 0.0, 0.0); b = {T: res[("plain", T, w1)] for T in ("chicago", "nyc")}
print(f"v1: Chicago {b['chicago'][0]:.3f} / {b['chicago'][1]:.3f}, NYC {b['nyc'][0]:.3f} / {b['nyc'][1]:.3f}  (weights over {names})")
for kind in ("plain", "hub+city"):
    rel = lambda w: [100 * (b[T][k] - res[(kind, T, w)][k]) / b[T][k] for T in ("chicago", "nyc") for k in (0, 1)]
    dom = sorted([w for w in grid if min(rel(w)) > 0], key=lambda w: -min(rel(w)))
    print(f"\n== {kind}: {len(dom)} of {len(grid)} mixes beat v1 on all four numbers (Chicago MAE, RMSE, NYC MAE, RMSE) ==")
    for w in dom[:10]:
        r = rel(w); print(f"   {dict(zip(names, w))}: Chicago {res[(kind, 'chicago', w)][0]:.3f} / {res[(kind, 'chicago', w)][1]:.3f}, NYC {res[(kind, 'nyc', w)][0]:.3f} / {res[(kind, 'nyc', w)][1]:.3f} | {', '.join(f'{x:+.1f}%' for x in r)}")
    best = max(grid, key=lambda w: min(rel(w))); r = rel(best)
    print(f"   best worst case: {dict(zip(names, best))}: Chicago {res[(kind, 'chicago', best)][0]:.3f} / {res[(kind, 'chicago', best)][1]:.3f}, NYC {res[(kind, 'nyc', best)][0]:.3f} / {res[(kind, 'nyc', best)][1]:.3f} | {', '.join(f'{x:+.1f}%' for x in r)}")
    for lab, w in (("xgb alone", (0.0, 0.0, 0.0, 1.0)), ("v1 half, xgb half (0.2, 0.3, 0, 0.5)", (0.2, 0.3, 0.0, 0.5)), ("eh, eo, xgb near thirds (0.3, 0.3, 0, 0.4)", (0.3, 0.3, 0.0, 0.4))):
        if (kind, "chicago", w) not in res: continue
        r = rel(w); print(f"   {lab}: Chicago {res[(kind, 'chicago', w)][0]:.3f} / {res[(kind, 'chicago', w)][1]:.3f}, NYC {res[(kind, 'nyc', w)][0]:.3f} / {res[(kind, 'nyc', w)][1]:.3f} | {', '.join(f'{x:+.1f}%' for x in r)}")
