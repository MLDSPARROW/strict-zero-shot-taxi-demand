"""PROTOCOL_V10.md, development targets Chicago and NYC (designed on these two cities, so this is a development result, not a held-out one).
Hub-and-city ensemble (HC): airport regions (enplanements > 0, data/enpl_<city>.npy) keep the shares of the final method v1 = (eh + eo)/2 in every
slot; the remaining mass of the slot is split over the other regions by the equal average of the city split of v1 and the city split of the
scale-matched XGBoost baseline (xgb_save.py, out_xgb/; each city split renormalised over the non-airport regions). 10-run ensembles. Strict trip
MAE and RMSE with the city totals of the paper; mean daily MAE difference v1 - HC with the 7-day block bootstrap (2,000 draws, coarse_eval.py).
Usage (from improve/): python v10_dev.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE, boot = ns["totals"], ns["YP"], ns["SHARE"], ns["boot"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
norm = lambda q: q / q.sum(1, keepdims=True)


def hc(v1, g, air):                                       # the rule of PROTOCOL_V10
    if not air.any(): return 0.5 * v1 + 0.5 * g
    na = ~air; P = np.empty_like(v1); P[:, air] = v1[:, air]
    c = 0.5 * v1[:, na] / v1[:, na].sum(1, keepdims=True) + 0.5 * g[:, na] / g[:, na].sum(1, keepdims=True)
    P[:, na] = (1 - v1[:, air].sum(1, keepdims=True)) * c; return P


rng = np.random.default_rng(0)
for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float64); nd = Y.shape[0]; di, si, Tt, unit, lev = totals(T, T); y = Y[di, si]
    air = np.load(f"{HERE}/data/enpl_{T}.npy") > 0
    H = {h: np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0) for h in ("eh", "eo")}
    v1 = 0.5 * H["eh"] + 0.5 * H["eo"]; g = norm(np.load(f"{HERE}/out_xgb/xgb_sm_{T}.npy").astype(np.float64))
    out, daily = {}, {}
    for k, P in (("final method v1", v1), ("XGBoost scale-matched", g), ("HC (this protocol)", hc(v1, g, air))):
        e = P * Tt[:, None] - y; a = np.abs(e); out[k] = (a.mean(), np.sqrt((e ** 2).mean()))
        daily[k] = np.bincount(di, weights=a.mean(1), minlength=nd) / np.maximum(np.bincount(di, minlength=nd), 1)
    print(f"\n===== {T}: {len(air)} regions, {int(air.sum())} airport regions, {len(di)} slots =====")
    for k, (m, r_) in out.items(): print(f"   {k:24s} strict MAE {m:.3f}  RMSE {r_:.3f}")
    d = daily["final method v1"] - daily["HC (this protocol)"]; b = boot(d, rng)
    m0, r0 = out["final method v1"]; m1, r1 = out["HC (this protocol)"]
    print(f"   v1 - HC: mean daily MAE difference {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], HC better on {100 * (d > 0).mean():.0f}% of days; "
          f"MAE change {100 * (m0 - m1) / m0:+.2f}%, RMSE change {100 * (r0 - r1) / r0:+.2f}% (positive = HC better)")
