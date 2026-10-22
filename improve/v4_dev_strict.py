"""PROTOCOL_V4.md, descriptive (computed after the confirmation of d, development targets only): fully strict trip MAE and RMSE of
v1 and of d on Chicago and New York City. Shares = the ten-run ensembles of the confirmation (runs r = 1..10 of eh, eo, ehd, eod in
out/, demand-reading mode; the calendar slots of these two cities coincide with the observed slots, checked below), final = equal
average of the two heads; city totals per slot = level x rhythm of the paper (coarse_eval.py totals, helper cities and public data
only). The v1 numbers must reproduce the paper (Chicago 1.83 / 5.21, NYC 7.45 / 17.59, within 0.02). Paired 7-day block bootstrap
(2,000 draws) of the daily MAE difference. Usage (from improve/): python v4_dev_strict.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, boot, SHARE = ns["totals"], ns["YP"], ns["boot"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})                  # as predict_final.py (checked in chi_eval.py)
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
PAPER = {"chicago": (1.83, 5.21), "nyc": (7.45, 17.59)}
norm = lambda p: p / p.sum(1, keepdims=True); rng = np.random.default_rng(0)
for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float64); tot = Y.sum(2); nd = Y.shape[0]
    idx = [(d, s) for d in range(nd) for s in range(48) if tot[d, s] > 0]
    di, si, Tt, unit, lev = totals(T, T)
    assert len(di) == len(idx) and all(a == b and c == e for (a, c), b, e in zip(idx, di, si)), "calendar slots differ from observed slots"
    y = Y[di, si]
    ld = lambda h: np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0)
    P = {"v1": 0.5 * ld("eh") + 0.5 * ld("eo"), "d": 0.5 * ld("ehd") + 0.5 * ld("eod")}
    print(f"\n===== {T}: {len(di)} calendar slots, level unit {unit} (predicted yearly total {lev:,.0f}, observed {Y.sum():,.0f}) =====")
    daily, out = {}, {}
    for k, p in P.items():
        lam = norm(p) * Tt[:, None]; e = lam - y; a = np.abs(e)
        daily[k] = np.bincount(di, weights=a.mean(1), minlength=nd) / np.maximum(np.bincount(di, minlength=nd), 1)
        out[k] = (a.mean(), np.sqrt((e ** 2).mean()))
        print(f"   {k:3s} strict MAE {out[k][0]:.3f}  RMSE {out[k][1]:.3f}")
    pm, pr = PAPER[T]; dm = out["v1"][0] - pm
    print(f"   reproduction check, v1 against the paper {pm} / {pr}: MAE {dm:+.3f}, RMSE {out['v1'][1] - pr:+.3f} -> {'OK (within 0.02)' if abs(dm) <= 0.02 else 'REPORT: outside 0.02'}")
    d = daily["v1"] - daily["d"]; b = boot(d, rng)
    print(f"   v1 - d daily MAE: {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], d better on {100 * np.mean(d > 0):.0f}% of days; "
          f"MAE change {100 * (out['v1'][0] - out['d'][0]) / out['v1'][0]:+.2f}%, RMSE change {100 * (out['v1'][1] - out['d'][1]) / out['v1'][1]:+.2f}% (positive = d better)")
