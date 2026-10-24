"""DESCRIPTIVE check of a helper-envelope calibration (development targets only; nothing here is adopted).
Law: log(yearly share) = a + beta * log(people share), people = LODES jobs + ACS population, airports left out, weights = people share
(v11_law.py). Rule under test: if the beta of the predicted yearly shares of the target lies outside the range of the betas of the target's
helpers at the target's scale, tilt the predicted shares of the non-airport regions by people_share ** delta (the smallest change in the
Kullback-Leibler sense that moves the slope by delta), with delta moving beta to the nearest end of the range; airport regions keep their
predicted mass in every slot. Shown for comparison: delta to the mean of the helpers, and to the TRUE beta of the target (an oracle, not
allowed). Final method v1 = (eh + eo)/2, 10-run ensembles; strict trip errors with the city totals of the paper.
Usage (from improve/): python v11_envelope.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/v11_law.py").read(); ns = {"__file__": f"{HERE}/v11_law.py"}
exec(src[:src.index('print("===== true yearly')], ns); people, law, yearly = ns["people"], ns["law"], ns["yearly"]
src = open(f"{HERE}/coarse_eval.py").read(); ns2 = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns2); totals, YP, SHARE = ns2["totals"], ns2["YP"], ns2["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
HEL = {"chicago": lambda r: [f"nyc_for_chicago_p{r}", f"sf_for_chicago_p{r}"], "nyc": lambda r: ["chicago", f"sf_for_nyc_p{r}"]}
norm = lambda q: q / q.sum(1, keepdims=True)


def tilt(P, p, air, d):                                  # multiply non-airport shares by p ** d, keep the airport mass of every slot
    Q = P.copy(); na = ~air; t = P[:, na] * p[na][None] ** d
    Q[:, na] = t / t.sum(1, keepdims=True) * P[:, na].sum(1, keepdims=True); return Q


for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float64); tot = Y.sum(2); di, si, Tt, unit, lev = totals(T, T); y = Y[di, si]
    p, air = people(T)
    hb = {}
    for r in range(1, 11):
        for h in HEL[T](r):
            ph, ah = people(h); hb.setdefault(h.split("_p")[0], []).append(law(yearly(h), ph, ah)[0])
    hb = {k: float(np.mean(v)) for k, v in hb.items()}; lo, hi, mean = min(hb.values()), max(hb.values()), float(np.mean(list(hb.values())))
    H = {h: np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0) for h in ("eh", "eo")}
    P = 0.5 * H["eh"] + 0.5 * H["eo"]
    sy = (P * Tt[:, None]).sum(0); sy /= sy.sum(); bp = law(sy, p, air)[0]; bt = law(yearly(T), p, air)[0]
    print(f"\n===== {T}: helper betas at the target's scale {', '.join(f'{k} {v:.2f}' for k, v in hb.items())} -> range [{lo:.2f}, {hi:.2f}], mean {mean:.2f}; "
          f"predicted beta of v1 {bp:.2f}; true beta {bt:.2f} =====")
    for name, target_beta in (("v1 unchanged", bp), ("RULE: to the nearest end of the helper range", min(max(bp, lo), hi)),
                              ("to the helper mean", mean), ("to the TRUE beta (oracle, not allowed)", bt)):
        Q = tilt(P, p, air, target_beta - bp); e = Q * Tt[:, None] - y; ea = Q * tot[di, si][:, None] - y
        qy = (Q * Tt[:, None]).sum(0); qy /= qy.sum(); ty = yearly(T)
        print(f"   {name:46s} delta {target_beta - bp:+.2f} | strict MAE {np.abs(e).mean():.3f} RMSE {np.sqrt((e ** 2).mean()):.3f} | "
              f"alloc-only MAE {np.abs(ea).mean():.3f} RMSE {np.sqrt((ea ** 2).mean()):.3f} | yearly TV {0.5 * np.abs(qy - ty).sum():.3f}")
