"""PROTOCOL_V10.md, DESCRIPTIVE only (reads the development targets' demand, so no best weight found here is allowed as a rule): how the error of
Chicago and New York City changes when the final method v1 (taxi helpers only) and a bike-share variant are mixed with a weight w,
P_w = (1 - w) P_v1 + w P_variant, using the 3-run ensembles of the screens in out/ (demand-reading mode; heads eh and eo averaged equally).
It shows how much a per-target weight could bring at best (the ceiling) and what one common weight for both targets would do.
Share error = MAE_share on the observed slots. Strict trip errors = predicted shares x city total per slot (level x rhythm of the paper,
coarse_eval.py totals: helper cities and public data only), calendar slots checked against the observed slots as in v4_dev_strict.py.
Usage (from improve/): python v10_curve.py [variant ...]      default: k dk"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v9_common import srcs
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE = ns["totals"], ns["YP"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})                  # as predict_final.py (checked in chi_eval.py)
V1 = lambda T, r: {"chicago": f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": f"chicago+sf_for_nyc_p{r}"}[T]
KS = lambda T, r: "+".join(srcs(T, r))
VARS = sys.argv[1:] or ["k", "dk"]; N = 3; W = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
norm = lambda p: p / p.sum(1, keepdims=True)


def ens(T, sf, v):
    return {h: np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{sf(T, r)}_v{h}{v}_seed{r - 1}.npy")) for r in range(1, N + 1)], 0) for h in ("eh", "eo")}


R = {}                                                                   # (variant, target) -> list over W of (share error, strict MAE, strict RMSE)
for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float64); tot = Y.sum(2); nd = Y.shape[0]
    idx = [(d, s) for d in range(nd) for s in range(48) if tot[d, s] > 0]
    di, si, Tt, unit, lev = totals(T, T)
    assert len(di) == len(idx) and all(a == b and c == e for (a, c), b, e in zip(idx, di, si)), "calendar slots differ from observed slots"
    y = Y[di, si]; Sh = y / tot[di, si][:, None]
    b = ens(T, V1, ""); P0 = 0.5 * b["eh"] + 0.5 * b["eo"]
    for name in VARS:
        e = ens(T, KS, name); P1 = 0.5 * e["eh"] + 0.5 * e["eo"]; rows = []
        for w in W:
            P = (1 - w) * P0 + w * P1; err = P * Tt[:, None] - y
            rows.append((np.abs(P - Sh).mean(), np.abs(err).mean(), np.sqrt((err ** 2).mean())))
        R[(name, T)] = rows
for name in VARS:
    print(f"\n===== mix of v1 (taxi helpers only) and variant {name}:  P = (1 - w) * v1 + w * {name}   (3-run ensembles, development targets) =====")
    print("      w |   Chicago share err  change |  MAE   RMSE |    NYC share err  change |   MAE    RMSE | screen rule")
    c, n = R[(name, "chicago")], R[(name, "nyc")]
    for i, w in enumerate(W):
        gc, gn = [100 * (x[0][0] - x[i][0]) / x[0][0] for x in (c, n)]
        print(f"   {w:4.1f} |   {c[i][0]:.5f}  {gc:+6.1f}% | {c[i][1]:.3f} {c[i][2]:.3f} |   {n[i][0]:.5f}  {gn:+6.1f}% | {n[i][1]:.3f} {n[i][2]:.3f} | "
              f"{'pass' if max(gc, gn) >= 3 and min(gc, gn) >= -3 else '-'}")
    bc = min(range(len(W)), key=lambda i: c[i][0]); bn = min(range(len(W)), key=lambda i: n[i][0])
    print(f"   best w read off each target's own demand (an oracle, NOT a rule): Chicago w = {W[bc]:.1f} ({100 * (c[0][0] - c[bc][0]) / c[0][0]:+.1f}%), "
          f"NYC w = {W[bn]:.1f} ({100 * (n[0][0] - n[bn][0]) / n[0][0]:+.1f}%)")
