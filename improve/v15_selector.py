"""PROTOCOL_V15.md: helper-based hard selector among the allocators v1 (final method), g (scale-matched XGBoost), jr (jobs + residents) and tilt (the ridge tilt of PROTOCOL_V13/V14, lambda 1).
Pseudo-target errors (allocation MAE with the true slot totals of the pseudo-target, observed slots) come from the share helpers of the target only: v1 and tilt from the pseudo-target files of
v14_gate.py, g from xgb_pseudo.py, jr from public data. Rule (frozen): a challenger is eligible if its mean pseudo-target MAE is at least 5% below that of v1 and at least 2% below on every pseudo-target;
among the eligible the lowest mean MAE wins (ties of 1e-6 relative go to the earlier of g, jr, tilt); otherwise v1. The target demand is read only to score.
Usage (from improve/): python v15_selector.py dev | python v15_selector.py heldout"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
g_ = open(f"{HERE}/v14_gate.py").read(); N = {"__file__": f"{HERE}/v14_gate.py", "__name__": "v14prefix"}; exec(g_[:g_.index("rng = np.random.default_rng(0); mode")], N)
(pseudo, fit, tilt, mae, feats, airports, PSEUDO, TG, totals, YP, boot, SRC, norm, OUT, O, MCD, LAM) = (N[k] for k in ("pseudo", "fit", "tilt", "mae", "feats", "airports", "PSEUDO", "TG", "totals", "YP", "boot", "SRC", "norm", "OUT", "O", "MCD", "LAM"))
NAMES = ("v1", "g", "jr", "tilt"); MEAN_MIN, FOLD_MIN, TIE = 0.05, 0.02, 1e-6


def select(E):
    E = np.asarray(E, float); assert E.shape[1] == 4 and np.all(np.isfinite(E)) and np.all(E[:, 0] > 0)
    mean = E.mean(0); mg = (mean[0] - mean) / mean[0]; fg = (E[:, [0]] - E) / E[:, [0]]; elig = [c for c in (1, 2, 3) if mg[c] >= MEAN_MIN and np.all(fg[:, c] >= FOLD_MIN)]
    best = 0
    for c in elig:                                         # fixed order g, jr, tilt; a later one replaces the incumbent only if lower by more than TIE of the v1 mean
        if best == 0 or mean[c] < mean[best] - TIE * mean[0]: best = c
    return best, mean, mg, fg, elig


def pseudo_errors(T):
    P = pseudo(T); E = []
    for j, c in enumerate(P):
        tr = [x for k, x in enumerate(P) if k != j]; beta, mu, sd = fit(tr, LAM); tl = tilt(c["share"], c["Z"], c["air"], beta, mu, sd)
        key = [(t, s) for t, s in PSEUDO[T] if t == c["tag"]][0]; gx = norm(np.load(f"{HERE}/out_xgb/xgbps__{key[0]}__{key[1](1)}.npy").astype(np.float64)); assert gx.shape == c["share"].shape
        Ex = np.load(f"{HERE}/data/extra_{c['tag']}.npz"); jr = (Ex["jobs"] + Ex["res"]) / (Ex["jobs"] + Ex["res"]).sum()
        E.append([mae(c["share"], c), mae(gx, c), mae(np.repeat(jr[None], len(gx), 0), c), mae(tl, c)])
    return P, np.array(E)


def candidates(T, P):
    if T in ("chicago", "nyc"):
        H = {h: np.mean([norm(np.load(f"{OUT}/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0) for h in ("eh", "eo")}
        gx = np.load(f"{HERE}/out_xgb/xgb_sm_{T}.npy").astype(np.float64)
    else:
        mat = TG[T][3]; H = {h: np.mean([norm(np.load(f"{O}/multi_{T}_from_{mat(r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0) for h in ("eh", "eo")}
        gx = np.load(f"{HERE}/out_xgb/xgb_sm_{T}.npy").astype(np.float64)
    v1 = 0.5 * H["eh"] + 0.5 * H["eo"]; Ex = np.load(f"{HERE}/data/extra_{T}.npz"); jr = (Ex["jobs"] + Ex["res"]) / (Ex["jobs"] + Ex["res"]).sum()
    beta, mu, sd = fit(P, LAM); tl = tilt(v1, feats(T), airports(T), beta, mu, sd)
    return dict(zip(NAMES, (v1, norm(gx), norm(np.repeat(jr[None], len(v1), 0)), tl)))


def scorer(T):
    if T in ("chicago", "nyc"):
        Y = np.load(YP[T]).astype(np.float64); nd = Y.shape[0]; di, si, Tt, unit, lev = totals(T, T); y = Y[di, si]
        def err(p):
            e = p * Tt[:, None] - y; a = np.abs(e); return a.mean(), np.sqrt((e ** 2).mean()), np.bincount(di, weights=a.mean(1), minlength=nd) / np.maximum(np.bincount(di, minlength=nd), 1)
        return err
    city, yfile, hourly, mat = TG[T]
    if T == "austin":
        asrc = open(f"{HERE}/austin_eval.py").read(); ans = {"__file__": f"{HERE}/austin_eval.py"}; MC = MCD
        exec(asrc[:asrc.index("Y = np.load(f\"{MC}/austin_2016_cell_30min.npy\")")], ans); di, si, Tt, ndays = ans["di"], ans["si"], ans["Tt"], ans["ndays"]
    else:
        di, si, Tt, unit, lev = totals(city, T); ndays = None
    Y = np.load(yfile).astype(np.float64); nd = Y.shape[0]
    def err(p):
        F = np.full(Y.shape, np.nan); F[di, si] = norm(p) * Tt[:, None]; E = F - Y
        if hourly: E = E.reshape(nd, 24, 2, -1).sum(2)
        E = E.reshape(nd, -1); ok = ~np.isnan(E); d_ = np.nanmean(np.abs(E), 1)
        return np.abs(E[ok]).mean(), np.sqrt((E[ok] ** 2).mean()), (d_[:ndays] if ndays else d_)
    return err


rng = np.random.default_rng(0); mode = sys.argv[1]; targets = ["chicago", "nyc"] if mode == "dev" else ["sf", "dc", "sf_zcta", "dc_cluster", "austin"]; summ = {}; chosen = []
for T in targets:
    P, E = pseudo_errors(T); best, mean, mg, fg, elig = select(E)
    print(f"\n===== {T}: pseudo-targets {[c['tag'] for c in P]} =====")
    print("   pseudo-target allocation MAE (columns v1, g, jr, tilt):"); [print(f"      {c['tag']:24s} " + "  ".join(f"{x:.5f}" for x in E[j])) for j, c in enumerate(P)]
    print("   mean gain over v1: " + "  ".join(f"{n} {100 * m:+.1f}%" for n, m in zip(NAMES, mg)) + " | worst helper gain: " + "  ".join(f"{n} {100 * fg[:, c].min():+.1f}%" for c, n in enumerate(NAMES)))
    print(f"   eligible challengers (mean >= 5%, every helper >= 2%): {[NAMES[c] for c in elig]} -> SELECTED {NAMES[best]}")
    C = candidates(T, P); err = scorer(T); res = {n: err(C[n]) for n in NAMES}
    for n in NAMES: print(f"   {n:5s} MAE {res[n][0]:.3f}  RMSE {res[n][1]:.3f}" + ("   <- selected" if NAMES.index(n) == best else ""))
    diff = res["v1"][2] - res[NAMES[best]][2]; bb = boot(diff, rng) if best else None
    gm = 100 * (res["v1"][0] - res[NAMES[best]][0]) / res["v1"][0]; gr = 100 * (res["v1"][1] - res[NAMES[best]][1]) / res["v1"][1]
    bestn = min(NAMES, key=lambda n: res[n][0])
    print(f"   selected vs v1: MAE change {gm:+.2f}%, RMSE change {gr:+.2f}%" + (f", daily MAE difference {diff.mean():+.4f} [{np.percentile(bb, 2.5):+.4f}, {np.percentile(bb, 97.5):+.4f}]" if best else "") + f" | lowest MAE among the four on the target: {bestn}")
    summ[T] = (gm, gr, NAMES[best], bestn); chosen.append(NAMES[best])
print("\n===== summary (positive = selector better than v1) =====")
for t, (gm, gr, s, b) in summ.items(): print(f"   {t:11s} selected {s:5s} MAE change {gm:+6.2f}%  RMSE change {gr:+6.2f}%  (lowest-MAE candidate on the target: {b})")
m = np.mean([v[0] for v in summ.values()]); r = np.mean([v[1] for v in summ.values()]); cm = sum(v[0] > 0 for v in summ.values()); cr = sum(v[1] > 0 for v in summ.values()); worst = min(v[0] for v in summ.values())
print(f"   selections {dict((n, chosen.count(n)) for n in NAMES)}; mean MAE change {m:+.2f}%, mean RMSE change {r:+.2f}%, better in MAE in {cm} of {len(summ)} and in RMSE in {cr}; worst MAE change {worst:+.2f}%")
if mode == "heldout":
    adopt = m > 0 and r > 0 and cm >= 3 and cr >= 3 and worst >= -5
    print(f"   adoption rule of PROTOCOL_V15 (mean MAE and RMSE change above zero, better in at least 3 of 5 in each, no MAE change below -5%): {'ADOPTED' if adopt else 'NOT adopted, v1 stays'}")
