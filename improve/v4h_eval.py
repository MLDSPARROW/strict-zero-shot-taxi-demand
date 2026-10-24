"""PROTOCOL_V4.md addendum 2: held-out evaluation of Way 2 (switch d), fully strict calendar-slot predictions (the target demand is read here only).
Final method v1 = equal average of the eh and eo 10-run ensembles (helper partition r, seed r - 1; San Francisco: seeds 0-9 on the native helpers);
Way 2 = the same with the heads ehd and eod. City totals per slot (level x rhythm) exactly as coarse_eval.py / austin_eval.py, so they are identical
for both. Scored on the calendar slots, hourly for Washington, DC and its clusters (as the paper); the mean daily MAE difference v1 - d with the
7-day block bootstrap (2,000 draws), jobs-proportional and jobs-and-residents for reference. Reported as obtained.
Usage (from improve/): python v4h_eval.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); MC = f"{HERE}/../multicity/data"; O = f"{HERE}/out_cal"
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, boot = ns["totals"], ns["YP"], ns["boot"]
norm = lambda p: p / p.sum(1, keepdims=True)
TG = {"sf": ("sf", YP["sf"], False, lambda r: "chicago+nyc"),
      "dc": ("dc", YP["dc"], True, lambda r: f"chicago+nyc+sf_for_dc_p{r}"),
      "sf_zcta": ("sf", f"{MC}/sf_zcta_30min.npy", False, lambda r: f"chicago+nyc_for_sf_zcta_p{r}"),
      "dc_cluster": ("dc", f"{MC}/dc_cluster_30min.npy", True, lambda r: f"chicago+nyc_for_dc_cluster_p{r}+sf_for_dc_cluster_p{r}"),
      "austin": ("austin", f"{MC}/austin_2016_cell_30min.npy", False, lambda r: f"chicago+nyc+sf_for_austin_p{r}")}
rng = np.random.default_rng(0); summary = {}
for tag, (city, yfile, hourly, mat) in TG.items():
    runs = {h: [f"{O}/multi_{tag}_from_{mat(r)}_v{h}_seed{r - 1}.npy" for r in range(1, 11)] for h in ("eh", "eo", "ehd", "eod")}
    miss = sum(not os.path.exists(f) for h in runs for f in runs[h])
    if miss: print(f"\n===== {tag}: {miss} of 40 run files missing, not scored yet ====="); continue
    if tag == "austin":                                        # level and rhythm of austin_eval.py (248 days, no month factor)
        asrc = open(f"{HERE}/austin_eval.py").read(); ans = {"__file__": f"{HERE}/austin_eval.py"}
        exec(asrc[:asrc.index("Y = np.load(f\"{MC}/austin_2016_cell_30min.npy\")")], ans); di, si, Tt, ndays = ans["di"], ans["si"], ans["Tt"], ans["ndays"]
    else:
        di, si, Tt, unit, lev = totals(city, tag); ndays = None
    Y = np.load(yfile).astype(np.float64); nd = Y.shape[0]                                  # evaluation only
    def err(p):
        F = np.full(Y.shape, np.nan); F[di, si] = norm(p) * Tt[:, None]; E = F - Y
        if hourly: E = E.reshape(nd, 24, 2, -1).sum(2)
        E = E.reshape(nd, -1); ok = ~np.isnan(E); d_ = np.nanmean(np.abs(E), 1)
        return np.abs(E[ok]).mean(), np.sqrt((E[ok] ** 2).mean()), (d_[:ndays] if ndays else d_)
    ld = lambda h: [np.load(f) for f in runs[h]]
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); R = len(E_["jobs"]); n = len(di); jr = E_["jobs"] + E_["res"]
    base = {"Jobs-proportional": np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0), "Jobs + residents": np.repeat((jr / jr.sum())[None], n, 0)}
    res, single = {}, {}
    for name, hs in (("v1", ("eh", "eo")), ("d", ("ehd", "eod"))):
        A, B = ld(hs[0]), ld(hs[1]); res[name] = err(0.5 * norm(np.mean(A, 0)) + 0.5 * norm(np.mean(B, 0)))
        single[name] = np.array([err(0.5 * norm(a) + 0.5 * norm(b))[0] for a, b in zip(A, B)])
    print(f"\n===== {tag} ({city}, {R} regions, {n} calendar slots{', hourly' if hourly else ''}) =====")
    for k, p in base.items(): m, r_, _ = err(p); print(f"   {k:20s} MAE {m:.3f}  RMSE {r_:.3f}")
    for name in ("v1", "d"):
        print(f"   {'final method, v1' if name == 'v1' else 'with Way 2 (d)':20s} MAE {res[name][0]:.3f}  RMSE {res[name][1]:.3f} | single runs MAE {single[name].mean():.3f} +- {single[name].std(ddof=1):.3f}")
    diff = res["v1"][2] - res["d"][2]; b = boot(diff, rng); g = 100 * (res["v1"][0] - res["d"][0]) / res["v1"][0]
    print(f"   v1 minus d: mean daily MAE difference {diff.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], d better on {100 * (diff > 0).mean():.0f}% of days; "
          f"MAE change {g:+.2f}%, RMSE change {100 * (res['v1'][1] - res['d'][1]) / res['v1'][1]:+.2f}%")
    summary[tag] = (g, np.percentile(b, 2.5), np.percentile(b, 97.5))
print("\n===== summary (positive = Way 2 better) =====")
for t, (g, lo, hi) in summary.items(): print(f"   {t:11s} MAE change {g:+.2f}%  interval of the daily difference [{lo:+.4f}, {hi:+.4f}] -> {'above zero' if lo > 0 else ('below zero' if hi < 0 else 'includes zero')}")
