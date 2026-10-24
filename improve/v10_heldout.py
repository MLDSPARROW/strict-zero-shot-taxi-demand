"""PROTOCOL_V10.md: held-out evaluation of the hub-and-city ensemble HC, fully strict calendar-slot predictions (the target demand is read here only).
v1 = equal average of the eh and eo 10-run ensembles in out_cal/ (built exactly as v4h_eval.py, which reproduces the paper); g = scale-matched XGBoost
10-run ensemble (out_xgb/, xgb_heldout.py); HC = the rule hc() of v10_dev.py (airport regions keep the v1 shares; the rest of each slot is split by
the equal average of the city splits of v1 and g; Austin is the only held-out target with an airport region). City totals per slot (level x rhythm)
exactly as coarse_eval.py / austin_eval.py, identical for all rows. Scored on the calendar slots, hourly for Washington, DC and its clusters, Austin
on its 248 days, as the paper; mean daily MAE difference v1 - HC with the 7-day block bootstrap (2,000 draws). The adoption rule of the protocol is
applied at the end. Usage (from improve/): python v10_heldout.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); MC = f"{HERE}/../multicity/data"; O = f"{HERE}/out_cal"
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, boot = ns["totals"], ns["YP"], ns["boot"]
s = open(f"{HERE}/v10_dev.py").read(); nh = {"np": np}; exec(s[s.index("def hc"):s.index("rng = np.random")], nh); hc = nh["hc"]
norm = lambda p: p / p.sum(1, keepdims=True)
TG = {"sf": ("sf", YP["sf"], False, lambda r: "chicago+nyc"),
      "dc": ("dc", YP["dc"], True, lambda r: f"chicago+nyc+sf_for_dc_p{r}"),
      "sf_zcta": ("sf", f"{MC}/sf_zcta_30min.npy", False, lambda r: f"chicago+nyc_for_sf_zcta_p{r}"),
      "dc_cluster": ("dc", f"{MC}/dc_cluster_30min.npy", True, lambda r: f"chicago+nyc_for_dc_cluster_p{r}+sf_for_dc_cluster_p{r}"),
      "austin": ("austin", f"{MC}/austin_2016_cell_30min.npy", False, lambda r: f"chicago+nyc+sf_for_austin_p{r}")}
rng = np.random.default_rng(0); summary = {}
for tag, (city, yfile, hourly, mat) in TG.items():
    runs = {h: [f"{O}/multi_{tag}_from_{mat(r)}_v{h}_seed{r - 1}.npy" for r in range(1, 11)] for h in ("eh", "eo")}; gx = f"{HERE}/out_xgb/xgb_sm_{tag}.npy"
    miss = sum(not os.path.exists(f) for h in runs for f in runs[h]) + int(not os.path.exists(gx))
    if miss: print(f"\n===== {tag}: {miss} input files missing, not scored ====="); continue
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
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); n = len(di); jr = E_["jobs"] + E_["res"]
    v1 = 0.5 * norm(np.mean([np.load(f) for f in runs["eh"]], 0)) + 0.5 * norm(np.mean([np.load(f) for f in runs["eo"]], 0))
    g = norm(np.load(gx).astype(np.float64)); air = np.load(f"{HERE}/data/enpl_{tag}.npy") > 0
    rows = {"Jobs-proportional": np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0), "Jobs + residents": np.repeat((jr / jr.sum())[None], n, 0),
            "XGBoost, scale-matched": g, "final method v1": v1, "HC (this protocol)": hc(v1, g, air)}
    res = {k: err(p) for k, p in rows.items()}
    print(f"\n===== {tag} ({city}, {Y.shape[2]} regions, {int(air.sum())} airport regions, {n} calendar slots{', hourly' if hourly else ''}) =====")
    for k, (m, r_, _) in res.items(): print(f"   {k:24s} MAE {m:.3f}  RMSE {r_:.3f}")
    diff = res["final method v1"][2] - res["HC (this protocol)"][2]; b = boot(diff, rng)
    gm = 100 * (res["final method v1"][0] - res["HC (this protocol)"][0]) / res["final method v1"][0]
    gr = 100 * (res["final method v1"][1] - res["HC (this protocol)"][1]) / res["final method v1"][1]
    print(f"   v1 - HC: mean daily MAE difference {diff.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], HC better on {100 * (diff > 0).mean():.0f}% of days; "
          f"MAE change {gm:+.2f}%, RMSE change {gr:+.2f}% (positive = HC better)")
    summary[tag] = (gm, gr, np.percentile(b, 2.5), np.percentile(b, 97.5))
print("\n===== summary (positive = HC better than v1) =====")
for t, (gm, gr, lo, hi) in summary.items(): print(f"   {t:11s} MAE change {gm:+6.2f}%  RMSE change {gr:+6.2f}%  interval of the daily MAE difference [{lo:+.4f}, {hi:+.4f}]")
if len(summary) == 5:
    m = np.mean([v[0] for v in summary.values()]); r = np.mean([v[1] for v in summary.values()])
    cm = sum(v[0] > 0 for v in summary.values()); cr = sum(v[1] > 0 for v in summary.values())
    ok = m > 0 and r > 0 and cm >= 3 and cr >= 3
    print(f"   (a) mean MAE change {m:+.2f}% > 0: {m > 0} | (b) mean RMSE change {r:+.2f}% > 0: {r > 0} | (c) better in MAE in {cm} of 5 and in RMSE in {cr} of 5 (need 3 each): {cm >= 3 and cr >= 3}")
    print(f"   DECISION by the rule of PROTOCOL_V10: {'HC is ADOPTED as the final method' if ok else 'HC is NOT adopted; v1 stays the final method'}")
