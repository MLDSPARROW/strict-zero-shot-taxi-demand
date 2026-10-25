"""PROTOCOL_V12.md: second look at the five held-out targets (SF, DC hourly, sf_zcta, dc_cluster hourly, Austin 248 days) with the fixed rules R1 to R3 of v12_rules.py.
Same inputs, city totals, calendar slots and bootstrap as v10_heldout.py (v1 = out_cal/ ten-run ensembles of eh and eo, g = out_xgb/, jr = jobs + residents of data/extra_<tag>.npz).
The target demand is read here only to score. The held-out numbers of the jobs + residents allocation were already known (PROTOCOL_V10.md), so this is a second look and not an
independent test. Usage (from improve/): python v12_heldout.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); MC = f"{HERE}/../multicity/data"; O = f"{HERE}/out_cal"
from v12_rules import rules, norm
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, boot = ns["totals"], ns["YP"], ns["boot"]
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
    if tag == "austin":
        asrc = open(f"{HERE}/austin_eval.py").read(); ans = {"__file__": f"{HERE}/austin_eval.py"}
        exec(asrc[:asrc.index("Y = np.load(f\"{MC}/austin_2016_cell_30min.npy\")")], ans); di, si, Tt, ndays = ans["di"], ans["si"], ans["Tt"], ans["ndays"]
    else:
        di, si, Tt, unit, lev = totals(city, tag); ndays = None
    Y = np.load(yfile).astype(np.float64); nd = Y.shape[0]
    def err(p):
        F = np.full(Y.shape, np.nan); F[di, si] = norm(p) * Tt[:, None]; E = F - Y
        if hourly: E = E.reshape(nd, 24, 2, -1).sum(2)
        E = E.reshape(nd, -1); ok = ~np.isnan(E); d_ = np.nanmean(np.abs(E), 1)
        return np.abs(E[ok]).mean(), np.sqrt((E[ok] ** 2).mean()), (d_[:ndays] if ndays else d_)
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); n = len(di); jr = (E_["jobs"] + E_["res"]) / (E_["jobs"] + E_["res"]).sum()
    v1 = 0.5 * norm(np.mean([np.load(f) for f in runs["eh"]], 0)) + 0.5 * norm(np.mean([np.load(f) for f in runs["eo"]], 0))
    g = norm(np.load(gx).astype(np.float64))
    rows = {"final method v1": v1, "XGBoost, scale-matched": g, "Jobs + residents": norm(np.repeat(jr[None], n, 0)), **rules(v1, g, jr)}
    res = {k: err(p) for k, p in rows.items()}
    print(f"\n===== {tag} ({city}, {Y.shape[2]} regions, {n} calendar slots{', hourly' if hourly else ''}) =====")
    for k, (m, r_, _) in res.items(): print(f"   {k:46s} MAE {m:.3f}  RMSE {r_:.3f}")
    for k in rows:
        if not k.startswith("R"): continue
        diff = res["final method v1"][2] - res[k][2]; b = boot(diff, rng)
        gm = 100 * (res["final method v1"][0] - res[k][0]) / res["final method v1"][0]; gr = 100 * (res["final method v1"][1] - res[k][1]) / res["final method v1"][1]
        print(f"   v1 - {k[:2]}: daily MAE difference {diff.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]; MAE change {gm:+.2f}%, RMSE change {gr:+.2f}% (positive = rule better)")
        summary.setdefault(k, {})[tag] = (gm, gr)
print("\n===== adoption rule of PROTOCOL_V12 (as PROTOCOL_V10): mean MAE change > 0, mean RMSE change > 0, better in MAE in at least 3 of 5 and in RMSE in at least 3 of 5 =====")
for k, v in summary.items():
    if len(v) < 5: print(f"   {k}: not all five targets scored"); continue
    m = np.mean([x[0] for x in v.values()]); r = np.mean([x[1] for x in v.values()]); cm = sum(x[0] > 0 for x in v.values()); cr = sum(x[1] > 0 for x in v.values())
    print(f"   {k:46s} mean MAE {m:+.2f}%, mean RMSE {r:+.2f}%, better in MAE in {cm}/5 and in RMSE in {cr}/5 -> {'PASSES' if m > 0 and r > 0 and cm >= 3 and cr >= 3 else 'does not pass'}")
