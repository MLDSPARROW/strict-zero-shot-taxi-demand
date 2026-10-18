"""EVALUATION of the Chi, Xu and Chen (2026) allocation baseline (PROTOCOL_CHI_BASELINE.md). Same city totals per slot
as predict_final.py, same error definition as evaluate_strict.py (DC hourly). The uniform / jobs / jobs + residents
baselines are recomputed here as a check: on the native targets they must equal the archived results/eval_*.txt values,
which shows the totals are reproduced exactly, so the Chi numbers are comparable with the archived final method.
Usage (from improve/): python chi_eval.py"""
import os, numpy as np
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coarse_eval.py")).read()
ns = {"__file__": __file__}; exec(src[:src.index("def boot")], ns)
totals, HERE, MC, YP, SHARE = ns["totals"], ns["HERE"], ns["MC"], ns["YP"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})
TARGETS = {"chicago": ("chicago", YP["chicago"]), "nyc": ("nyc", YP["nyc"]), "sf": ("sf", YP["sf"]), "dc": ("dc", YP["dc"]),
           "sf_zcta": ("sf", f"{MC}/sf_zcta_30min.npy"), "dc_cluster": ("dc", f"{MC}/dc_cluster_30min.npy")}
norm = lambda p: p / p.sum(1, keepdims=True)
for tag, (city, yfile) in TARGETS.items():
    di, si, Tt, unit, lev = totals(city, tag)
    Y = np.load(yfile).astype(np.float64); nd = Y.shape[0]; hourly = city == "dc"
    def err(p):
        F = np.full(Y.shape, np.nan); F[di, si] = norm(p) * Tt[:, None]; E = F - Y
        if hourly: E = E.reshape(nd, 24, 2, -1).sum(2)
        E = E.reshape(nd, -1); ok = ~np.isnan(E); return np.abs(E[ok]).mean(), np.sqrt((E[ok] ** 2).mean())
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); R = len(E_["jobs"]); n = len(di); jr = E_["jobs"] + E_["res"]
    rows = {"Uniform": np.full((n, R), 1 / R), "Jobs-proportional": np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0),
            "Jobs + residents": np.repeat((jr / jr.sum())[None], n, 0)}
    runs = [np.load(f"{HERE}/out_cal/chi_{tag}_seed{s}.npy") for s in range(10)]
    rows["Chi et al. allocation (10-run ens.)"] = np.mean([norm(r) for r in runs], 0)
    single = np.array([err(r) for r in runs])
    print(f"\n===== {tag} ({'hourly' if hourly else '30-min'}, {R} regions) =====")
    for k, p in rows.items():
        m, r_ = err(p); print(f"   {k:36s} MAE {m:.2f}  RMSE {r_:.2f}")
    print(f"   Chi single runs: MAE {single[:,0].mean():.2f} +- {single[:,0].std(ddof=1):.2f}, RMSE {single[:,1].mean():.2f} +- {single[:,1].std(ddof=1):.2f}")
