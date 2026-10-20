"""PROTOCOL_V2.md: final evaluation of v1 and v2 on every target, run once after the v2 calendar-only predictions exist.
City totals and slots exactly as predict_final.py / coarse_eval.py / austin_eval.py (fully strict). v1 = eh/2 + eo/2,
v2 = eh/2 + eo/4 + ep/4 (10-run ensembles, condition M). Scores: mean-forecast MAE and RMSE (as v1), MAE of the NB median
(I1, dispersion from the target's share helpers, condition M partition 1), 80% interval coverage. DC and dc_cluster hourly.
Baselines: zero forecast, uniform, jobs, jobs + residents (v1 files and v2 re-interpolated covariates), Chi et al.
Paired 7-day block bootstrap (2,000 draws) of daily differences. Target demand is read for scoring only.
Usage (from improve/): python v2_eval.py [targets, default all]"""
import os, sys, io, contextlib, numpy as np
from datetime import date
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from point_functional import dispersion, Quantiles
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, MC, YP, boot, SHARE = ns["totals"], ns["MC"], ns["YP"], ns["boot"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})                  # as predict_final.py (checked in chi_eval.py)
B, AG, O = f"{HERE}/../bench77/data", f"{HERE}/data/agg", f"{HERE}/out_cal"
norm = lambda p: p / p.sum(1, keepdims=True)
TG = {"chicago": ("chicago", YP["chicago"], lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", ["nyc_for_chicago_p1", "sf_for_chicago_p1"]),
      "nyc": ("nyc", YP["nyc"], lambda r: f"chicago+sf_for_nyc_p{r}", ["chicago", "sf_for_nyc_p1"]),
      "sf": ("sf", YP["sf"], lambda r: "chicago+nyc", ["chicago", "nyc"]),
      "dc": ("dc", YP["dc"], lambda r: f"chicago+nyc+sf_for_dc_p{r}", ["chicago", "nyc", "sf_for_dc_p1"]),
      "sf_zcta": ("sf", f"{MC}/sf_zcta_30min.npy", lambda r: f"chicago+nyc_for_sf_zcta_p{r}", ["chicago", "nyc_for_sf_zcta_p1"]),
      "dc_cluster": ("dc", f"{MC}/dc_cluster_30min.npy", lambda r: f"chicago+nyc_for_dc_cluster_p{r}+sf_for_dc_cluster_p{r}",
                     ["chicago", "nyc_for_dc_cluster_p1", "sf_for_dc_cluster_p1"]),
      "austin": ("austin", None, lambda r: f"chicago+nyc+sf_for_austin_p{r}", ["chicago", "nyc", "sf_for_austin_p1"])}
GOAL = {"nyc": "MAE < 6.63 (zero) and < 6.83 (UrbanGPT); RMSE < 17.51 (jobs-proportional)", "sf": "MAE <= 0.40, RMSE < 0.98",
        "dc": "MAE < 0.66, RMSE < 2.40 (hourly)", "chicago": "not > 3% worse than 1.83 / 5.21"}


def helper(tag):
    if tag in ("chicago", "nyc"): return np.load(f"{B}/{tag}_2021_{'ca' if tag == 'chicago' else 'zone'}_30min.npy"), date(2021, 1, 1)
    if tag == "sf": return np.load(YP["sf"]), date(2023, 1, 1)
    return np.load(f"{AG}/{tag}_Y.npy"), (date(2023, 1, 1) if tag.startswith("sf_") else date(2021, 1, 1))


def target(tag):
    city, yf, mat, H = TG[tag]
    if tag == "austin":
        with contextlib.redirect_stdout(io.StringIO()):
            a = {"__file__": f"{HERE}/austin_eval.py"}; exec(open(f"{HERE}/austin_eval.py").read(), a)
        return a["Y"], a["di"], a["si"], a["Tt"], a["ndays"], False
    di, si, Tt, unit, lev = totals(city, tag); Y = np.load(yf).astype(np.float64)
    return Y, di, si, Tt, None, city == "dc"


def evaluate(tag):
    city, yf, mat, H = TG[tag]; Y, di, si, Tt, ndays, hourly = target(tag); n = len(di)
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); A = np.load(f"{HERE}/data/acs_{tag}.npz"); R = len(E_["jobs"])
    rep = lambda v: np.repeat((v / v.sum())[None], n, 0)
    ld = lambda h: np.mean([norm(np.load(f"{O}/multi_{tag}_from_{mat(r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0)
    Hd = {h: ld(h) for h in ("eh", "eo", "ep")}
    M = {"Zero forecast": None, "Uniform": np.full((n, R), 1 / R), "Jobs-proportional": rep(E_["jobs"]), "Jobs + residents": rep(E_["jobs"] + E_["res"]),
         "Jobs-proportional (v2 covariates)": rep(A["jobs_n"]), "Jobs + residents (v2 covariates)": rep(A["jobs_n"] + A["res_n"]),
         "Chi et al. allocation": np.mean([norm(np.load(f"{O}/chi_{tag}_seed{s}.npy")) for s in range(10)], 0),
         "v1 (eh/2 + eo/2)": Hd["eh"] / 2 + Hd["eo"] / 2, "v2 (eh/2 + eo/4 + ep/4)": Hd["eh"] / 2 + Hd["eo"] / 4 + Hd["ep"] / 4}
    k_of, tab = dispersion([helper(h) for h in H], hourly=hourly); Q = Quantiles(k_of, 2 * float(Tt.max()) + 1)
    Yv = Y.reshape(Y.shape[0], 24, 2, -1).sum(2) if hourly else Y
    print(f"\n===== {tag} ({'hourly' if hourly else '30-min'}, {R} regions, {n} calendar slots) =====")
    print(f"   {'method':36s} mean-forecast MAE / RMSE | median-forecast MAE | 80% coverage")
    D = {}
    for name, p in M.items():
        L = np.full(Y.shape, np.nan); L[di, si] = 0.0 if p is None else norm(p) * Tt[:, None]
        if hourly: L = L.reshape(Y.shape[0], 24, 2, -1).sum(2)
        ok = ~np.isnan(L); assert np.nanmax(L) <= Q.g[-1]; lam, y = L[ok], Yv[ok]; med = Q(lam, 0.5)
        Med = np.full(Yv.shape, np.nan); Med[ok] = med; Em = L - Yv
        f = lambda Z: (Z.reshape(Yv.shape[0], -1))[:ndays] if ndays else Z.reshape(Yv.shape[0], -1)
        D[name] = {"mae": np.nanmean(np.abs(f(Em)), 1), "mse": np.nanmean(f(Em) ** 2, 1), "med": np.nanmean(np.abs(f(Med - Yv)), 1)}
        cov = np.mean((y >= Q(lam, 0.1)) & (y <= Q(lam, 0.9)))
        print(f"   {name:36s} {np.abs(lam - y).mean():.3f} / {np.sqrt(((lam - y) ** 2).mean()):.3f}           | {np.abs(med - y).mean():.3f}"
              f"               | {cov:.3f}")
    rng = np.random.default_rng(0)
    base = [k for k in M if not k.startswith(("v1", "v2", "Zero"))]
    best = {m: min(base, key=lambda k: np.nanmean(D[k][m])) for m in ("med", "mse")}
    for ref in ["v1 (eh/2 + eo/2)", best["med"], "Zero forecast"]:
        for m, lab in (("med", "median-forecast MAE"), ("mae", "mean-forecast MAE"), ("mse", "MSE")):
            if ref == "Zero forecast" and m != "med": continue
            d = D[ref][m] - D["v2 (eh/2 + eo/4 + ep/4)"][m]; b = boot(d, rng)
            print(f"   {ref} minus v2, daily {lab}: {np.nanmean(d):+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], v2 better on {100 * np.mean(d > 0):.0f}% of days")
    if best["mse"] != best["med"]:
        d = D[best["mse"]]["mse"] - D["v2 (eh/2 + eo/4 + ep/4)"]["mse"]; b = boot(d, rng)
        print(f"   {best['mse']} (best RMSE baseline) minus v2, daily MSE: {np.nanmean(d):+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]")
    if tag in GOAL: print(f"   success criterion fixed in the protocol: {GOAL[tag]}")


for t in (sys.argv[1:] or list(TG)): evaluate(t)
print("\ndone")
