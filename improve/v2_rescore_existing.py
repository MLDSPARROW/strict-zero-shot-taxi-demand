"""PROTOCOL_V2.md, I1 applied to the predictions that already exist (v1 final N and M, Chi et al., simple baselines) for
Washington, DC (native, hourly), dc_cluster (hourly), sf_zcta and Austin. City totals, slots and mean forecasts are
exactly those of coarse_eval.py / austin_eval.py / chi_eval.py (their MAE and RMSE are reprinted as a reproduction check).
New columns: MAE of the NB median (dispersion from the target's share helpers, condition M partition 1) and coverage of
the central 80% predictive interval. Target demand is read for scoring only. Usage (from improve/): python v2_rescore_existing.py"""
import os, sys, io, contextlib, numpy as np
from datetime import date
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from point_functional import dispersion, Quantiles, check
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, MC, YP, boot = ns["totals"], ns["MC"], ns["YP"], ns["boot"]
B, AG, O = f"{HERE}/../bench77/data", f"{HERE}/data/agg", f"{HERE}/out_cal"
norm = lambda p: p / p.sum(1, keepdims=True)


def helper(tag):
    if tag in ("chicago", "nyc"): return np.load(f"{B}/{tag}_2021_{'ca' if tag == 'chicago' else 'zone'}_30min.npy"), date(2021, 1, 1)
    return np.load(f"{AG}/{tag}_Y.npy"), (date(2023, 1, 1) if tag.startswith("sf_") else date(2021, 1, 1))


def shares(tag, n, ours):
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); A = np.load(f"{HERE}/data/acs_{tag}.npz"); R = len(E_["jobs"])
    rep = lambda v: np.repeat((v / v.sum())[None], n, 0)
    ld = lambda fs: np.mean([norm(np.load(f)) for f in fs], 0)
    M = {"Zero forecast": None, "Uniform": np.full((n, R), 1 / R), "Jobs-proportional": rep(E_["jobs"]),
         "Jobs + residents": rep(E_["jobs"] + E_["res"]), "Jobs-proportional (v2 covariates)": rep(A["jobs_n"]),
         "Jobs + residents (v2 covariates)": rep(A["jobs_n"] + A["res_n"]),
         "Chi et al. allocation": ld([f"{O}/chi_{tag}_seed{s}.npy" for s in range(10)])}
    for c, files in ours.items(): M[c] = 0.5 * ld(files("eh")) + 0.5 * ld(files("eo"))
    return M


def report(tag, Y, di, si, Tt, M, helpers, hourly, ndays=None):
    k_of, tab = dispersion([helper(h) for h in helpers], hourly=hourly)
    Yv = Y.reshape(Y.shape[0], 24, 2, -1).sum(2) if hourly else Y
    print(f"\n===== {tag} ({'hourly' if hourly else '30-min'}, {Y.shape[2]} regions); dispersion from helpers {', '.join(helpers)} =====")
    print("   NB k by lambda bin: " + ", ".join(f"{l:.3g}:{k:.3g}" for l, _, k, _ in tab[:: max(1, len(tab) // 8)]))
    res, Q = {}, Quantiles(k_of, 2 * float(Tt.max()) + 1)           # no region-slot can exceed the (hourly) city total
    for name, p in M.items():
        L = np.full(Y.shape, np.nan); L[di, si] = 0.0 if p is None else norm(p) * Tt[:, None]
        if hourly: L = L.reshape(Y.shape[0], 24, 2, -1).sum(2)
        ok = ~np.isnan(L); assert np.nanmax(L) <= Q.g[-1]
        lam, y = L[ok], Yv[ok]; med = Q(lam, 0.5)
        Med = np.full(Yv.shape, np.nan); Med[ok] = med
        daily = np.nanmean(np.abs(Med - Yv).reshape(Yv.shape[0], -1), 1); daily = daily[:ndays] if ndays else daily
        res[name] = daily
        cov = np.mean((y >= Q(lam, 0.1)) & (y <= Q(lam, 0.9)))
        print(f"   {name:36s} mean-forecast MAE {np.abs(lam - y).mean():.3f} RMSE {np.sqrt(((lam - y) ** 2).mean()):.3f} | "
              f"median-forecast MAE {np.abs(med - y).mean():.3f} | 80% interval coverage {cov:.3f}")
        if name == "Uniform": print(f"   (grid median check: {100 * check(Q, k_of, lam):.3f}% of sampled cells differ from exact NB median)")
    rng = np.random.default_rng(0); ours = [k for k in res if k.startswith("Ours")]
    base = {k: v for k, v in res.items() if not k.startswith("Ours") and k != "Zero forecast"}
    best = min(base, key=lambda k: np.nanmean(base[k]))
    for o in ours:
        for ref in ("Zero forecast", best):
            d = res[ref] - res[o]; b = boot(d, rng)
            print(f"   {ref} minus {o} (median-forecast daily MAE): {np.nanmean(d):+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], {o} better on {100 * np.mean(d > 0):.0f}% of days")


T = {"dc": ("dc", YP["dc"], "chicago+nyc+sf", lambda r: f"chicago+nyc+sf_for_dc_p{r}", ["chicago", "nyc", "sf_for_dc_p1"]),
     "dc_cluster": ("dc", f"{MC}/dc_cluster_30min.npy", "chicago+nyc+sf", lambda r: f"chicago+nyc_for_dc_cluster_p{r}+sf_for_dc_cluster_p{r}",
                    ["chicago", "nyc_for_dc_cluster_p1", "sf_for_dc_cluster_p1"]),
     "sf_zcta": ("sf", f"{MC}/sf_zcta_30min.npy", "chicago+nyc", lambda r: f"chicago+nyc_for_sf_zcta_p{r}", ["chicago", "nyc_for_sf_zcta_p1"])}
for tag, (city, yf, nat, mat, H) in T.items():
    di, si, Tt, unit, lev = totals(city, tag); Y = np.load(yf).astype(np.float64)
    ours = {"Ours v1, native helpers (N)": lambda h: [f"{O}/multi_{tag}_from_{nat}_v{h}_seed{s}.npy" for s in range(10)],
            "Ours v1, scale-matched (M)": lambda h: [f"{O}/multi_{tag}_from_{mat(r)}_v{h}_seed{r - 1}.npy" for r in range(1, 11)]}
    report(tag, Y, di, si, Tt, shares(tag, len(di), ours), H, hourly=city == "dc")

with contextlib.redirect_stdout(io.StringIO()):                     # austin_eval.py builds the Austin totals and slots
    ans = {"__file__": f"{HERE}/austin_eval.py"}; exec(open(f"{HERE}/austin_eval.py").read(), ans)
oursA = {"Ours v1, native helpers (N)": lambda h: [f"{O}/multi_austin_from_chicago+nyc+sf_v{h}_seed{r - 1}.npy" for r in range(1, 11)],
         "Ours v1, scale-matched (M)": lambda h: [f"{O}/multi_austin_from_chicago+nyc+sf_for_austin_p{r}_v{h}_seed{r - 1}.npy" for r in range(1, 11)]}
MA = shares("austin", len(ans["di"]), oursA); MA["Base allocator"] = ans["M_"]["Base allocator"]
report("austin", ans["Y"], ans["di"], ans["si"], ans["Tt"], MA, ["chicago", "nyc", "sf_for_austin_p1"], hourly=False, ndays=ans["ndays"])
print("\ndone")
