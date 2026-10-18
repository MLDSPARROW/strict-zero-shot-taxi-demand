"""EVALUATION of PROTOCOL_COARSE_TARGETS.md (written before any result). Fully strict trip errors on the calendar slots.
City totals per slot (level x rhythm) are computed exactly as in predict_final.py from helper cities and public data only;
they are the same for a city's native and coarse geography. Shares: N (native helpers, seeds 0-9) and M (scale-matched,
run r = partition r, seed r-1), final = equal average of the eh and eo 10-run ensembles. The target demand is read here only.
Usage (from improve/): python coarse_eval.py"""
import os, json, numpy as np
from datetime import date, timedelta
from slots import calendar_slots
HERE = os.path.dirname(os.path.abspath(__file__)); B, MC = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data"
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy",
      "sf": f"{MC}/sf_2023_tract_30min.npy", "dc": f"{MC}/dc_2021_tract_30min.npy"}
START = {"chicago": date(2021, 1, 1), "nyc": date(2021, 1, 1), "sf": date(2023, 1, 1), "dc": date(2021, 1, 1)}
SHARE = {"sf": ["chicago", "nyc"], "dc": ["chicago", "nyc", "sf"]}
G = {"chicago": "16000US1714000", "nyc": "16000US3651000", "sf": "16000US0667000", "dc": "16000US1150000"}
TARGETS = {"sf_zcta": ("sf", f"{MC}/sf_zcta_30min.npy", "chicago+nyc", lambda r: f"chicago+nyc_for_sf_zcta_p{r}"),
           "dc_cluster": ("dc", f"{MC}/dc_cluster_30min.npy", "chicago+nyc+sf", lambda r: f"chicago+nyc_for_dc_cluster_p{r}+sf_for_dc_cluster_p{r}"),
           "dc": ("dc", f"{MC}/dc_2021_tract_30min.npy", "chicago+nyc+sf", lambda r: f"chicago+nyc+sf_for_dc_p{r}")}
acs = json.load(open(f"{HERE}/raw/acs_city.json"))["data"]
U = {}
for c, g in G.items():
    a = acs[g]; e = np.load(f"{HERE}/data/extra_{c}.npz")
    U[c] = {"jobs": e["jobs"].sum(), "residents": e["res"].sum(), "population": a["B01003"]["estimate"]["B01003001"],
            "taxi commuters": a["B08301"]["estimate"]["B08301016"], "transit commuters": a["B08301"]["estimate"]["B08301010"],
            "car-free households": a["B25044"]["estimate"]["B25044003"] + a["B25044"]["estimate"]["B25044010"]}


def days(c):
    n = (date(START[c].year + 1, 1, 1) - START[c]).days; return [START[c] + timedelta(days=d) for d in range(n)]


def totals(T, tag):                                   # identical to predict_final.py, level and rhythm only
    helpers = [c for c in YP if c != T]
    Y = {c: np.load(YP[c]).astype(np.float64) for c in helpers}; N = {c: Y[c].sum() for c in helpers}
    DOW = {c: np.array([d.weekday() for d in days(c)]) for c in YP}; MON = {c: np.array([d.month for d in days(c)]) for c in YP}
    loo = {u: np.mean([abs(np.exp(np.mean([np.log(N[c] / U[c][u]) for c in helpers if c != h])) * U[h][u] / N[h] - 1) for h in helpers]) for u in U[T]}
    unit = min(loo, key=loo.get); lev = U[T][unit] * np.exp(np.mean([np.log(N[c] / U[c][unit]) for c in helpers]))
    sl = calendar_slots(tag, START[T]); di, si = np.array([a for a, _ in sl]), np.array([b for _, b in sl])
    def profile(c):
        tot = Y[c].sum(2); r = np.array([tot[DOW[c] == k].sum(0) for k in range(7)]); return r / r.sum()
    S = SHARE[T]; rh = np.mean([profile(c) for c in S], 0); nd = np.bincount(DOW[T], minlength=7); w = DOW[T][di]
    Tt = lev * rh[w, si] / nd[w]
    same = [c for c in S if START[c].year == START[T].year]
    if same:
        mf = np.mean([(lambda tc: np.array([tc[MON[c] == k].mean() for k in range(1, 13)]))(Y[c].sum((1, 2))) for c in same], 0); mf /= mf.mean()
        Tt = Tt * mf[MON[T][di] - 1] / mf[MON[T] - 1].mean()
    return di, si, Tt / Tt.sum() * lev, unit, lev


def boot(x, rng):
    nb = len(x) // 7
    return np.array([x[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(x) - 7, nb)])].mean() for _ in range(2000)])


norm = lambda p: p / p.sum(1, keepdims=True)
summary = {}
for tag, (city, yfile, nat, mat) in TARGETS.items():
    di, si, Tt, unit, lev = totals(city, tag)
    Yfull = np.load(yfile).astype(np.float64); nd = Yfull.shape[0]          # evaluation only
    hourly = city == "dc"                         # DC raw data are hourly (30-min values are equal splits): scored hourly, as in evaluate_strict.py
    def err(p):
        F = np.full(Yfull.shape, np.nan); F[di, si] = norm(p) * Tt[:, None]; E = F - Yfull
        if hourly: E = E.reshape(nd, 24, 2, -1).sum(2)
        E = E.reshape(nd, -1); ok = ~np.isnan(E)
        return np.abs(E[ok]).mean(), np.sqrt((E[ok] ** 2).mean()), np.nanmean(np.abs(E), 1)
    def load(srcs, h, runs):
        return [np.load(f"{HERE}/out_cal/multi_{tag}_from_{s}_v{h}_seed{sd}.npy") for s, sd in runs]
    cond = {"N": lambda h: load(None, h, [(nat, s) for s in range(10)]),
            "M": lambda h: load(None, h, [(mat(r), r - 1) for r in range(1, 11)])}
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); R = len(E_["jobs"]); n = len(di); jr = E_["jobs"] + E_["res"]
    base = {"Uniform": np.full((n, R), 1 / R), "Jobs-proportional": np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0),
            "Jobs + residents": np.repeat((jr / jr.sum())[None], n, 0)}
    print(f"\n===== {tag} ({city}, {R} regions, {n} calendar slots; level unit '{unit}', yearly total {lev:,.0f}) =====")
    for k, p in base.items():
        m, r_, _ = err(p); print(f"   {k:22s} MAE {m:.3f}  RMSE {r_:.3f}")
    res = {}
    for c in ("N", "M"):
        F = {h: cond[c](h) for h in ("eh", "eo")}
        fin = 0.5 * norm(np.mean(F["eh"], 0)) + 0.5 * norm(np.mean(F["eo"], 0))
        single = np.array([err(0.5 * norm(a) + 0.5 * norm(b))[:2] for a, b in zip(F["eh"], F["eo"])])
        res[c] = err(fin)
        print(f"   {c} final {'(native helpers)' if c == 'N' else '(scale-matched)':18s} MAE {res[c][0]:.3f}  RMSE {res[c][1]:.3f} | "
              f"single runs MAE {single[:,0].mean():.3f} +- {single[:,0].std(ddof=1):.3f}")
    d = res["N"][2] - res["M"][2]; b = boot(d, np.random.default_rng(0)); gain = (res["N"][0] - res["M"][0]) / res["N"][0]
    print(f"   N minus M: mean daily MAE difference {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], "
          f"M better on {100*(d>0).mean():.0f}% of days; relative MAE gain (N-M)/N {100*gain:+.1f}%")
    summary[tag] = {"N": res["N"][:2], "M": res["M"][:2], "gain": gain, "ci": (np.percentile(b, 2.5), np.percentile(b, 97.5))}
print("\n===== READING RULE (fixed in the protocol) =====")
ok = all(summary[t]["ci"][0] > 0 for t in ("sf_zcta", "dc_cluster"))
print(f"   M better than N at both coarse targets with intervals excluding zero: {'YES' if ok else 'NO'}")
print(f"   relative gain dc_cluster {100*summary['dc_cluster']['gain']:+.1f}% vs native dc {100*summary['dc']['gain']:+.1f}%: "
      f"{'larger at coarse' if summary['dc_cluster']['gain'] > summary['dc']['gain'] else 'NOT larger at coarse'}")
