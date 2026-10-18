"""EVALUATION of Austin as a new target (PROTOCOL_MORE_CITIES.md, Austin addendum; written before any Austin run).
Level and rhythm exactly as predict_final.py, except the span (248 full days, 2016-06-04 to 2017-02-06) and the yearly
level prorated by 248/365. Methods: uniform, jobs, jobs + residents, base joint allocator, Chi et al. allocation, final
method with native helpers (N) and with scale-matched helpers (M). Austin is ride-hailing (RideAustin), not taxi.
The Austin demand file is read here only. Usage (from improve/): python austin_eval.py"""
import os, json, numpy as np
from datetime import date, timedelta
from slots import calendar_slots, END
HERE = os.path.dirname(os.path.abspath(__file__)); B, MC = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data"
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy",
      "sf": f"{MC}/sf_2023_tract_30min.npy", "dc": f"{MC}/dc_2021_tract_30min.npy"}
START = {"chicago": date(2021, 1, 1), "nyc": date(2021, 1, 1), "sf": date(2023, 1, 1), "dc": date(2021, 1, 1), "austin": date(2016, 6, 4)}
G = {"chicago": "16000US1714000", "nyc": "16000US3651000", "sf": "16000US0667000", "dc": "16000US1150000", "austin": "16000US4805000"}
SHARE = ["chicago", "nyc", "sf"]
acs = json.load(open(f"{HERE}/raw/acs_city.json"))["data"]; acs.update(json.load(open(f"{HERE}/raw/acs_austin.json"))["data"])
U = {}
for c, g in G.items():
    a = acs[g]; e = np.load(f"{HERE}/data/extra_{c}.npz")
    U[c] = {"jobs": e["jobs"].sum(), "residents": e["res"].sum(), "population": a["B01003"]["estimate"]["B01003001"],
            "taxi commuters": a["B08301"]["estimate"]["B08301016"], "transit commuters": a["B08301"]["estimate"]["B08301010"],
            "car-free households": a["B25044"]["estimate"]["B25044003"] + a["B25044"]["estimate"]["B25044010"]}


def days(c):
    n = (END[c] - START[c]).days + 1 if c in END else (date(START[c].year + 1, 1, 1) - START[c]).days
    return [START[c] + timedelta(days=d) for d in range(n)]


T = "austin"; helpers = list(YP)
Yh = {c: np.load(YP[c]).astype(np.float64) for c in helpers}; N = {c: Yh[c].sum() for c in helpers}
DOW = {c: np.array([d.weekday() for d in days(c)]) for c in helpers + [T]}
loo = {u: np.mean([abs(np.exp(np.mean([np.log(N[c] / U[c][u]) for c in helpers if c != h])) * U[h][u] / N[h] - 1) for h in helpers]) for u in U[T]}
unit = min(loo, key=loo.get); lev_year = U[T][unit] * np.exp(np.mean([np.log(N[c] / U[c][unit]) for c in helpers]))
ndays = len(days(T)); lev = lev_year * ndays / 365
sl = calendar_slots(T, START[T]); di, si = np.array([a for a, _ in sl]), np.array([b for _, b in sl])
def profile(c):
    tot = Yh[c].sum(2); r = np.array([tot[DOW[c] == k].sum(0) for k in range(7)]); return r / r.sum()
rh = np.mean([profile(c) for c in SHARE], 0); nd = np.bincount(DOW[T], minlength=7); w = DOW[T][di]
Tt = lev * rh[w, si] / nd[w]; Tt = Tt / Tt.sum() * lev                     # no month factor: no helper from 2016

Y = np.load(f"{MC}/austin_2016_cell_30min.npy").astype(np.float64)        # evaluation only
true_T = Y[di, si].sum(1)
norm = lambda p: p / p.sum(1, keepdims=True)
def err(p, tot):
    F = np.full(Y.shape, np.nan); F[di, si] = norm(p) * tot[:, None]; E = F - Y; ok = ~np.isnan(E)
    return np.abs(E[ok]).mean(), np.sqrt((E[ok] ** 2).mean()), np.nanmean(np.abs(E).reshape(Y.shape[0], -1), 1)[:ndays]
E_ = np.load(f"{HERE}/data/extra_{T}.npz"); R = len(E_["jobs"]); n = len(di); jr = E_["jobs"] + E_["res"]
ld = lambda fs: np.mean([norm(np.load(f)) for f in fs], 0)
O = f"{HERE}/out_cal"
M_ = {"Uniform": np.full((n, R), 1 / R), "Jobs-proportional": np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0),
      "Jobs + residents": np.repeat((jr / jr.sum())[None], n, 0),
      "Base allocator": ld([f"{HERE}/../multicity/out_cal/multi_austin_from_chicago+nyc+sf_seed{s}.npy" for s in range(10)]),
      "Chi et al. allocation": ld([f"{O}/chi_austin_seed{s}.npy" for s in range(10)])}
for cond, name in (("N", lambda r: "chicago+nyc+sf", ), ("M", lambda r: f"chicago+nyc+sf_for_austin_p{r}")):
    seed = (lambda r: r - 1)
    fs = {h: [f"{O}/multi_austin_from_{name(r)}_v{h}_seed{seed(r)}.npy" for r in range(1, 11)] for h in ("eh", "eo")}
    M_[f"Final, {'native helpers (N)' if cond == 'N' else 'scale-matched (M)'}"] = 0.5 * ld(fs["eh"]) + 0.5 * ld(fs["eo"])

print(f"===== AUSTIN (RideAustin ride-hailing, not taxi): {R} regions, {ndays} full days, {n} slots =====")
print(f"   level unit '{unit}', predicted total over the span {lev:,.0f} vs observed {true_T.sum():,.0f} ({100*(lev/true_T.sum()-1):+.0f}%)")
print(f"   {'method':36s} fully strict MAE / RMSE | allocation only (true slot totals) MAE / RMSE")
res = {}
for k, p in M_.items():
    s, a = err(p, Tt), err(p, true_T); res[k] = s
    print(f"   {k:36s} {s[0]:.3f} / {s[1]:.3f}        | {a[0]:.3f} / {a[1]:.3f}")
rng = np.random.default_rng(0); d = res["Final, native helpers (N)"][2] - res["Final, scale-matched (M)"][2]; nb = len(d) // 7
b = np.array([d[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(d) - 7, nb)])].mean() for _ in range(2000)])
print(f"   N minus M: mean daily MAE difference {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], M better on {100*(d>0).mean():.0f}% of days")
