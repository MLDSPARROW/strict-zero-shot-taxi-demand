"""Generic FULLY STRICT scoring for any of the four targets (PROTOCOL_HELDOUT.md rules). Validated by reproducing the paper's
Chicago and NYC numbers. Usage: python score_all.py chicago nyc sf dc"""
import os, sys, json, numpy as np
from datetime import date, timedelta
HERE = os.path.dirname(os.path.abspath(__file__)); B, M = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data"
YP = {"chicago": (f"{B}/chicago_2021_ca_30min.npy", f"{B}/region_static_chicago.npz", date(2021, 1, 1)),
      "nyc": (f"{B}/nyc_2021_zone_30min.npy", f"{B}/region_static_nyc.npz", date(2021, 1, 1)),
      "sf": (f"{M}/sf_2023_tract_30min.npy", f"{M}/region_static_sf.npz", date(2023, 1, 1)),
      "dc": (f"{M}/dc_2021_tract_30min.npy", f"{M}/region_static_dc.npz", date(2021, 1, 1))}
SHARE = {"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"], "sf": ["chicago", "nyc"], "dc": ["chicago", "nyc", "sf"]}
MODEL_H = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}",
           "sf": lambda r: "chicago+nyc", "dc": lambda r: f"chicago+nyc+sf_for_dc_p{r}"}
Y = {c: np.load(v[0]).astype(np.float64) for c, v in YP.items()}
DAYS = {c: [YP[c][2] + timedelta(days=d) for d in range(Y[c].shape[0])] for c in YP}
DOW = {c: np.array([d.weekday() for d in DAYS[c]]) for c in YP}; MON = {c: np.array([d.month for d in DAYS[c]]) for c in YP}
acs = json.load(open(f"{HERE}/raw/acs_city.json"))["data"]
G = {"chicago": "16000US1714000", "nyc": "16000US3651000", "sf": "16000US0667000", "dc": "16000US1150000"}
U = {}
for c, g in G.items():
    a = acs[g]; e = np.load(f"{HERE}/data/extra_{c}.npz")
    U[c] = {"jobs": e["jobs"].sum(), "residents": e["res"].sum(), "population": a["B01003"]["estimate"]["B01003001"],
            "taxi commuters": a["B08301"]["estimate"]["B08301016"], "transit commuters": a["B08301"]["estimate"]["B08301010"],
            "car-free households": a["B25044"]["estimate"]["B25044003"] + a["B25044"]["estimate"]["B25044010"]}
N = {c: Y[c].sum() for c in YP}
def level(T):
    H = [c for c in YP if c != T]
    loo = {u: np.mean([abs(np.exp(np.mean([np.log(N[c] / U[c][u]) for c in H if c != h])) * U[h][u] / N[h] - 1) for h in H]) for u in U[T]}
    u = min(loo, key=loo.get); return u, U[T][u] * np.exp(np.mean([np.log(N[c] / U[c][u]) for c in H]))
def profile(c):
    tot = Y[c].sum(2); r = np.array([tot[DOW[c] == k].sum(0) for k in range(7)]); return r / r.sum()
def strict_total(T, di, si):
    u, lev = level(T); S = SHARE[T]; rh = np.mean([profile(c) for c in S], 0); nd = np.bincount(DOW[T], minlength=7); w = DOW[T][di]
    Tt = lev * rh[w, si] / nd[w]
    same = [c for c in S if YP[c][2].year == YP[T][2].year]
    if same:
        mf = []
        for c in same:
            tc = Y[c].sum((1, 2)); m = np.array([tc[MON[c] == k].mean() for k in range(1, 13)]); mf.append(m / m.mean())
        mf = np.mean(mf, 0); Tt = Tt * mf[MON[T][di] - 1] / mf[MON[T] - 1].mean()
    return u, lev, Tt / Tt.sum() * lev
def ens(fs): p = np.mean([np.load(f) for f in fs], 0); return p / p.sum(1, keepdims=True)
def run(T):
    tot = Y[T].sum(2); idx = [(a, b) for a in range(Y[T].shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Y[T][di, si]; R = Yc.shape[1]
    u, lev, Tt = strict_total(T, di, si)
    print(f"\n===== TARGET {T.upper()}: level unit '{u}' (helper-only LOO), predicted yearly total {lev:,.0f} vs true {N[T]:,.0f} ({100*(lev/N[T]-1):+.1f}%); "
          f"per-slot total error {100*np.abs(Tt-tot[di,si]).mean()/tot[di,si].mean():.1f}%")
    E_ = np.load(f"{HERE}/data/extra_{T}.npz"); n = len(di); rows = []
    rows.append(("Uniform", np.full((n, R), 1 / R)))
    rows.append(("Jobs-proportional", np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0)))
    jr = E_["jobs"] + E_["res"]; rows.append(("Jobs + residents", np.repeat((jr / jr.sum())[None], n, 0)))
    def add(name, fs):
        if all(os.path.exists(f) for f in fs): rows.append((name, ens(fs)))
        else: print(f"   (missing: {name})")
    add("Base joint allocator", [f"{HERE}/../multicity/multi_{T}_from_{'+'.join(SHARE[T])}_seed{s}.npy" for s in range(10)])
    heads = {}
    for h in ("eh", "eo"):
        fs = [f"{HERE}/out/multi_{T}_from_{MODEL_H[T](r)}_v{h}_seed{r-1}.npy" for r in range(1, 11)]
        if all(os.path.exists(f) for f in fs): heads[h] = ens(fs); rows.append((f"Head {h}", heads[h]))
    if len(heads) == 2: rows.append(("FINAL (average of heads)", 0.5 * heads["eh"] + 0.5 * heads["eo"]))
    hourly = T == "dc"
    print(f"   {'method':42s} MAE    RMSE" + ("   | hourly MAE  RMSE (primary for DC)" if hourly else ""))
    for name, p in rows:
        P = p * Tt[:, None]; E = P - Yc; line = f"   {name:42s} {np.abs(E).mean():5.2f}  {np.sqrt((E**2).mean()):6.2f}"
        if hourly:
            F = np.zeros_like(Y[T]); F[di, si] = P; Eh = (F.reshape(F.shape[0], 24, 2, -1).sum(2) - Y[T].reshape(F.shape[0], 24, 2, -1).sum(2))
            line += f"   | {np.abs(Eh).mean():6.2f}  {np.sqrt((Eh**2).mean()):6.2f}"
        print(line, flush=True)
for T in sys.argv[1:]: run(T)
