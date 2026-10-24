"""DESCRIPTIVE check (helper data and development targets only): stability across taxi cities and scales of several cross-sectional laws
log(yearly taxi share) = a + beta * log(covariate share)  (extensive)   and   log(share / area) = a + beta * log(covariate / area)  (intensive),
airports left out, weights = covariate share. Covariates: LODES jobs, ACS population, jobs + population, ACS car-free households, rail stations
+ 1. For each law: the betas of the helper sets at the scale of each development target, their spread, and the beta of the TRUE shares and of
the shares PREDICTED by v1 = (eh + eo)/2 for the two development targets. Usage (from improve/): python v11_law2.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
B, MC, AG, D = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data", f"{HERE}/data/agg", f"{HERE}/data"
NAT = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy", "sf": f"{MC}/sf_2023_tract_30min.npy"}
STA = {"chicago": f"{B}/region_static_chicago.npz", "nyc": f"{B}/region_static_nyc.npz", "sf": f"{MC}/region_static_sf.npz"}


def cov(tag):
    e = np.load(f"{D}/extra_{tag}.npz"); a = np.load(f"{D}/acs_{tag}.npz")
    area = np.load(STA[tag] if tag in STA else f"{AG}/{tag}_static.npz")["area_km2"].astype(np.float64)
    j, p = e["jobs"].astype(np.float64), a["pop"].astype(np.float64)
    air = (np.load(f"{D}/enpl_{tag}.npy") > 0) if os.path.exists(f"{D}/enpl_{tag}.npy") else np.zeros(len(j), bool)
    return {"jobs": j, "population": p, "jobs+population": j + p, "car-free households": a["carfree"].astype(np.float64), "rail stations + 1": e["rail"].astype(np.float64) + 1}, area, air


def yearly(tag):
    Y = np.load(NAT[tag]) if tag in NAT else np.load(f"{AG}/{tag}_Y.npy"); s = Y.sum((0, 1)).astype(np.float64); return s / s.sum()


def law(s, c, area, air, intensive):
    m = (s > 0) & (c > 0) & ~air; w = c[m] / c[m].sum()
    x, y = (np.log(c[m] / area[m]), np.log(s[m] / area[m])) if intensive else (np.log(c[m] / c[m].sum()), np.log(s[m]))
    X = np.c_[np.ones(m.sum()), x]; b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]; return b[1]


SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
HEL = {"chicago": lambda r: [f"nyc_for_chicago_p{r}", f"sf_for_chicago_p{r}"], "nyc": lambda r: ["chicago", f"sf_for_nyc_p{r}"]}
norm = lambda q: q / q.sum(1, keepdims=True)
pred = {}
for T in ("chicago", "nyc"):
    Y = np.load(NAT[T]).astype(np.float64); tot = Y.sum(2); di, si = np.nonzero(tot > 0)
    P = np.mean([0.5 * norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_veh_seed{r - 1}.npy")) + 0.5 * norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_veo_seed{r - 1}.npy"))
                 for r in range(1, 11)], 0)
    sy = (P * tot[di, si][:, None]).sum(0); pred[T] = sy / sy.sum()
for intensive in (False, True):
    print(f"\n===== {'INTENSIVE: log(share/area) on log(covariate/area)' if intensive else 'EXTENSIVE: log share on log covariate share'} =====")
    print("   covariate             | helpers at Chicago scale (NYC, SF)  spread | helpers at NYC scale (Chicago, SF)  spread || Chicago true / v1 | NYC true / v1")
    for k in ("jobs", "population", "jobs+population", "car-free households", "rail stations + 1"):
        out = []
        for T in ("chicago", "nyc"):
            hb = {}
            for r in range(1, 11):
                for h in HEL[T](r):
                    c, area, air = cov(h); hb.setdefault(h.split("_p")[0], []).append(law(yearly(h), c[k], area, air, intensive))
            v = [float(np.mean(x)) for x in hb.values()]; out.append((v, max(v) - min(v)))
        tr = {}
        for T in ("chicago", "nyc"):
            c, area, air = cov(T); tr[T] = (law(yearly(T), c[k], area, air, intensive), law(pred[T], c[k], area, air, intensive))
        print(f"   {k:21s} | {out[0][0][0]:5.2f} {out[0][0][1]:5.2f}                      {out[0][1]:5.2f} | {out[1][0][0]:5.2f} {out[1][0][1]:5.2f}                       {out[1][1]:5.2f} || "
              f"{tr['chicago'][0]:5.2f} / {tr['chicago'][1]:5.2f}    | {tr['nyc'][0]:5.2f} / {tr['nyc'][1]:5.2f}")
