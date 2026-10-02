"""diagnostic tests 1, 2, 7, 8 (2026-10-01). Tests 1, 2, 8 use target taxi data for DIAGNOSIS ONLY; nothing is fed back
into the model. Test 7 uses each city's yearly total only as the held-out label in leave-one-city-out."""
import json, numpy as np
from scipy.stats import spearmanr
from datetime import date, timedelta
B = "../bench77/data"; M = "../multicity/data"
TG = {"chicago": ("chicago_2021_ca_30min.npy", "nyc+sf"), "nyc": ("nyc_2021_zone_30min.npy", "chicago+sf")}
def load(t):
    Y = np.load(f"{B}/{TG[t][0]}").astype(np.float64); tot = Y.sum(2)
    idx = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Sh = Y[di, si] / tot[di, si][:, None]
    p = np.mean([np.load(f"out/multi_{t}_from_{TG[t][1]}_veh_seed{s}.npy") for s in range(10)], 0); p /= p.sum(1, keepdims=True)
    return Y, Sh, p
print("#" * 30, "TEST 1: squeezed (too flat) or wrong order?  [diagnosis, uses target data]")
for t in TG:
    Y, Sh, p = load(t); R = Sh.shape[1]
    b, a = np.polyfit(p.ravel(), Sh.ravel(), 1); r2 = np.corrcoef(p.ravel(), Sh.ravel())[0, 1] ** 2
    top = p >= np.quantile(p, 0.9, axis=1, keepdims=True); bt = np.polyfit(p[top], Sh[top], 1)[0]
    base = np.abs(p - Sh).mean()
    best = (1.0, base)                                             # oracle sharpening p^g (g chosen ON TARGET -> diagnosis only)
    for g in np.arange(1.0, 3.01, 0.1):
        q = p ** g; q /= q.sum(1, keepdims=True); e = np.abs(q - Sh).mean()
        if e < best[1]: best = (g, e)
    srt = np.sort(Sh, 1); rk = np.argsort(np.argsort(-p, 1), 1); rankonly = np.take_along_axis(srt[:, ::-1], rk, 1)  # true sizes, predicted order
    print(f"{t:8s}: slope b={b:.2f} (top decile {bt:.2f}), R2={r2:.2f} | share error: now {base:.5f}; best oracle sharpening "
          f"p^{best[0]:.1f} -> {best[1]:.5f} ({100*(best[1]/base-1):+.0f}%); right sizes in OUR order -> {np.abs(rankonly-Sh).mean():.5f} "
          f"({100*(np.abs(rankonly-Sh).mean()/base-1):+.0f}%)")
print("\n" + "#" * 30, "TEST 2: error by area type (types from public data only)  [diagnosis]")
for t in TG:
    Y, Sh, p = load(t); S = np.load(f"{B}/region_static_{t}.npz"); E = np.load(f"data/extra_{t}.npz"); enpl = np.load(f"data/enpl_{t}.npy")
    area = S["area_km2"]; jd = E["jobs"] / area; rd = E["res"] / area; night = S["X"][:, 0] + S["X"][:, 6]
    typ = np.array(["other"] * len(area), dtype=object)
    typ[(rd >= np.quantile(rd, 0.9)) & (jd <= np.median(jd))] = "residential"
    typ[night >= np.quantile(night, 0.9)] = "nightlife"; typ[E["rail"] > 0] = "rail station"
    typ[jd >= np.quantile(jd, 0.9)] = "CBD (top 10% jobs)"; typ[enpl > 0] = "airport"
    ms, mp = Sh.mean(0), p.mean(0); ae = np.abs(p - Sh); se = (p - Sh) ** 2
    print(f"--- {t.upper()} ({len(area)} areas): type | #areas | true share | predicted share | % of abs error | % of squared error")
    for k in ["airport", "CBD (top 10% jobs)", "rail station", "nightlife", "residential", "other"]:
        i = typ == k
        if i.any(): print(f"   {k:20s} {i.sum():4d} | {100*ms[i].sum():5.1f}% | {100*mp[i].sum():5.1f}% | {100*ae[:, i].sum()/ae.sum():5.1f}% | {100*se[:, i].sum()/se.sum():5.1f}%")
print("\n" + "#" * 30, "TEST 7: which unit predicts a city's yearly trips? leave-one-city-out  [yearly totals as held-out labels]")
acs = json.load(open("raw/acs_city.json"))["data"]; acsA = json.load(open("raw/acs_austin.json"))["data"]
geo = {"chicago": acs["16000US1714000"], "nyc": acs["16000US3651000"], "sf": acs["16000US0667000"], "dc": acs["16000US1150000"], "austin": acsA["16000US4805000"]}
trips = {"chicago": np.load(f"{B}/chicago_2021_ca_30min.npy").sum(), "nyc": np.load(f"{B}/nyc_2021_zone_30min.npy").sum(),
         "sf": np.load(f"{M}/sf_2023_tract_30min.npy").sum(), "dc": np.load(f"{M}/dc_2021_tract_30min.npy").sum()}
U = {}
for c in list(trips) + ["austin"]:
    g = geo[c]; e = np.load(f"data/extra_{c}.npz")
    U[c] = {"jobs": e["jobs"].sum(), "residents": e["res"].sum(), "population": g["B01003"]["estimate"]["B01003001"],
            "taxi commuters": g["B08301"]["estimate"]["B08301016"], "transit commuters": g["B08301"]["estimate"]["B08301010"],
            "car-free households": g["B25044"]["estimate"]["B25044003"] + g["B25044"]["estimate"]["B25044010"]}
for cities, label in [(["chicago", "nyc", "sf", "dc"], "4 taxi cities"), (["chicago", "nyc", "sf"], "3 cities without DC")]:
    print(f"--- {label}: held-out city error of the predicted yearly total (geometric-mean trips per unit of the others)")
    for u in U["chicago"]:
        errs = []
        for h in cities:
            rate = np.exp(np.mean([np.log(trips[c] / U[c][u]) for c in cities if c != h])); errs.append(100 * (rate * U[h][u] / trips[h] - 1))
        spread = max(trips[c] / U[c][u] for c in cities) / min(trips[c] / U[c][u] for c in cities)
        print(f"   {u:20s} spread {spread:5.2f}x | " + " ".join(f"{h} {e:+5.0f}%" for h, e in zip(cities, errs)) + f" | mean |err| {np.mean(np.abs(errs)):4.0f}%")
print("   trips per car-free household:", ", ".join(f"{c} {trips[c]/U[c]['car-free households']:.1f}" for c in trips), "(Austin ride-hailing, 8 months:",
      f"{np.load(f'{M}/austin_2016_cell_30min.npy').sum()/U['austin']['car-free households']:.1f})")
print("\n" + "#" * 30, "TEST 8: are the five trip datasets comparable?  [diagnosis]")
DATA = {"chicago": (f"{B}/chicago_2021_ca_30min.npy", date(2021, 1, 1), 76), "nyc": (f"{B}/nyc_2021_zone_30min.npy", date(2021, 1, 1), 132),
        "sf": (f"{M}/sf_2023_tract_30min.npy", date(2023, 1, 1), None), "dc": (f"{M}/dc_2021_tract_30min.npy", date(2021, 1, 1), None),
        "austin": (f"{M}/austin_2016_cell_30min.npy", date(2016, 6, 4), None)}
for c, (f, st, air) in DATA.items():
    Y = np.load(f).astype(np.float64); day = Y.sum((1, 2)); med = np.median(day[day > 0])
    half = Y.sum((0, 2)); odd = half[1::2].sum() / half.sum()                 # share in the :30-:59 half-hours
    top1 = np.sort(Y.sum((0, 1)))[::-1]; topshare = top1[: max(1, len(top1) // 100)].sum() / top1.sum()
    print(f"{c:8s} days {len(day)}, days with <20% of median volume {int((day < 0.2*med).sum())}, empty days {int((day == 0).sum())} | "
          f"share in :30-:59 half-hours {100*odd:.1f}% | night share (0-5h) {100*half[:10].sum()/half.sum():.1f}% | "
          f"top-1% regions hold {100*topshare:.1f}% | regions {Y.shape[2]}")
