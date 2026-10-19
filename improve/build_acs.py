"""ACS taxi-propensity covariates per region (PROTOCOL_V2.md, I2, flag `a` and the population share of flag `p`).
NO taxi demand is used. Source: ACS 2020-2024 five-year tract tables via the Census Reporter API (the plain Census API now
needs a key): B01003 population, B25044 households by vehicles available, B19001 households by income.
Tract counts are spread to regions by tract-area overlap, exactly as LODES in build_extra.py. Coarse targets and the
re-aggregated helper blocks are exact sums of member regions; the membership is rebuilt with the original k-means seeds
(aggregate*.py, build_coarse_targets.py) and accepted only if it reproduces the stored jobs sums.
Raw API responses are cached in raw/acs_tract/, so a rerun downloads nothing.
ADDENDUM (PROTOCOL_V2.md, before any v2 run): where the regions cover the listed counties completely (NYC, SF, DC), each
tract's overlap weights are normalised to sum to 1, because the generalised cartographic tract outlines leave Manhattan
waterfront tracts up to 74% outside every taxi zone (NYC lost 4.0% of its population, 5.6% of its LODES jobs). Chicago
and Austin regions cover only part of their counties, so plain area weights are kept there. LODES jobs and resident
workers are re-interpolated the same way (jobs_n, res_n) for flag `p` and the v2 baselines; v1 files are not changed.
Output: data/acs_<city or tag>.npz with pop, hh, carfree, inc_hh, rich, jobs_n, res_n (counts per region).
Usage (from improve/): python build_acs.py"""
import os, re, json, glob, numpy as np, pandas as pd, geopandas as gpd, requests
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from sklearn.cluster import KMeans
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = f"{HERE}/raw"; OUT = f"{HERE}/data"
CACHE = f"{RAW}/acs_tract"; os.makedirs(CACHE, exist_ok=True)
B77, MC = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data"
RELEASE = "acs2024_5yr"
UA = {"User-Agent": "strict-zero-shot-taxi-demand research script (github.com/MLDSPARROW/strict-zero-shot-taxi-demand)"}
COUNTIES = {"chicago": ("17", ["031"]), "nyc": ("36", ["005", "047", "061", "081", "085"]), "sf": ("06", ["075"]),
            "dc": ("11", ["001"]), "austin": ("48", ["453", "491", "209"])}
PLACE = {"chicago": "16000US1714000", "nyc": "16000US3651000", "sf": "16000US0667000", "dc": "16000US1150000", "austin": "16000US4805000"}
SP = {"chicago": f"{B77}/region_static_chicago.npz", "nyc": f"{B77}/region_static_nyc.npz",
      "sf": f"{MC}/region_static_sf.npz", "dc": f"{MC}/region_static_dc.npz", "austin": f"{MC}/region_static_austin.npz"}
COLS = ["pop", "hh", "carfree", "inc_hh", "rich", "jobs_n", "res_n"]
FULL = {"nyc", "sf", "dc"}                     # regions cover the listed counties completely -> normalised tract weights
LODES_ST = {"chicago": "il", "nyc": "ny", "sf": "ca", "dc": "dc", "austin": "tx"}

src = open(f"{HERE}/build_extra.py").read(); ns = {"__file__": f"{HERE}/build_extra.py"}
exec(src[:src.index("def lodes")], ns); regions = ns["regions"]           # identical region polygons and order
exec(src[src.index("def lodes"):src.index("ap = pd.read_csv")], ns); lodes = ns["lodes"]


def county(st, co):
    path = f"{CACHE}/{RELEASE}_{st}{co}.json"
    if not os.path.exists(path):
        r = requests.get(f"https://api.censusreporter.org/1.0/data/show/{RELEASE}", headers=UA, timeout=180,
                         params={"table_ids": "B01003,B25044,B19001", "geo_ids": f"140|05000US{st}{co}"})
        r.raise_for_status(); open(path, "w").write(r.text)
    d = json.load(open(path))["data"]; rows = []
    for g, t in d.items():
        e = lambda tb, c: float(t[tb]["estimate"].get(c) or 0.0)
        rows.append({"GEOID": g[7:], "pop": e("B01003", "B01003001"), "hh": e("B25044", "B25044001"),
                     "carfree": e("B25044", "B25044003") + e("B25044", "B25044010"),
                     "inc_hh": e("B19001", "B19001001"), "rich": e("B19001", "B19001016") + e("B19001", "B19001017")})
    return pd.DataFrame(rows)


def native(city):
    st, cos = COUNTIES[city]
    reg = regions(city); utm = reg.estimate_utm_crs(); regp = reg.to_crs(utm)
    tr = gpd.read_file(f"zip://{RAW}/cb_2021_{st}_tract_500k.zip").to_crs(utm)
    if city not in FULL:                         # every county with a tract touching the regions (Chicago: O'Hare reaches DuPage)
        cos = sorted(set(gpd.sjoin(tr[["COUNTYFP", "geometry"]], regp[["geometry"]], predicate="intersects")["COUNTYFP"]))
    tr = tr[tr["COUNTYFP"].isin(cos)]
    acs = pd.concat([county(st, co) for co in cos]); tr = tr.merge(acs, on="GEOID", how="left"); miss = int(tr["pop"].isna().sum())
    tr["jobs_n"] = tr["GEOID"].map(lodes(LODES_ST[city], "wac", "w_geocode")); tr["res_n"] = tr["GEOID"].map(lodes(LODES_ST[city], "rac", "h_geocode"))
    tr[COLS] = tr[COLS].fillna(0.0); tr["tarea"] = tr.area
    inter = gpd.overlay(tr[["GEOID", "tarea", "geometry"] + COLS], regp, how="intersection")
    inter["ia"] = inter.area
    w = inter["ia"] / (inter.groupby("GEOID")["ia"].transform("sum") if city in FULL else inter["tarea"])
    k = inter["rid"].map({r: i for i, r in enumerate(reg.rid)}).values
    V = {c: np.bincount(k, weights=inter[c] * w, minlength=len(reg)) for c in COLS}
    n = len(np.load(SP[city])["area_km2"]); assert len(reg) == n, f"{city}: {len(reg)} regions vs {n} in features"
    E = np.load(f"{OUT}/extra_{city}.npz"); assert len(E["jobs"]) == n
    if city not in FULL: assert np.allclose(V["jobs_n"], E["jobs"], rtol=1e-6, atol=1e-6), f"{city}: LODES re-interpolation differs from extra_{city}"
    print(f"{city}: LODES jobs {V['jobs_n'].sum():,.0f} (v1 file {E['jobs'].sum():,.0f}), resident workers {V['res_n'].sum():,.0f} (v1 {E['res'].sum():,.0f})"
          f"{' [normalised tract weights]' if city in FULL else ' [area weights, identical to v1]'}", flush=True)
    city_pop = json.load(open(f"{RAW}/{'acs_austin' if city == 'austin' else 'acs_city'}.json"))["data"][PLACE[city]]["B01003"]["estimate"]["B01003001"]
    print(f"{city}: {len(reg)} regions, {len(tr)} tracts ({miss} without ACS row) | population in regions {V['pop'].sum():,.0f}"
          f" vs ACS city {city_pop:,.0f} ({100 * (V['pop'].sum() / city_pop - 1):+.1f}%) | car-free share {V['carfree'].sum() / V['hh'].sum():.3f}"
          f" | >=150k share {V['rich'].sum() / V['inc_hh'].sum():.3f}", flush=True)
    np.savez(f"{OUT}/acs_{city}.npz", **V)
    return V


def membership_kmeans(H, r, n_blocks, jobs_stored):
    S = np.load(SP[H]); area, cen = S["area_km2"], S["centroid_km"]; jobs = np.load(f"{OUT}/extra_{H}.npz")["jobs"]
    for K in range(n_blocks, n_blocks + 8):
        lab = KMeans(K, n_init=4, random_state=r).fit(cen, sample_weight=area).labels_
        Mm = np.zeros((lab.max() + 1, len(area))); Mm[lab, np.arange(len(area))] = 1; Mm = Mm[Mm.sum(1) > 0]
        if Mm.shape[0] == n_blocks and np.allclose(Mm @ jobs, jobs_stored, rtol=1e-9, atol=1e-6): return Mm
    raise RuntimeError(f"membership of {H} partition {r} with {n_blocks} blocks not reproduced")


def membership_coarse(city, name, jobs_stored):
    src = open(f"{HERE}/build_coarse_targets.py").read(); ns2 = {"__file__": f"{HERE}/build_coarse_targets.py"}
    cwd = os.getcwd(); os.chdir(HERE)
    try: exec(src[:src.index("def build_target")].replace("for c, n in T1.items():", "for c, n in []:"), ns2)
    finally: os.chdir(cwd)
    tr = ns2["tracts"](city); S = np.load(SP[city]); area = S["area_km2"]; utm = tr.estimate_utm_crs(); trp = tr.to_crs(utm)
    U = ns2["units"](city, tr, utm); ov = np.zeros((len(tr), len(U)))
    for j, geom in enumerate(U.geometry): ov[:, j] = trp.intersection(geom).area.values
    lab = ov.argmax(1); none = ov.max(1) <= 0
    if none.any():
        uc = np.stack([U.centroid.x, U.centroid.y], 1); tc = np.stack([trp.centroid.x, trp.centroid.y], 1)
        lab[none] = np.argmin(np.linalg.norm(tc[none][:, None] - uc[None], axis=2), 1)
    used = np.unique(lab); lab = np.searchsorted(used, lab); Mm = np.zeros((len(used), len(area))); Mm[lab, np.arange(len(area))] = 1
    assert np.allclose(Mm @ np.load(f"{OUT}/extra_{city}.npz")["jobs"], jobs_stored, rtol=1e-9, atol=1e-6), f"{name}: membership not reproduced"
    return Mm


V = {c: native(c) for c in COUNTIES}
for name, city in (("sf_zcta", "sf"), ("dc_cluster", "dc")):
    Mm = membership_coarse(city, name, np.load(f"{OUT}/extra_{name}.npz")["jobs"])
    np.savez(f"{OUT}/acs_{name}.npz", **{c: Mm @ V[city][c] for c in COLS}); print(f"{name}: {Mm.shape[0]} coarse regions, membership verified", flush=True)
for f in sorted(glob.glob(f"{OUT}/extra_*_for_*.npz")):
    tag = os.path.basename(f)[6:-4]; m = re.fullmatch(r"(chicago|nyc|sf|dc)_for_(.+?)(?:_p(\d+))?", tag)
    if not m: print(f"{tag}: skipped (not a helper partition)"); continue
    H, r = m.group(1), int(m.group(3) or 0); js = np.load(f)["jobs"]
    Mm = membership_kmeans(H, r, len(js), js)
    np.savez(f"{OUT}/acs_{tag}.npz", **{c: Mm @ V[H][c] for c in COLS}); print(f"{tag}: {Mm.shape[0]} blocks, membership verified", flush=True)
print("done")
