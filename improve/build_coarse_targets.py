"""Coarse versions of the held-out targets (PROTOCOL_COARSE_TARGETS.md, written and pushed before this script ran).
Each SF / DC tract is assigned to the official coarse unit it overlaps most (SF: 2020 Census ZCTAs clipped to the city,
DC: Neighborhood Clusters); coarse-region features, trips and covariates are exact sums over member tracts, using the same
block-feature formula as the re-aggregated helpers. Then the scale-matched helper partitions p1..p10 for each coarse target
(and for native DC, the contrast condition) are built exactly as in aggregate.py / aggregate_multi.py.
Run from improve/. Outputs: ../multicity/data/{sf_zcta,dc_cluster}_30min.npy, region_static_{sf_zcta,dc_cluster}.npz,
data/extra_*.npz, data/enpl_*.npy, data/agg/<H>_for_<T>_p<r>_*."""
import os, numpy as np, geopandas as gpd
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from sklearn.cluster import KMeans
src = open("aggregate.py").read()
ns = {}; exec(src[:src.index("# check: the formula")], ns); feats, YP, SP, G = ns["feats"], ns["YP"], ns["SP"], ns["G"]
M = "../multicity/data"
YP.setdefault("dc", f"{M}/dc_2021_tract_30min.npy"); SP.setdefault("dc", f"{M}/region_static_dc.npz")
os.makedirs("data/agg", exist_ok=True)

# 0. rebuilt demand arrays must reproduce the paper's Table 1 trip totals
T1 = {"chicago": 3639126, "nyc": 30533607, "sf": 1703882, "dc": 1195111}
for c, n in T1.items():
    s = np.load(YP[c]).astype(np.float64).sum(); print(f"{c}: rebuilt trips {s:,.0f} vs Table 1 {n:,} -> {'OK' if abs(s - n) < 1 else 'MISMATCH'}")
    assert abs(s - n) < 1, c


def tracts(city):
    if city == "sf":
        g = gpd.read_file(f"{M}/sf_tracts_2020.geojson").to_crs(4326)
        idc = next(c for c in g.columns if c.lower() in ("geoid", "tractce", "geoid20", "tractce20", "name"))
        cen = g.to_crs(g.estimate_utm_crs()).geometry.centroid.to_crs(4326)
        g = g[cen.x.values > -122.6].dissolve(by=idc, as_index=False).sort_values(idc).reset_index(drop=True)
    else:
        g = gpd.read_file(f"zip://{M}/dc/tl_2020_11_tract.zip").to_crs(4326)
        g = g[g["ALAND"] > 0].sort_values("GEOID").reset_index(drop=True)
    return g


def units(city, tr, utm):
    if city == "sf":
        z = gpd.read_file(f"{M}/sf_zcta_raw.geojson").to_crs(utm); city_poly = tr.to_crs(utm).union_all()
        z = z[z.intersection(city_poly).area / z.area > 0.5].reset_index(drop=True)
        z["geometry"] = z.intersection(city_poly); return z
    return gpd.read_file(f"{M}/dc_neighborhood_clusters.geojson").to_crs(utm).reset_index(drop=True)


def build_target(city, name):
    tr = tracts(city); S = np.load(SP[city]); area = S["area_km2"]; utm = tr.estimate_utm_crs(); trp = tr.to_crs(utm)
    a_chk = trp.area.values / 1e6
    assert len(tr) == len(area) and np.abs(a_chk - area).max() < 1e-3, f"{city}: tract order/area differs from archived features"
    U = units(city, tr, utm); ov = np.zeros((len(tr), len(U)))
    for j, geom in enumerate(U.geometry): ov[:, j] = trp.intersection(geom).area.values
    lab = ov.argmax(1)
    none = ov.max(1) <= 0
    if none.any():                                               # tracts touching no kept unit: nearest unit centroid
        uc = np.stack([U.centroid.x, U.centroid.y], 1); tc = np.stack([trp.centroid.x, trp.centroid.y], 1)
        lab[none] = np.argmin(np.linalg.norm(tc[none][:, None] - uc[None], axis=2), 1)
    used = np.unique(lab); lab = np.searchsorted(used, lab); K = len(used)
    Mm = np.zeros((K, len(area))); Mm[lab, np.arange(len(area))] = 1
    g = np.load(f"{G}/grid_{city}.npz"); frac = Mm @ g["frac"]; a2 = Mm @ area
    cen = (Mm @ (S["centroid_km"] * area[:, None])) / a2[:, None]; X2 = feats(frac, g["cell_km2"], g["X"], a2, cen)
    Y = np.load(YP[city]).astype(np.float32); Y2 = (Y.reshape(-1, Y.shape[2]) @ Mm.T.astype(np.float32)).reshape(Y.shape[0], 48, K)
    E = np.load(f"data/extra_{city}.npz"); jobs, res, rail = Mm @ E["jobs"], Mm @ E["res"], Mm @ E["rail"]
    enpl = Mm @ np.load(f"data/enpl_{city}.npy")
    E2 = np.stack([np.log1p(jobs / a2), np.log1p(res / a2), np.log1p(rail / a2), (enpl > 0).astype(float)], 1).astype(np.float32)
    np.save(f"{M}/{name}_30min.npy", Y2)
    np.savez(f"{M}/region_static_{name}.npz", X=X2, rid=np.arange(K), centroid_km=cen, area_km2=a2, cover=X2[:, 18])
    np.savez(f"data/extra_{name}.npz", E=E2, jobs=jobs, res=res, rail=rail); np.save(f"data/enpl_{name}.npy", enpl.astype(np.float32))
    print(f"{name}: {len(area)} tracts -> {K} coarse regions, median area {np.median(a2):.2f} km2 "
          f"(native {np.median(area):.2f}), trips kept {Y2.sum()/Y.sum():.4f}, regions with no trips {(Y2.sum((0,1))==0).sum()}")
    return np.median(a2)


def build_partitions(H, T, aT, tag_T):
    S = np.load(SP[H]); area = S["area_km2"]
    if np.median(area) >= aT: print(f"{H} for {tag_T}: already >= target size, left native"); return
    best = None
    for K in range(4, len(area)):                               # same search as aggregate.py (random_state 0)
        lab_ = KMeans(K, n_init=4, random_state=0).fit(S["centroid_km"], sample_weight=area).labels_
        med = np.median(np.bincount(lab_, weights=area, minlength=K))
        if best is None or abs(np.log(med / aT)) < abs(np.log(best[1] / aT)): best = (K, med)
        if med < aT * 0.8: break
    K = best[0]; g = np.load(f"{G}/grid_{H}.npz"); Y = np.load(YP[H]).astype(np.float32); E = np.load(f"data/extra_{H}.npz")
    meds = []
    for r in range(1, 11):                                      # partitions as in aggregate_multi.py / aggregate_multi2.py
        lab = KMeans(K, n_init=4, random_state=r).fit(S["centroid_km"], sample_weight=area).labels_
        Mm = np.zeros((lab.max() + 1, len(area))); Mm[lab, np.arange(len(area))] = 1; Mm = Mm[Mm.sum(1) > 0]
        frac = Mm @ g["frac"]; a2 = Mm @ area; cen = (Mm @ (S["centroid_km"] * area[:, None])) / a2[:, None]
        X2 = feats(frac, g["cell_km2"], g["X"], a2, cen); Y2 = (Y.reshape(-1, Y.shape[2]) @ Mm.T.astype(np.float32)).reshape(Y.shape[0], 48, -1)
        jobs, res, rail = Mm @ E["jobs"], Mm @ E["res"], Mm @ E["rail"]; enpl = Mm @ np.load(f"data/enpl_{H}.npy")
        E2 = np.stack([np.log1p(jobs / a2), np.log1p(res / a2), np.log1p(rail / a2), (enpl > 0).astype(float)], 1).astype(np.float32)
        tag = f"{H}_for_{tag_T}_p{r}"
        np.save(f"data/agg/{tag}_Y.npy", Y2); np.savez(f"data/agg/{tag}_static.npz", X=X2, rid=np.arange(len(a2)), centroid_km=cen, area_km2=a2, cover=X2[:, 18])
        np.savez(f"data/extra_{tag}.npz", E=E2, jobs=jobs, res=res, rail=rail); np.save(f"data/enpl_{tag}.npy", enpl.astype(np.float32))
        meds.append(np.median(a2))
    print(f"{H} for {tag_T}: {len(area)} regions -> K={K} blocks, median block area {np.mean(meds):.2f} km2 over 10 partitions (target {aT:.2f})")


a_sf = build_target("sf", "sf_zcta")
a_dc = build_target("dc", "dc_cluster")
for H in ("chicago", "nyc"): build_partitions(H, "sf", a_sf, "sf_zcta")
for H in ("chicago", "nyc", "sf"): build_partitions(H, "dc", a_dc, "dc_cluster")
a_dcn = np.median(np.load(SP["dc"])["area_km2"])
for H in ("chicago", "nyc", "sf"): build_partitions(H, "dc", a_dcn, "dc")
print("done")
