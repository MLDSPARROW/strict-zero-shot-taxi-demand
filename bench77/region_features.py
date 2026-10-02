"""Static features per region (Chicago 77 community areas, NYC 263 taxi zones), from the stable
2000-cell map features (static_core.npz), weighting each fine cell by its overlap area with the region.
Map-only: no demand data is used here. Output: bench77/data/region_static_<city>.npz
Features (19): 8 log place densities, 3 log distances, log busyness, place-mix entropy, 3 missing flags
(all area-weighted from fine cells), log relative area (area / city median region area),
local contrast (log busyness minus mean of 6 nearest regions), grid coverage fraction.
Also stores region centroids (km, local projection) for the neighbour graph.
"""
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__))
STEP2 = os.path.join(HERE, "..", "step2")
D = f"{HERE}/data"
core = np.load(f"{STEP2}/cache/static_core.npz")
STATIC = {"chicago": core["STATIC_CHI"], "nyc": core["STATIC_NYC"]}


def regions(city):
    if city == "chicago":
        g = gpd.read_file(f"{D}/chicago_community_areas.geojson")
        g["rid"] = g["area_numbe"].astype(int)
    else:
        g = gpd.read_file(os.path.join(HERE, "..", "retune_out", "taxi_zones_extracted", "taxi_zones", "taxi_zones.shp"))
        g["rid"] = g["LocationID"].astype(int)
        g = g.dissolve(by="rid", as_index=False)          # a few zone ids have several polygons
    g = g.to_crs(epsg=4326).sort_values("rid").reset_index(drop=True)
    return g[["rid", "geometry"]]


for city in ["chicago", "nyc"]:
    z = np.load(f"{STEP2}/cache/{city}_dense.npz")
    lat_e, lon_e = z["lat_edges"], z["lon_edges"]
    R, C = len(lat_e) - 1, len(lon_e) - 1
    cells = gpd.GeoDataFrame({"cell": np.arange(R * C)},
                             geometry=[box(lon_e[j], lat_e[i], lon_e[j + 1], lat_e[i + 1]) for i in range(R) for j in range(C)],
                             crs="EPSG:4326")
    reg = regions(city)
    utm = reg.estimate_utm_crs()
    reg_p, cells_p = reg.to_crs(utm), cells.to_crs(utm)
    inter = gpd.overlay(cells_p, reg_p, how="intersection")
    inter["a"] = inter.geometry.area
    ids = reg["rid"].values
    idx = {r: k for k, r in enumerate(ids)}
    W = np.zeros((len(ids), R * C))
    np.add.at(W, (inter["rid"].map(idx).values, inter["cell"].values), inter["a"].values)
    area = reg_p.geometry.area.values
    cover = W.sum(axis=1) / area

    f = STATIC[city].reshape(R * C, 16).astype(np.float64)
    Wn = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
    dens = Wn @ np.expm1(f[:, :8])
    other = Wn @ f[:, 8:]                                   # distances, busyness(old), mix(old), flags
    busy = dens.sum(axis=1)
    props = dens / np.clip(busy[:, None], 1e-9, None)
    mix = -np.sum(np.where(props > 0, props * np.log(props + 1e-9), 0.0), axis=1)
    feats = np.concatenate([np.log1p(dens), other[:, 0:3], np.log1p(busy)[:, None], mix[:, None], other[:, 5:8]], axis=1)
    empty = cover < 1e-6
    if empty.any():                                          # outside the map grid: city medians
        feats[empty] = np.median(feats[~empty], axis=0)
    cen = np.stack([reg_p.geometry.centroid.x.values, reg_p.geometry.centroid.y.values], axis=1) / 1000.0
    dist = np.linalg.norm(cen[:, None] - cen[None], axis=2)
    nbr = np.argsort(dist, axis=1)[:, 1:7]
    lbusy = feats[:, 11]
    contrast = lbusy - lbusy[nbr].mean(axis=1)
    rel_area = np.log(area / np.median(area))
    X = np.concatenate([feats, rel_area[:, None], contrast[:, None], np.clip(cover, 0, 1)[:, None]], axis=1).astype(np.float32)
    np.savez(f"{D}/region_static_{city}.npz", X=X, rid=ids, centroid_km=cen, area_km2=area / 1e6, cover=cover)
    print(f"{city}: {len(ids)} regions, features {X.shape}, median area {np.median(area)/1e6:.2f} km2, "
          f"coverage <50%: {(cover < 0.5).sum()} regions, zero coverage: {empty.sum()}")
