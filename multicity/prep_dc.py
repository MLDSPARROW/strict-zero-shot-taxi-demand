"""Washington DC as an extra SOURCE city.
  demand : DC DFHV / OCTO taxi trips 2021 (OpenDataDC_Taxi_2021.zip, 1.43M trips), pickup = city-block lat/lon,
           pickup time rounded to the hour -> 2020 census tracts (TIGER), hourly counts split into 30-min halves
  map    : the notebook's build_static_features() on a 40x50 grid over DC. Place layers fetched by a direct Overpass
           query to the maps.mail.ru mirror (feature centre points; the main Overpass server refused us for SF),
           osmnx as fallback; hard check that every layer + water is non-empty
  regions: tract features exactly like prep_sf.py / bench77/region_features.py
Output: multicity/data/dc_2021_tract_30min.npy, region_static_dc.npz
"""
import os, io, re, json, time, zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__))
D = f"{HERE}/data"; DC = f"{D}/dc"
NB = json.load(open(os.path.join(HERE, "..", "kaggle-fix", "notebook.ipynb"), encoding="utf-8"))
N_DAYS = 365

# ---------------- tracts ----------------
tz = f"{DC}/tl_2020_11_tract.zip"
if not os.path.exists(tz):
    r = requests.get("https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_11_tract.zip", timeout=300); r.raise_for_status()
    open(tz, "wb").write(r.content)
tracts = gpd.read_file(f"zip://{tz}").to_crs(epsg=4326)
tracts = tracts[tracts["ALAND"] > 0].sort_values("GEOID").reset_index(drop=True)
tracts["rid"] = np.arange(len(tracts))
print(f"DC tracts: {len(tracts)}", flush=True)

# ---------------- trips ----------------
zf = zipfile.ZipFile(f"{DC}/OpenDataDC_Taxi_2021.zip")
parts = []
for name in sorted(n for n in zf.namelist() if n.lower().endswith(".csv")):
    df = pd.read_csv(zf.open(name), usecols=["ORIGIN_BLOCK_LATITUDE", "ORIGIN_BLOCK_LONGITUDE", "ORIGINDATETIME_TR"],
                     sep=",", encoding="utf-8-sig")
    parts.append(df.dropna())
    print(f"  {name}: {len(df)} rows", flush=True)
df = pd.concat(parts, ignore_index=True)
t = pd.to_datetime(df["ORIGINDATETIME_TR"], format="%m/%d/%Y %H:%M", errors="coerce")
keep = t.notna() & (t.dt.year == 2021)
df, t = df[keep], t[keep]
day = (t.dt.normalize() - pd.Timestamp("2021-01-01")).dt.days.values
hour = t.dt.hour.values
pts = gpd.GeoDataFrame({"day": day, "hour": hour}, geometry=gpd.points_from_xy(df["ORIGIN_BLOCK_LONGITUDE"], df["ORIGIN_BLOCK_LATITUDE"]), crs="EPSG:4326")
j = gpd.sjoin(pts, tracts[["rid", "geometry"]], how="inner", predicate="within")
H = np.zeros((N_DAYS, 24, len(tracts)), np.float32)
np.add.at(H, (j["day"].values, j["hour"].values, j["rid"].values), 1)
Y = np.repeat(H / 2.0, 2, axis=1)
np.save(f"{D}/dc_2021_tract_30min.npy", Y)
print(f"DC 2021 pickups with block coordinates: {len(pts)}, inside DC tracts: {int(H.sum())} ({100*H.sum()/max(len(pts),1):.1f}%)", flush=True)

# ---------------- fine-grid map features ----------------
fine_path = f"{D}/dc_fine_static.npz"
if not os.path.exists(fine_path):
    import osmnx as ox
    ns = {"np": np, "ox": ox, "GRID_ROWS": 40, "GRID_COLS": 50}
    c2 = "".join(NB["cells"][2]["source"])
    exec(re.search(r"def assign_cell\(.*?(?=\ndef |\Z)", c2, re.S).group(0), ns)
    c17 = "".join(NB["cells"][17]["source"]).replace("!pip -q install osmnx xgboost", "")
    c17 = c17[:c17.index("# Querying OSM for 8 POI categories")] if "# Querying OSM for 8 POI categories" in c17 else c17
    exec(c17, ns)
    ns["DOWNTOWN_QUERY"]["dc"] = "Farragut Square, Washington, DC, USA"
    orig = ns["fetch_points"]

    def direct_fetch(north, south, east, west, tags, label, url="https://maps.mail.ru/osm/tools/overpass/api/interpreter"):
        bb = f"({south},{west},{north},{east})"; q = []
        for k, v in tags.items():
            vals = [v] if isinstance(v, str) else v
            flt = f'["{k}"]' if v is True else f'["{k}"~"^({"|".join(vals)})$"]'
            q += [f"node{flt}{bb};", f"way{flt}{bb};", f"relation{flt}{bb};"]
        r = requests.post(url, data={"data": f"[out:json][timeout:240];({''.join(q)});out center;"}, timeout=300); r.raise_for_status()
        pts_ = [(e.get("lat", e.get("center", {}).get("lat")), e.get("lon", e.get("center", {}).get("lon"))) for e in r.json()["elements"]]
        out = np.array([p for p in pts_ if p[0] is not None], dtype=np.float32).reshape(-1, 2)
        print(f"  {label}: {len(out)} features (direct Overpass query)", flush=True)
        return out

    def strict_fetch(north, south, east, west, tags, label):
        for attempt in range(6):
            try:
                p = direct_fetch(north, south, east, west, tags, label) if attempt < 5 else orig(north, south, east, west, tags, label)
            except Exception as e:
                print(f"  {label}: attempt {attempt+1} failed ({type(e).__name__}: {str(e)[:100]})", flush=True)
                p = np.empty((0, 2), np.float32)
            if len(p): return p
            time.sleep(20)
        raise RuntimeError(f"OSM layer {label} still empty after retries -- refusing to build incomplete features")
    ns["fetch_points"] = strict_fetch
    ox.geocode(ns["DOWNTOWN_QUERY"]["dc"])
    minx, miny, maxx, maxy = tracts.total_bounds
    lat_e, lon_e = np.linspace(miny, maxy, 41), np.linspace(minx, maxx, 51)
    static = ns["build_static_features"]("dc", lat_e, lon_e)
    assert static.shape == (40, 50, 16) and (static[..., 13:16] == 0).all(), "a missing-data flag is set"
    np.savez(fine_path, static=static, lat_edges=lat_e, lon_edges=lon_e)
fz = np.load(fine_path); static, lat_e, lon_e = fz["static"], fz["lat_edges"], fz["lon_edges"]
print("DC fine static:", static.shape, "empty place layers:", int((np.expm1(static[..., :8]).sum(axis=(0, 1)) == 0).sum()), flush=True)

# ---------------- tract features ----------------
R, C = 40, 50
cells = gpd.GeoDataFrame({"cell": np.arange(R * C)}, geometry=[box(lon_e[j_], lat_e[i], lon_e[j_ + 1], lat_e[i + 1])
                          for i in range(R) for j_ in range(C)], crs="EPSG:4326")
utm = tracts.estimate_utm_crs()
reg_p, cells_p = tracts[["rid", "geometry"]].to_crs(utm), cells.to_crs(utm)
inter = gpd.overlay(cells_p, reg_p, how="intersection"); inter["a"] = inter.geometry.area
W = np.zeros((len(tracts), R * C)); np.add.at(W, (inter["rid"].values, inter["cell"].values), inter["a"].values)
area = reg_p.geometry.area.values; cover = W.sum(axis=1) / area
f = static.reshape(R * C, 16).astype(np.float64)
Wn = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
dens = Wn @ np.expm1(f[:, :8]); other = Wn @ f[:, 8:]
busy = dens.sum(axis=1)
props = dens / np.clip(busy[:, None], 1e-9, None)
mix = -np.sum(np.where(props > 0, props * np.log(props + 1e-9), 0.0), axis=1)
feats = np.concatenate([np.log1p(dens), other[:, 0:3], np.log1p(busy)[:, None], mix[:, None], other[:, 5:8]], axis=1)
cen_km = np.stack([reg_p.geometry.centroid.x.values, reg_p.geometry.centroid.y.values], axis=1) / 1000.0
nbr = np.argsort(np.linalg.norm(cen_km[:, None] - cen_km[None], axis=2), axis=1)[:, 1:7]
contrast = feats[:, 11] - feats[:, 11][nbr].mean(axis=1)
X = np.concatenate([feats, np.log(area / np.median(area))[:, None], contrast[:, None], np.clip(cover, 0, 1)[:, None]], axis=1).astype(np.float32)
np.savez(f"{D}/region_static_dc.npz", X=X, rid=tracts["rid"].values, centroid_km=cen_km, area_km2=area / 1e6, cover=cover)
print(f"DC tract features {X.shape}, median tract area {np.median(area)/1e6:.2f} km2, min cover {cover.min():.2f}", flush=True)
print("done", flush=True)
