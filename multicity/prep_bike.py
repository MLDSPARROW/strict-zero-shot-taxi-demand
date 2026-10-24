"""PROTOCOL_V9.md: a bike-share city as an extra SOURCE city for the allocation (map data and bike-share trip origins only; no taxi data).
  demand : bike-share trips of 2021 (start time, start coordinates) -> 2020 census tracts (cartographic boundary 2021), counts per day and 30-minute slot
  regions: the tracts that contain at least one trip origin of 2021 and whose centroid lies within 12 km of the downtown point of the city
  map    : exactly as prep_dc.py: the notebook's build_static_features() on a 40x50 grid over the regions, place layers by Overpass, region
           features area-weighted from the grid cells as in bench77/region_features.py
Usage (from multicity/): python prep_bike.py <boston|philadelphia|la|minneapolis|columbus>
Output: multicity/data/bike/<city>/bike_<city>_2021_tract_30min.npy, region_static_bike_<city>.npz, summary.txt; improve/data/enpl_bike_<city>.npy (zeros)"""
import os, sys, io, re, json, time, glob, zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__)); BK = f"{HERE}/data/bike"; CITY = sys.argv[1]; OUT = f"{BK}/{CITY}"
CFG = {"boston": dict(state="25", fmt="bluebikes", downtown="Downtown Crossing, Boston, MA, USA"),
       "philadelphia": dict(state="42", fmt="indego", downtown="Philadelphia City Hall, Philadelphia, PA, USA"),
       "la": dict(state="06", fmt="indego", downtown="Pershing Square, Los Angeles, CA, USA"),
       "minneapolis": dict(state="27", fmt="lyft", downtown="Nicollet Mall, Minneapolis, MN, USA"),
       "columbus": dict(state="39", fmt="lyft", downtown="Capitol Square, Columbus, Ohio, USA")}[CITY]
KEY = f"bike_{CITY}"; RADIUS_KM = 12.0; N_DAYS = 365
NB = json.load(open(os.path.join(HERE, "..", "kaggle-fix", "notebook.ipynb"), encoding="utf-8"))

# ---------------- trips ----------------
parts = []
for z in sorted(glob.glob(f"{OUT}/raw/*.zip")):
    zf = zipfile.ZipFile(z)
    for name in zf.namelist():
        base = name.lower().split("/")[-1]
        if not base.endswith(".csv") or "__macosx" in name.lower() or ("trip" not in base): continue
        if CFG["fmt"] == "bluebikes":
            df = pd.read_csv(zf.open(name), usecols=["starttime", "start station latitude", "start station longitude"])
            df.columns = ["t", "lat", "lon"]; df["t"] = pd.to_datetime(df["t"], errors="coerce")
        elif CFG["fmt"] == "indego":
            df = pd.read_csv(zf.open(name), usecols=["trip_id", "start_time", "start_lat", "start_lon"])
            df.columns = ["id", "t", "lat", "lon"]; df["t"] = pd.to_datetime(df["t"], format="%m/%d/%Y %H:%M", errors="coerce")
        else:
            df = pd.read_csv(zf.open(name), usecols=["ride_id", "started_at", "start_lat", "start_lng"])
            df.columns = ["id", "t", "lat", "lon"]; df["t"] = pd.to_datetime(df["t"], errors="coerce")
        parts.append(df); print(f"  {os.path.basename(z)} :: {name}: {len(df)} rows", flush=True)
trips = pd.concat(parts, ignore_index=True)
if "id" in trips: trips = trips.drop_duplicates("id")
n_raw = len(trips); trips = trips.dropna(subset=["t", "lat", "lon"]); trips = trips[trips["t"].dt.year == 2021]
print(f"{CITY}: {n_raw} rows read, {len(trips)} trips in 2021 with a start time and coordinates", flush=True)

# ---------------- tracts ----------------
ss = CFG["state"]; tz = f"{OUT}/cb_2021_{ss}_tract_500k.zip"
local = os.path.join(HERE, "..", "improve", "raw", f"cb_2021_{ss}_tract_500k.zip")
if not os.path.exists(tz):
    if os.path.exists(local): import shutil; shutil.copy(local, tz)
    else:
        r = requests.get(f"https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_{ss}_tract_500k.zip", timeout=300); r.raise_for_status(); open(tz, "wb").write(r.content)
tr = gpd.read_file(f"zip://{tz}").to_crs(epsg=4326); tr = tr[tr["ALAND"] > 0].sort_values("GEOID").reset_index(drop=True)
utm = tr.estimate_utm_crs()

# downtown point of the city (geocoded once and stored)
import osmnx as ox
dpath = f"{OUT}/downtown.json"
if not os.path.exists(dpath): json.dump(list(ox.geocode(CFG["downtown"])), open(dpath, "w"))
dlat, dlon = json.load(open(dpath)); dpt = gpd.GeoSeries(gpd.points_from_xy([dlon], [dlat]), crs="EPSG:4326").to_crs(utm).iloc[0]

pts = gpd.GeoDataFrame(trips[["t"]].reset_index(drop=True), geometry=gpd.points_from_xy(trips["lon"].values, trips["lat"].values), crs="EPSG:4326")
j = gpd.sjoin(pts, tr[["GEOID", "geometry"]], how="inner", predicate="within")
cen = tr.to_crs(utm).geometry.centroid; near = (cen.distance(dpt) / 1000.0) <= RADIUS_KM
has = tr["GEOID"].isin(set(j["GEOID"]))
tracts = tr[near & has].reset_index(drop=True); tracts["rid"] = np.arange(len(tracts))
print(f"{CITY}: {len(tracts)} tracts with trip origins within {RADIUS_KM:.0f} km of the downtown point (of {int(has.sum())} tracts with any origin)", flush=True)
j = j[j["GEOID"].isin(set(tracts["GEOID"]))].merge(tracts[["GEOID", "rid"]], on="GEOID")
t = j["t"]; day = (t.dt.normalize() - pd.Timestamp("2021-01-01")).dt.days.values; slot = (t.dt.hour * 2 + (t.dt.minute >= 30)).values
Y = np.zeros((N_DAYS, 48, len(tracts)), np.float32); np.add.at(Y, (day, slot, j["rid"].values), 1)
np.save(f"{OUT}/bike_{CITY}_2021_tract_30min.npy", Y)
tot = Y.sum(2); active = int((tot.sum(1) > 0).sum())
print(f"{CITY}: {int(Y.sum())} trips kept ({100 * Y.sum() / max(len(trips), 1):.1f}% of the 2021 trips), {active} days with trips, mean {tot[tot > 0].mean():.1f} trips per active slot", flush=True)

# ---------------- fine-grid map features (as prep_dc.py) ----------------
fine_path = f"{OUT}/{CITY}_fine_static.npz"
if not os.path.exists(fine_path):
    ns = {"np": np, "ox": ox, "GRID_ROWS": 40, "GRID_COLS": 50}
    c2 = "".join(NB["cells"][2]["source"])
    exec(re.search(r"def assign_cell\(.*?(?=\ndef |\Z)", c2, re.S).group(0), ns)
    c17 = "".join(NB["cells"][17]["source"]).replace("!pip -q install osmnx xgboost", "")
    c17 = c17[:c17.index("# Querying OSM for 8 POI categories")] if "# Querying OSM for 8 POI categories" in c17 else c17
    exec(c17, ns)
    ns["DOWNTOWN_QUERY"][KEY] = CFG["downtown"]; orig = ns["fetch_points"]
    # the Swiss mirror (overpass.osm.ch) holds Switzerland only and the other mirrors time out, so the main server is used, one query at a time
    ENDPOINTS = ["https://overpass-api.de/api/interpreter"]
    UA = {"User-Agent": "strict-zero-shot-taxi-demand research script (github.com/MLDSPARROW/strict-zero-shot-taxi-demand)"}

    def direct_fetch_one(north, south, east, west, tags, label, url):
        bb = f"({south},{west},{north},{east})"; q = []
        for k, v in tags.items():
            vals = [v] if isinstance(v, str) else v
            flt = f'["{k}"]' if v is True else f'["{k}"~"^({"|".join(vals)})$"]'
            q += [f"node{flt}{bb};", f"way{flt}{bb};", f"relation{flt}{bb};"]
        r = requests.post(url, data={"data": f"[out:json][timeout:300];({''.join(q)});out center;"}, headers=UA, timeout=420); r.raise_for_status()
        pts_ = [(e.get("lat", e.get("center", {}).get("lat")), e.get("lon", e.get("center", {}).get("lon"))) for e in r.json()["elements"]]
        return np.array([p for p in pts_ if p[0] is not None], dtype=np.float32).reshape(-1, 2)

    def direct_fetch(north, south, east, west, tags, label, url, tile=1):
        if tile == 1: out = direct_fetch_one(north, south, east, west, tags, label, url)
        else:
            lat_e_, lon_e_ = np.linspace(south, north, tile + 1), np.linspace(west, east, tile + 1); parts_ = []
            for i in range(tile):
                for j_ in range(tile):
                    parts_.append(direct_fetch_one(lat_e_[i + 1], lat_e_[i], lon_e_[j_ + 1], lon_e_[j_], tags, f"{label}[{i},{j_}]", url)); time.sleep(3)
            out = np.concatenate(parts_, axis=0) if parts_ else np.empty((0, 2), dtype=np.float32)
        print(f"  {label}: {len(out)} features (direct Overpass query, tile={tile})", flush=True); return out

    CACHE_DIR = f"{HERE}/../osm_cache"; os.makedirs(CACHE_DIR, exist_ok=True)

    def strict_fetch(north, south, east, west, tags, label):
        cache_path = f"{CACHE_DIR}/{label.replace('/', '_')}.npy"
        if os.path.exists(cache_path):
            p = np.load(cache_path); print(f"  {label}: {len(p)} features (from local cache, no network used)", flush=True); return p
        for attempt in range(8):                                        # attempts 0-3: one query; 4-6: the box in four tiles; 7: osmnx
            if attempt == 7:
                try: p = orig(north, south, east, west, tags, label)
                except Exception as e:
                    print(f"  {label}: osmnx default failed ({type(e).__name__}: {str(e)[:120]})", flush=True); p = np.empty((0, 2), np.float32)
            else:
                url = ENDPOINTS[0]; tile = 1 if attempt <= 3 else 2
                try: p = direct_fetch(north, south, east, west, tags, label, url, tile=tile)
                except Exception as e:
                    print(f"  {label}: attempt {attempt + 1} via {url} (tile={tile}) failed ({type(e).__name__}: {str(e)[:100]})", flush=True); p = np.empty((0, 2), np.float32)
            if len(p): np.save(cache_path, p); return p
            wait = 45 * (1 + attempt); print(f"  {label}: empty on attempt {attempt + 1}, waiting {wait}s", flush=True); time.sleep(wait)
        raise RuntimeError(f"OSM layer {label} still empty after retries -- refusing to build incomplete features")
    ns["fetch_points"] = strict_fetch
    minx, miny, maxx, maxy = tracts.total_bounds
    lat_e, lon_e = np.linspace(miny, maxy, 41), np.linspace(minx, maxx, 51)
    static = ns["build_static_features"](KEY, lat_e, lon_e)
    assert static.shape == (40, 50, 16) and (static[..., 13:16] == 0).all(), "a missing-data flag is set"
    np.savez(fine_path, static=static, lat_edges=lat_e, lon_edges=lon_e)
fz = np.load(fine_path); static, lat_e, lon_e = fz["static"], fz["lat_edges"], fz["lon_edges"]
print(f"{CITY} fine static:", static.shape, "empty place layers:", int((np.expm1(static[..., :8]).sum(axis=(0, 1)) == 0).sum()), flush=True)

# ---------------- tract features (as prep_dc.py and bench77/region_features.py) ----------------
R, C = 40, 50
cells = gpd.GeoDataFrame({"cell": np.arange(R * C)}, geometry=[box(lon_e[j_], lat_e[i], lon_e[j_ + 1], lat_e[i + 1]) for i in range(R) for j_ in range(C)], crs="EPSG:4326")
reg_p, cells_p = tracts[["rid", "geometry"]].to_crs(utm), cells.to_crs(utm)
inter = gpd.overlay(cells_p, reg_p, how="intersection"); inter["a"] = inter.geometry.area
W = np.zeros((len(tracts), R * C)); np.add.at(W, (inter["rid"].values, inter["cell"].values), inter["a"].values)
area = reg_p.geometry.area.values; cover = W.sum(axis=1) / area
ca = cells_p.area.values; frac = np.zeros((len(tracts), R * C), np.float32)         # grid inputs for the scale-matched re-aggregation (grid/grid_prep.py)
np.add.at(frac, (inter["rid"].values, inter["cell"].values), (inter["a"].values / ca[inter["cell"].values]).astype(np.float32))
os.makedirs(f"{HERE}/../grid/data", exist_ok=True)
np.savez(f"{HERE}/../grid/data/grid_bike_{CITY}.npz", X=static.reshape(R * C, -1).astype(np.float32), frac=frac, cell_km2=(ca / 1e6).astype(np.float32),
         enpl=np.zeros(R * C, np.float32), shape=np.array([R, C]))
f = static.reshape(R * C, 16).astype(np.float64); Wn = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
dens = Wn @ np.expm1(f[:, :8]); other = Wn @ f[:, 8:]; busy = dens.sum(axis=1); props = dens / np.clip(busy[:, None], 1e-9, None)
mix = -np.sum(np.where(props > 0, props * np.log(props + 1e-9), 0.0), axis=1)
feats = np.concatenate([np.log1p(dens), other[:, 0:3], np.log1p(busy)[:, None], mix[:, None], other[:, 5:8]], axis=1)
cen_km = np.stack([reg_p.geometry.centroid.x.values, reg_p.geometry.centroid.y.values], axis=1) / 1000.0
nbr = np.argsort(np.linalg.norm(cen_km[:, None] - cen_km[None], axis=2), axis=1)[:, 1:7]
contrast = feats[:, 11] - feats[:, 11][nbr].mean(axis=1)
X = np.concatenate([feats, np.log(area / np.median(area))[:, None], contrast[:, None], np.clip(cover, 0, 1)[:, None]], axis=1).astype(np.float32)
np.savez(f"{OUT}/region_static_bike_{CITY}.npz", X=X, rid=tracts["rid"].values, centroid_km=cen_km, area_km2=area / 1e6, cover=cover)
np.save(os.path.join(HERE, "..", "improve", "data", f"enpl_bike_{CITY}.npy"), np.zeros(len(tracts)))
msg = (f"{CITY}: {len(tracts)} tracts, {int(Y.sum())} trips, {active} active days, features {X.shape}, median tract area {np.median(area) / 1e6:.2f} km2, "
       f"min grid cover {cover.min():.2f}, tracts with no trip in the year {int(((Y.sum((0, 1)) == 0)).sum())}")
open(f"{OUT}/summary.txt", "w").write(msg + "\n"); print(msg, flush=True); print("done", flush=True)
