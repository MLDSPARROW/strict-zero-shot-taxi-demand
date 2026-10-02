"""San Francisco as an extra SOURCE city (multi-city step 1).
  demand : SFMTA taxi trips 2023 (DataSF m8hk-2ipk), pickup points -> 2020 census tracts, 30-min slots
  map    : the notebook's own build_static_features() on a 40x50 grid over SF (identical feature definitions),
           with OSM retries and a hard check that every place layer + water came back non-empty (live OSM
           failures silently emptied layers in earlier Kaggle runs)
  regions: tract-level features exactly like bench77/region_features.py (area-weighted from the fine grid)
Output: multicity/data/sf_2023_tract_30min.npy, region_static_sf.npz, sf_fine_static.npz
"""
import os, json, time, re
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__))
D = f"{HERE}/data"; os.makedirs(D, exist_ok=True)
NB = json.load(open(os.path.join(HERE, "..", "kaggle-fix", "notebook.ipynb"), encoding="utf-8"))
DAYS = pd.date_range("2023-01-01", "2023-12-31", freq="D"); N_DAYS = len(DAYS)

# ---------------- tracts ----------------
tr_path = f"{D}/sf_tracts_2020.geojson"
if not os.path.exists(tr_path):
    r = requests.get("https://data.sfgov.org/resource/tmph-tgz9.geojson", params={"$limit": 5000}, timeout=300)
    r.raise_for_status(); open(tr_path, "wb").write(r.content)
tracts = gpd.read_file(tr_path).to_crs(epsg=4326)
idcol = next(c for c in tracts.columns if c.lower() in ("geoid", "tractce", "geoid20", "tractce20", "name"))
cen = tracts.to_crs(tracts.estimate_utm_crs()).geometry.centroid.to_crs(epsg=4326)
tracts = tracts[cen.x.values > -122.6]                       # drop the Farallon Islands tract
tracts = tracts.dissolve(by=idcol, as_index=False).sort_values(idcol).reset_index(drop=True)
tracts["rid"] = np.arange(len(tracts))
print(f"SF tracts: {len(tracts)} (id column '{idcol}')", flush=True)

# ---------------- trips ----------------
raw_path = f"{D}/sf_2023_pickups.npz"
if not os.path.exists(raw_path):
    ts, lat, lon, off = [], [], [], 0
    while True:
        params = {"$where": "start_time_local >= '2023-01-01T00:00:00' AND start_time_local < '2024-01-01T00:00:00'",
                  "$order": "start_time_local", "$limit": 50000, "$offset": off}
        for attempt in range(6):
            try:
                r = requests.get("https://data.sfgov.org/resource/m8hk-2ipk.json", params=params, timeout=600)
                r.raise_for_status(); rows = r.json(); break
            except Exception as e:
                print("  retry", off, e, flush=True); time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError(f"SF page at offset {off} failed")
        for x in rows:
            if x.get("pickup_location_latitude") and x.get("pickup_location_longitude"):
                ts.append(x["start_time_local"]); lat.append(float(x["pickup_location_latitude"])); lon.append(float(x["pickup_location_longitude"]))
        off += len(rows)
        print(f"  SF rows fetched: {off}", flush=True)
        if len(rows) < 50000: break
    np.savez(raw_path, ts=np.array(ts), lat=np.array(lat), lon=np.array(lon))
z = np.load(raw_path)
t = pd.to_datetime(pd.Series(z["ts"]))
day = (t.dt.normalize() - pd.Timestamp("2023-01-01")).dt.days.values
slot = (t.dt.hour.values * 2 + (t.dt.minute.values >= 30)).astype(int)
pts = gpd.GeoDataFrame({"day": day, "slot": slot}, geometry=gpd.points_from_xy(z["lon"], z["lat"]), crs="EPSG:4326")
joined = gpd.sjoin(pts, tracts[["rid", "geometry"]], how="inner", predicate="within")
Y = np.zeros((N_DAYS, 48, len(tracts)), dtype=np.float32)
ok = (joined["day"].values >= 0) & (joined["day"].values < N_DAYS)
np.add.at(Y, (joined["day"].values[ok], joined["slot"].values[ok], joined["rid"].values[ok]), 1)
np.save(f"{D}/sf_2023_tract_30min.npy", Y)
print(f"SF 2023 pickups: {len(pts)} with coordinates, {int(Y.sum())} inside SF tracts "
      f"({100*Y.sum()/max(len(pts),1):.1f}%; the rest are mostly SFO airport, outside the city)", flush=True)

# ---------------- fine-grid map features, notebook code ----------------
fine_path = f"{D}/sf_fine_static.npz"
if not os.path.exists(fine_path):
    import osmnx as ox
    ns = {"np": np, "ox": ox, "GRID_ROWS": 40, "GRID_COLS": 50}
    c2 = "".join(NB["cells"][2]["source"])
    m = re.search(r"def assign_cell\(.*?(?=\ndef |\Z)", c2, re.S); exec(m.group(0), ns)
    c17 = "".join(NB["cells"][17]["source"]).replace("!pip -q install osmnx xgboost", "")
    c17 = c17[:c17.index("# Querying OSM for 8 POI categories")] if "# Querying OSM for 8 POI categories" in c17 else c17
    exec(c17, ns)
    ns["DOWNTOWN_QUERY"]["sf"] = "Union Square, San Francisco, CA, USA"
    orig = ns["fetch_points"]
    # attempt 0 = osmnx defaults (same request as the first run -> the 7 layers it fetched come from osmnx's cache);
    # then the mirror that currently answers (overpass-api.de: 406, kumi/private.coffee: 429 rate-limited)
    DEFAULT_URL, DEFAULT_TIMEOUT = ox.settings.overpass_url, ox.settings.requests_timeout
    ENDPOINTS = [DEFAULT_URL, "https://maps.mail.ru/osm/tools/overpass/api", "https://overpass.kumi.systems/api",
                 "https://maps.mail.ru/osm/tools/overpass/api", "https://overpass.private.coffee/api"]
    def direct_fetch(north, south, east, west, tags, label, url="https://maps.mail.ru/osm/tools/overpass/api/interpreter"):
        """Plain Overpass query (nodes + ways + relations, 'out center'): one point per OSM feature -- the
        feature centre, vs osmnx's representative point (only differs inside large polygons)."""
        bb = f"({south},{west},{north},{east})"
        parts = []
        for k, v in tags.items():
            vals = [v] if isinstance(v, str) else v            # a single tag value (e.g. natural=water) is a str
            flt = f'["{k}"]' if v is True else f'["{k}"~"^({"|".join(vals)})$"]'
            parts += [f"node{flt}{bb};", f"way{flt}{bb};", f"relation{flt}{bb};"]
        q = f"[out:json][timeout:240];({''.join(parts)});out center;"
        r = requests.post(url, data={"data": q}, timeout=300); r.raise_for_status()
        pts = [(e.get("lat", e.get("center", {}).get("lat")), e.get("lon", e.get("center", {}).get("lon"))) for e in r.json()["elements"]]
        out = np.array([p for p in pts if p[0] is not None], dtype=np.float32).reshape(-1, 2)
        print(f"  {label}: {len(out)} features (direct Overpass query)", flush=True)
        return out

    def strict_fetch(north, south, east, west, tags, label):
        for attempt in range(8):
            if attempt == 0:                              # osmnx defaults -> cache hit for layers fetched earlier
                ox.settings.overpass_url, ox.settings.requests_timeout = DEFAULT_URL, DEFAULT_TIMEOUT   # same key as the first run
                p = orig(north, south, east, west, tags, label)
            else:                                         # osmnx hangs on the working mirror; query it directly
                try:
                    p = direct_fetch(north, south, east, west, tags, label)
                except Exception as e:
                    print(f"  {label}: direct query failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
                    p = np.empty((0, 2), dtype=np.float32)
            if len(p): return p
            print(f"  {label}: empty via {ox.settings.overpass_url} -> retry {attempt+1}", flush=True); time.sleep(20)
        raise RuntimeError(f"OSM layer {label} still empty after retries -- refusing to build incomplete features")
    ns["fetch_points"] = strict_fetch
    ox.geocode(ns["DOWNTOWN_QUERY"]["sf"])                       # fail loudly rather than fall back to bbox centre
    minx, miny, maxx, maxy = tracts.total_bounds
    lat_e, lon_e = np.linspace(miny, maxy, 41), np.linspace(minx, maxx, 51)
    static = ns["build_static_features"]("sf", lat_e, lon_e)
    assert static.shape == (40, 50, 16) and (static[..., 13:16] == 0).all(), "a missing-data flag is set"
    np.savez(fine_path, static=static, lat_edges=lat_e, lon_edges=lon_e)
fz = np.load(fine_path); static, lat_e, lon_e = fz["static"], fz["lat_edges"], fz["lon_edges"]
print("SF fine static:", static.shape, "empty place layers:", int((np.expm1(static[..., :8]).sum(axis=(0, 1)) == 0).sum()), flush=True)

# ---------------- tract features (same recipe as bench77/region_features.py) ----------------
R, C = 40, 50
cells = gpd.GeoDataFrame({"cell": np.arange(R * C)}, geometry=[box(lon_e[j], lat_e[i], lon_e[j + 1], lat_e[i + 1])
                          for i in range(R) for j in range(C)], crs="EPSG:4326")
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
np.savez(f"{D}/region_static_sf.npz", X=X, rid=tracts["rid"].values, centroid_km=cen_km, area_km2=area / 1e6, cover=cover)
print(f"SF tract features {X.shape}, median tract area {np.median(area)/1e6:.2f} km2, min cover {cover.min():.2f}", flush=True)
print("done", flush=True)
