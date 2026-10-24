"""PROTOCOL_SYDNEY.md: Sydney suburbs as a target for a demand ESTIMATE without any Sydney taxi data (an application; there is no Sydney demand file).
  stage regions : Geoscape gazetted localities (data.gov.au, GDA2020) whose centroid lies within 25 km of Sydney Town Hall and that hold at least 200 jobs + residents
                  per km2 (drops water and bushland localities); residents and jobs of 2021 from Transport for NSW TZP24 travel zones (ERP_2021, EMP_2021), moved to the suburbs by the area overlap of each
                  zone with the suburbs (each zone's overlap weights normalised to 1, as PROTOCOL_V2 addendum 1)
  stage features: exactly as prep_bike.py / prep_dc.py: the notebook's build_static_features() on a 50 x 40 grid over the suburbs (rows along latitude), place layers by
                  Overpass, suburb features area-weighted from the grid cells as in bench77/region_features.py; airport value of the suburb that contains the
                  Sydney Airport reference point
Downloaded files are read from data/sydney_raw/ (untrusted: run this script with python -I). Output: data/sydney/.
Usage (from multicity/): python -I prep_sydney.py regions      then      python -I prep_sydney.py features"""
import os, sys, json, re, time
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__)); RAW = f"{HERE}/data/sydney_raw"; OUT = f"{HERE}/data/sydney"; os.makedirs(OUT, exist_ok=True)
STAGE = sys.argv[1]; RADIUS_KM = 25.0; MIN_DENS = 200.0; UTM = "EPSG:7856"   # GDA2020 / MGA zone 56, metres; MIN_DENS: jobs + residents per km2 (water and bushland localities fall below)
DOWNTOWN = "Sydney Town Hall, Sydney, NSW, Australia"; AIRPORT_ARP = (-33.9461, 151.1772)   # Kingsford Smith reference point (lat, lon)
AIRPORT_ENPL_M = 22.2                                                         # millions of enplanements: half of the 44.4 million passengers of 2019 (typical year; one airport only, see PROTOCOL_SYDNEY.md)

if STAGE == "regions":
    sub = gpd.read_file(f"{RAW}/nsw_localities_box.geojson").set_crs(4326, allow_override=True)
    sp = sub.to_crs(UTM); sub["area_all_km2"] = sp.area / 1e6
    import osmnx as ox
    dpath = f"{OUT}/downtown.json"
    if not os.path.exists(dpath): json.dump(list(ox.geocode(DOWNTOWN)), open(dpath, "w"))
    dlat, dlon = json.load(open(dpath)); town = gpd.GeoSeries(gpd.points_from_xy([dlon], [dlat]), crs=4326).to_crs(UTM).iloc[0]
    sub["d_km"] = sp.geometry.centroid.distance(town) / 1000
    # jobs and residents of 2021 per travel zone (TZP24), moved to the localities by area overlap
    emp = pd.read_csv(f"{RAW}/tzp24_employment_by_tz.csv", encoding="utf-8-sig"); pop = pd.read_csv(f"{RAW}/tzp24_population_by_tz.csv", encoding="utf-8-sig")
    tbl = emp[["TZ_CODE21", "EMP_2021"]].merge(pop[["TZ_CODE21", "ERP_2021"]], on="TZ_CODE21", validate="one_to_one")
    tz = gpd.read_file(f"{RAW}/tz21/TZ21_TfNSW_geojson.json"); tz["tz21_code"] = tz["tz21_code"].astype(int)
    tz = tz.merge(tbl, left_on="tz21_code", right_on="TZ_CODE21", validate="one_to_one")
    tz = tz[tz.intersects(box(*sub.total_bounds))].reset_index(drop=True)
    tzp = tz.to_crs(UTM); tzp["geometry"] = tzp.geometry.buffer(0); sp["geometry"] = sp.geometry.buffer(0)
    inter = gpd.overlay(tzp[["tz21_code", "geometry"]], sp[["LOC_PID", "geometry"]], how="intersection", keep_geom_type=True)
    inter["a"] = inter.geometry.area; inter["w"] = inter["a"] / inter.groupby("tz21_code")["a"].transform("sum")
    inter = inter.merge(tz[["tz21_code", "EMP_2021", "ERP_2021"]], on="tz21_code")
    inter["jobs"], inter["res"] = inter["w"] * inter["EMP_2021"], inter["w"] * inter["ERP_2021"]
    agg = inter.groupby("LOC_PID")[["jobs", "res"]].sum()
    sub = sub.merge(agg, left_on="LOC_PID", right_index=True, how="left").fillna({"jobs": 0.0, "res": 0.0})
    kept_mass = (inter["jobs"].sum() / tz["EMP_2021"].sum(), inter["res"].sum() / tz["ERP_2021"].sum())
    dens = (sub["jobs"] + sub["res"]) / sub["area_all_km2"]
    study = sub[(sub["d_km"] <= RADIUS_KM) & (dens >= MIN_DENS)].sort_values("LOC_PID").reset_index(drop=True)
    study["rid"] = np.arange(len(study)); study["area_km2"] = study["area_all_km2"]
    drop_empty = sub[(sub["d_km"] <= RADIUS_KM) & (dens < MIN_DENS)]
    study[["rid", "LOC_PID", "LOC_NAME", "area_km2", "d_km", "jobs", "res", "geometry"]].to_file(f"{OUT}/sydney_regions.geojson", driver="GeoJSON")
    msg = (f"Sydney regions: {len(study)} suburbs within {RADIUS_KM:.0f} km of ({dlat:.4f}, {dlon:.4f}) with at least {MIN_DENS:.0f} jobs + residents per km2; "
           f"{len(drop_empty)} localities in the disc dropped for being below that ({', '.join(drop_empty.nlargest(8, 'area_all_km2')['LOC_NAME'])}); "
           f"median area {study['area_km2'].median():.3f} km2, total {study['area_km2'].sum():.0f} km2; jobs {study['jobs'].sum():,.0f}, residents {study['res'].sum():,.0f}; "
           f"zone mass allocated to localities in the box: jobs {100 * kept_mass[0]:.1f}%, residents {100 * kept_mass[1]:.1f}%")
    open(f"{OUT}/summary_regions.txt", "w").write(msg + "\n"); print(msg, flush=True)
    print("smallest:", ", ".join(study.nsmallest(5, "area_km2")["LOC_NAME"]), "| most jobs:", ", ".join(study.nlargest(5, "jobs")["LOC_NAME"]), "| most residents:", ", ".join(study.nlargest(5, "res")["LOC_NAME"]))
    sys.exit(0)

# ---------------- stage features ----------------
NB = json.load(open(os.path.join(HERE, "..", "kaggle-fix", "notebook.ipynb"), encoding="utf-8"))
reg = gpd.read_file(f"{OUT}/sydney_regions.geojson").set_crs(4326, allow_override=True).sort_values("rid").reset_index(drop=True)
utm = UTM; KEY = "sydney"; R, C = 50, 40                                     # 50 rows along latitude, 40 columns along longitude: cells of about 1.0 x 0.9 km (helper cells: 0.07 to 0.92 km2)
import osmnx as ox
fine_path = f"{OUT}/sydney_fine_static.npz"
minx, miny, maxx, maxy = reg.total_bounds
lat_e, lon_e = np.linspace(miny, maxy, R + 1), np.linspace(minx, maxx, C + 1)
if not os.path.exists(fine_path):
    ns = {"np": np, "ox": ox, "GRID_ROWS": R, "GRID_COLS": C}
    c2 = "".join(NB["cells"][2]["source"])
    exec(re.search(r"def assign_cell\(.*?(?=\ndef |\Z)", c2, re.S).group(0), ns)
    c17 = "".join(NB["cells"][17]["source"]).replace("!pip -q install osmnx xgboost", "")
    c17 = c17[:c17.index("# Querying OSM for 8 POI categories")] if "# Querying OSM for 8 POI categories" in c17 else c17
    exec(c17, ns)
    ns["DOWNTOWN_QUERY"][KEY] = DOWNTOWN; orig = ns["fetch_points"]
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
    static = ns["build_static_features"](KEY, lat_e, lon_e)
    assert static.shape == (R, C, 16) and (static[..., 13:16] == 0).all(), "a missing-data flag is set"
    np.savez(fine_path, static=static, lat_edges=lat_e, lon_edges=lon_e)
fz = np.load(fine_path); static = fz["static"]
print("sydney fine static:", static.shape, "empty place layers:", int((np.expm1(static[..., :8]).sum(axis=(0, 1)) == 0).sum()), flush=True)

# ---------------- suburb features (as prep_bike.py / bench77/region_features.py) ----------------
cells = gpd.GeoDataFrame({"cell": np.arange(R * C)}, geometry=[box(lon_e[j_], lat_e[i], lon_e[j_ + 1], lat_e[i + 1]) for i in range(R) for j_ in range(C)], crs="EPSG:4326")
reg_p, cells_p = reg[["rid", "geometry"]].to_crs(utm), cells.to_crs(utm)
reg_p["geometry"] = reg_p.geometry.buffer(0)
inter = gpd.overlay(cells_p, reg_p, how="intersection"); inter["a"] = inter.geometry.area
W = np.zeros((len(reg), R * C)); np.add.at(W, (inter["rid"].values, inter["cell"].values), inter["a"].values)
area = reg_p.geometry.area.values; cover = W.sum(axis=1) / area
ca = cells_p.area.values; frac = np.zeros((len(reg), R * C), np.float32)
np.add.at(frac, (inter["rid"].values, inter["cell"].values), (inter["a"].values / ca[inter["cell"].values]).astype(np.float32))
f = static.reshape(R * C, 16).astype(np.float64); Wn = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
dens = Wn @ np.expm1(f[:, :8]); other = Wn @ f[:, 8:]; busy = dens.sum(axis=1); props = dens / np.clip(busy[:, None], 1e-9, None)
mix = -np.sum(np.where(props > 0, props * np.log(props + 1e-9), 0.0), axis=1)
feats = np.concatenate([np.log1p(dens), other[:, 0:3], np.log1p(busy)[:, None], mix[:, None], other[:, 5:8]], axis=1)
cen_km = np.stack([reg_p.geometry.centroid.x.values, reg_p.geometry.centroid.y.values], axis=1) / 1000.0
nbr = np.argsort(np.linalg.norm(cen_km[:, None] - cen_km[None], axis=2), axis=1)[:, 1:7]
contrast = feats[:, 11] - feats[:, 11][nbr].mean(axis=1)
X = np.concatenate([feats, np.log(area / np.median(area))[:, None], contrast[:, None], np.clip(cover, 0, 1)[:, None]], axis=1).astype(np.float32)
np.savez(f"{HERE}/data/region_static_sydney.npz", X=X, rid=reg["rid"].values, centroid_km=cen_km, area_km2=area / 1e6, cover=cover)
# airport: the suburb that contains the reference point of Sydney Kingsford Smith (large airport inside the area, as the airport rule of PROTOCOL_MORE_CITIES.md)
apt = gpd.GeoSeries(gpd.points_from_xy([AIRPORT_ARP[1]], [AIRPORT_ARP[0]]), crs=4326).iloc[0]
hit = [i for i, g in enumerate(reg.geometry) if g.contains(apt)]
enpl = np.zeros(len(reg), np.float32)
for i in hit: enpl[i] = AIRPORT_ENPL_M
np.save(os.path.join(HERE, "..", "improve", "data", "enpl_sydney.npy"), enpl)
np.savez(os.path.join(HERE, "..", "improve", "data", "extra_sydney.npz"), jobs=reg["jobs"].values.astype(np.float64), res=reg["res"].values.astype(np.float64), rail=np.zeros(len(reg)))
np.savez(f"{HERE}/../grid/data/grid_sydney.npz", X=static.reshape(R * C, -1).astype(np.float32), frac=frac, cell_km2=(ca / 1e6).astype(np.float32), enpl=np.zeros(R * C, np.float32), shape=np.array([R, C]))
msg = (f"sydney: {len(reg)} suburbs, features {X.shape}, median suburb area {np.median(area) / 1e6:.3f} km2, cell area {np.median(ca) / 1e6:.3f} km2, min grid cover {cover.min():.2f}, "
       f"airport suburb: {', '.join(reg.loc[hit, 'LOC_NAME']) if hit else 'NONE FOUND'}")
open(f"{OUT}/summary_features.txt", "w").write(msg + "\n"); print(msg, flush=True); print("done", flush=True)
