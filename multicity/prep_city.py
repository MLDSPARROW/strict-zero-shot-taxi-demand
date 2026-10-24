"""PROTOCOL_CITIES100.md: the data of one city of multicity/cities100.csv, with NO taxi data of that city (copied from prep_sydney.py
and generalised; prep_sydney.py itself is unchanged).
  stage centre  : the OpenStreetMap (Nominatim) place centre of "<city>, <region>, <country>" -> data/<slug>/centre.json
  stage regions : official regions (city_regions.py) whose centroid lies within R km of the centre (R = 25, reduced by 1 km while the box of the
                  kept regions is larger than the limit of the protocol), kept if their density of OpenStreetMap places (eight layers, Overpass,
                  per km2) is at least T (miliroute/keep_rule.json, from Sydney); a region with an airport reference point is always kept
                  -> data/<slug>/<slug>_regions.geojson
  stage features: exactly as prep_sydney.py: the notebook's build_static_features() on a grid with 50 cells along the longer side and 40 along the
                  shorter one over the kept regions, region features area-weighted from the cells (bench77/region_features.py), airport values
                  from multicity/airports_cities100.csv -> data/region_static_<slug>.npz, ../improve/data/{enpl,extra}_<slug>.*, ../grid/data/grid_<slug>.npz
Overpass: one query at a time, 10 s between queries, a User-Agent with a contact address, back-off on 429/504, the main instance first. Raw responses
are kept in ~/taxi-demand/raw/overpass/<slug>/ until the city is exported, then deleted. Downloads are data only: run this with python -I.
Usage (from multicity/): python -I prep_city.py <slug> <centre|regions|features|all>"""
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from city_regions import regions_in_box   # noqa: E402

RAW = os.path.expanduser("~/taxi-demand/raw")
UA = {"User-Agent": "strict-zero-shot-taxi-demand research (github.com/MLDSPARROW/strict-zero-shot-taxi-demand; contact miladghalejughi@gmail.com)"}
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
R0_KM, BOX_MAX_KM2 = 25.0, None          # BOX_MAX_KM2 is read from keep_rule.json (protocol value, see the addenda of PROTOCOL_CITIES100.md)
COUNTRY_OF = {"United States": "United States", "United Kingdom": "United Kingdom", "Canada": "Canada", "Australia": "Australia"}
NB = json.load(open(os.path.join(HERE, "..", "kaggle-fix", "notebook.ipynb"), encoding="utf-8"))


def city_row(slug):
    df = pd.read_csv(f"{HERE}/cities100.csv")
    row = df[df["slug"] == slug]
    assert len(row) == 1, f"{slug} is not in cities100.csv"
    return row.iloc[0].to_dict()


def keep_rule():
    return json.load(open(os.path.join(HERE, "..", "miliroute", "keep_rule.json")))


def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


# ---------------- OpenStreetMap places: the notebook's eight groups + water ----------------
def notebook_ns(rows, cols):
    import osmnx as ox
    ns = {"np": np, "ox": ox, "GRID_ROWS": rows, "GRID_COLS": cols}
    c2 = "".join(NB["cells"][2]["source"])
    exec(re.search(r"def assign_cell\(.*?(?=\ndef |\Z)", c2, re.S).group(0), ns)
    c17 = "".join(NB["cells"][17]["source"]).replace("!pip -q install osmnx xgboost", "")
    c17 = c17[:c17.index("# Querying OSM for 8 POI categories")]
    exec(c17, ns)
    return ns


POI_GROUPS = notebook_ns(1, 1)["POI_GROUPS"]
LAYERS = list(POI_GROUPS.items()) + [("water", {"natural": "water"})]
_last_query = [0.0]


def overpass_query(tags, bbox):
    """The query of prep_sydney.py (node, way, relation; out center) for bbox = (south, west, north, east)."""
    s, w, n, e = bbox
    bb = f"({s},{w},{n},{e})"
    q = []
    for k, v in tags.items():
        vals = [v] if isinstance(v, str) else v
        flt = f'["{k}"]' if v is True else f'["{k}"~"^({"|".join(vals)})$"]'
        q += [f"node{flt}{bb};", f"way{flt}{bb};", f"relation{flt}{bb};"]
    return f"[out:json][timeout:300];({''.join(q)});out center;"


def fetch_layer(slug, name, tags, bbox):
    """Points (lat, lon) of one layer in bbox; the raw response is kept in RAW/overpass/<slug>/ and reused."""
    folder = f"{RAW}/overpass/{slug}"
    os.makedirs(folder, exist_ok=True)
    raw_path = f"{folder}/{name}.json"
    meta_path = f"{folder}/{name}.meta.json"
    if not os.path.exists(raw_path):
        for attempt in range(12):
            url = OVERPASS[0] if attempt < 8 else OVERPASS[1 + (attempt % 2)]
            tile = 1 if attempt < 4 else 2                   # attempts 0-3: one query; later: the box in four tiles (as prep_sydney.py), merged
            try:
                south, west, north, east = bbox
                lat_e, lon_e = np.linspace(south, north, tile + 1), np.linspace(west, east, tile + 1)
                elements, seen, nbytes = [], set(), 0
                for ti in range(tile):
                    for tj in range(tile):
                        wait = 10 - (time.time() - _last_query[0])
                        if wait > 0:
                            time.sleep(wait)
                        _last_query[0] = time.time()
                        sub = (float(lat_e[ti]), float(lon_e[tj]), float(lat_e[ti + 1]), float(lon_e[tj + 1]))
                        r = requests.post(url, data={"data": overpass_query(tags, sub)}, headers=UA, timeout=420)
                        if r.status_code in (429, 504):
                            raise RuntimeError(f"HTTP {r.status_code}")
                        r.raise_for_status()
                        nbytes += len(r.content)
                        for el in r.json()["elements"]:
                            key = (el.get("type"), el.get("id"))
                            if key not in seen:
                                seen.add(key)
                                elements.append(el)
                body = json.dumps({"elements": elements}).encode()
                open(raw_path + ".part", "wb").write(body)
                os.replace(raw_path + ".part", raw_path)
                json.dump({"url": url, "tile": tile, "query_layer": name, "bbox": bbox, "bytes": nbytes, "elements": len(elements),
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(meta_path, "w"))
                with open(f"{RAW}/sources.jsonl", "a") as f:
                    f.write(json.dumps({"source": f"overpass/{slug}", "url": url, "file": raw_path, "bytes": nbytes,
                                        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "licence": "OpenStreetMap contributors, ODbL 1.0", "query_layer": name}) + "\n")
                print(f"  {slug}/{name}: {len(elements)} elements, {nbytes / 1e6:.1f} MB from {url} (tile={tile})", flush=True)
                break
            except Exception as e:
                pause = 60 * min(attempt + 1, 5)
                print(f"  {slug}/{name}: attempt {attempt + 1} failed ({type(e).__name__}: {str(e)[:100]}); waiting {pause}s", flush=True)
                time.sleep(pause)
        else:
            raise RuntimeError(f"Overpass layer {name} of {slug} failed after retries")
    el = json.load(open(raw_path))["elements"]
    pts = [(x.get("lat", x.get("center", {}).get("lat")), x.get("lon", x.get("center", {}).get("lon"))) for x in el]
    return np.array([p for p in pts if p[0] is not None], dtype=np.float32).reshape(-1, 2)


def all_layers(slug, bbox):
    return {name: fetch_layer(slug, name, tags, bbox) for name, tags in LAYERS}


def place_density(regions_m, layers, utm):
    """Places of the eight layers (water excluded) per km2 of each region; every layer is counted separately, as the features do."""
    pts = np.concatenate([layers[name] for name, _ in POI_GROUPS.items()], axis=0)
    g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts[:, 1], pts[:, 0]), crs=4326).to_crs(utm)
    j = gpd.sjoin(g, regions_m[["geometry"]], how="inner", predicate="within")
    counts = np.bincount(j["index_right"].values, minlength=len(regions_m))
    return counts / (regions_m.geometry.area.values / 1e6)


# ---------------- airports ----------------
def airports(slug):
    path = os.path.join(HERE, "airports_cities100.csv")
    a = pd.read_csv(path)
    return a[a["slug"] == slug]


# ---------------- stages ----------------
def stage_centre(slug):
    row = city_row(slug)
    out = f"{HERE}/data/{slug}"
    os.makedirs(out, exist_ok=True)
    path = f"{out}/centre.json"
    if os.path.exists(path):
        return json.load(open(path))
    q = f"{row['city']}, {row['region']}, {row['country']}"
    time.sleep(1.1)
    r = requests.get("https://nominatim.openstreetmap.org/search", params={"q": q, "format": "jsonv2", "limit": 1},
                     headers=UA, timeout=60)
    r.raise_for_status()
    hits = r.json()
    assert hits, f"Nominatim found nothing for {q!r}"
    h = hits[0]
    c = {"query": q, "lat": float(h["lat"]), "lon": float(h["lon"]), "display_name": h["display_name"],
         "osm_type": h.get("osm_type"), "osm_id": h.get("osm_id"), "category": h.get("category"), "type": h.get("type"),
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "licence": "Data (c) OpenStreetMap contributors, ODbL 1.0"}
    json.dump(c, open(path, "w"), indent=1)
    print(f"{slug}: centre {c['lat']:.5f}, {c['lon']:.5f} ({c['display_name']})", flush=True)
    return c


def box_km2(g4326):
    """Area of the latitude/longitude box of the regions, measured as the notebook measures its grid (grid_geometry)."""
    minx, miny, maxx, maxy = g4326.total_bounds
    mid_lat, mid_lon = (miny + maxy) / 2, (minx + maxx) / 2
    hav = notebook_ns(1, 1)["haversine_km"]
    return float(hav(miny, mid_lon, maxy, mid_lon) * hav(mid_lat, minx, mid_lat, maxx))


def stage_regions(slug):
    row, c, kr = city_row(slug), stage_centre(slug), keep_rule()
    T, box_max = kr["T"], kr["box_max_km2"]
    utm = utm_epsg(c["lat"], c["lon"])
    dlat = (R0_KM + 15) / 111.0
    dlon = dlat / max(0.2, math.cos(math.radians(c["lat"])))
    cand = regions_in_box(row["country"], row["region"], (c["lon"] - dlon, c["lat"] - dlat, c["lon"] + dlon, c["lat"] + dlat))
    cm = cand.to_crs(utm)
    centre = gpd.GeoSeries(gpd.points_from_xy([c["lon"]], [c["lat"]]), crs=4326).to_crs(utm).iloc[0]
    cand["d_km"] = cm.geometry.centroid.distance(centre).values / 1000
    cand = cand[cand["d_km"] <= R0_KM].reset_index(drop=True)
    cm = cand.to_crs(utm)
    minx, miny, maxx, maxy = cand.total_bounds
    layers = all_layers(slug, (miny, minx, maxy, maxx))
    cand["density"] = place_density(cm, layers, utm)
    cand["area_km2"] = cm.geometry.area.values / 1e6
    ap = airports(slug)
    cand["airport"] = False
    for _, a in ap.iterrows():                       # addendum 4: only an airport with a value above 0 (at least 5 million enplanements) pins its region
        if float(a["enplanements_m"]) < 5.0:
            continue
        p = gpd.GeoSeries(gpd.points_from_xy([a["lon"]], [a["lat"]]), crs=4326).iloc[0]
        cand.loc[cand.geometry.contains(p), "airport"] = True
    R = R0_KM
    while True:
        kept = cand[((cand["d_km"] <= R) & (cand["density"] >= T)) | cand["airport"]]     # an airport region is always kept (addendum 4), whatever the radius
        if box_km2(kept) <= box_max or R <= 5:
            break
        R -= 1.0
    kept = kept.sort_values("uid").reset_index(drop=True)
    kept["rid"] = np.arange(len(kept))
    dropped = cand[(cand["d_km"] <= R) & ~cand.index.isin(kept.index) & (cand["density"] < T) & ~cand["airport"]]
    out = f"{HERE}/data/{slug}"
    kept[["rid", "uid", "name", "area_km2", "d_km", "density", "airport", "geometry"]].to_file(f"{out}/{slug}_regions.geojson", driver="GeoJSON")
    info = {"slug": slug, "radius_km": R, "candidates_within_25km": int((cand["d_km"] <= R0_KM).sum()), "kept": len(kept),
            "dropped_below_T": int(((cand["d_km"] <= R) & (cand["density"] < T) & ~cand["airport"]).sum()),
            "box_km2": box_km2(kept), "median_area_km2": float(kept["area_km2"].median()), "T": T, "utm": utm,
            "airport_regions": kept.loc[kept["airport"], "name"].tolist(),
            "largest_dropped": cand[(cand["d_km"] <= R) & (cand["density"] < T) & ~cand["airport"]].nlargest(6, "area_km2")["name"].tolist()}
    json.dump(info, open(f"{out}/summary_regions.json", "w"), indent=1)
    print(f"{slug}: {info['kept']} regions kept within {R:.0f} km ({info['dropped_below_T']} below T dropped), box {info['box_km2']:.0f} km2, "
          f"median area {info['median_area_km2']:.2f} km2, airport regions {info['airport_regions']}", flush=True)
    return info


def stage_features(slug):
    c = stage_centre(slug)
    out = f"{HERE}/data/{slug}"
    reg = gpd.read_file(f"{out}/{slug}_regions.geojson").set_crs(4326, allow_override=True).sort_values("rid").reset_index(drop=True)
    info = json.load(open(f"{out}/summary_regions.json"))
    utm = info["utm"]
    minx, miny, maxx, maxy = reg.total_bounds
    rm = reg.to_crs(utm)
    b = rm.total_bounds
    tall = (b[3] - b[1]) >= (b[2] - b[0])
    R, C = (50, 40) if tall else (40, 50)                  # rows along latitude, columns along longitude; 50 cells along the longer side
    lat_e, lon_e = np.linspace(miny, maxy, R + 1), np.linspace(minx, maxx, C + 1)
    fine_path = f"{out}/{slug}_fine_static.npz"
    if os.path.exists(fine_path):                          # a cached grid is only valid for the same box and grid shape
        old = np.load(fine_path)
        if old["static"].shape[:2] != (R, C) or not (np.allclose(old["lat_edges"], lat_e) and np.allclose(old["lon_edges"], lon_e)):
            os.remove(fine_path)
    if not os.path.exists(fine_path):
        ns = notebook_ns(R, C)
        cache = all_layers(slug, None)                     # the layers fetched by stage regions (their box contains this one)

        def fetch_points(north, south, east, west, tags, label):
            name = label.split("/")[-1]
            p = cache[name]
            keep = (p[:, 0] >= south) & (p[:, 0] <= north) & (p[:, 1] >= west) & (p[:, 1] <= east)
            print(f"  {label}: {int(keep.sum())} features in the grid box", flush=True)
            return p[keep]

        class _Geo:                                         # the city centre of stage centre, not a second Nominatim query
            @staticmethod
            def geocode(_q):
                return c["lat"], c["lon"]
        ns["fetch_points"] = fetch_points
        ns["ox"] = _Geo
        ns["DOWNTOWN_QUERY"][slug] = "centre.json"
        static = ns["build_static_features"](slug, lat_e, lon_e)
        assert static.shape == (R, C, 16) and (static[..., 13:16] == 0).all(), "a missing-data flag is set"
        np.savez(fine_path, static=static, lat_edges=lat_e, lon_edges=lon_e)
    fz = np.load(fine_path)
    static = fz["static"]
    empty = int((np.expm1(static[..., :8]).sum(axis=(0, 1)) == 0).sum())
    # ---- region features (as prep_sydney.py) ----
    cells = gpd.GeoDataFrame({"cell": np.arange(R * C)}, geometry=[box(lon_e[j_], lat_e[i], lon_e[j_ + 1], lat_e[i + 1]) for i in range(R) for j_ in range(C)], crs="EPSG:4326")
    reg_p, cells_p = reg[["rid", "geometry"]].to_crs(utm), cells.to_crs(utm)
    reg_p["geometry"] = reg_p.geometry.buffer(0)
    inter = gpd.overlay(cells_p, reg_p, how="intersection")
    inter["a"] = inter.geometry.area
    W = np.zeros((len(reg), R * C))
    np.add.at(W, (inter["rid"].values, inter["cell"].values), inter["a"].values)
    area = reg_p.geometry.area.values
    cover = W.sum(axis=1) / area
    ca = cells_p.area.values
    frac = np.zeros((len(reg), R * C), np.float32)
    np.add.at(frac, (inter["rid"].values, inter["cell"].values), (inter["a"].values / ca[inter["cell"].values]).astype(np.float32))
    f = static.reshape(R * C, 16).astype(np.float64)
    Wn = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
    dens = Wn @ np.expm1(f[:, :8])
    other = Wn @ f[:, 8:]
    busy = dens.sum(axis=1)
    props = dens / np.clip(busy[:, None], 1e-9, None)
    mix = -np.sum(np.where(props > 0, props * np.log(props + 1e-9), 0.0), axis=1)
    feats = np.concatenate([np.log1p(dens), other[:, 0:3], np.log1p(busy)[:, None], mix[:, None], other[:, 5:8]], axis=1)
    cen_km = np.stack([reg_p.geometry.centroid.x.values, reg_p.geometry.centroid.y.values], axis=1) / 1000.0
    nbr = np.argsort(np.linalg.norm(cen_km[:, None] - cen_km[None], axis=2), axis=1)[:, 1:7]
    contrast = feats[:, 11] - feats[:, 11][nbr].mean(axis=1)
    X = np.concatenate([feats, np.log(area / np.median(area))[:, None], contrast[:, None], np.clip(cover, 0, 1)[:, None]], axis=1).astype(np.float32)
    np.savez(f"{HERE}/data/region_static_{slug}.npz", X=X, rid=reg["rid"].values, centroid_km=cen_km, area_km2=area / 1e6, cover=cover)
    # ---- airports (multicity/airports_cities100.csv; a value below 5 million enplanements is 0 by the protocol) ----
    enpl = np.zeros(len(reg), np.float32)
    hits = []
    for _, a in airports(slug).iterrows():
        p = gpd.GeoSeries(gpd.points_from_xy([a["lon"]], [a["lat"]]), crs=4326).iloc[0]
        for i, g in enumerate(reg.geometry):
            if g.contains(p):
                v = float(a["enplanements_m"]) if float(a["enplanements_m"]) >= 5.0 else 0.0
                enpl[i] += v
                hits.append((a["iata"], reg.loc[i, "name"], v))
    np.save(os.path.join(HERE, "..", "improve", "data", f"enpl_{slug}.npy"), enpl)
    np.savez(os.path.join(HERE, "..", "improve", "data", f"extra_{slug}.npz"), jobs=np.zeros(len(reg)), res=np.zeros(len(reg)), rail=np.zeros(len(reg)))
    np.savez(f"{HERE}/../grid/data/grid_{slug}.npz", X=static.reshape(R * C, -1).astype(np.float32), frac=frac, cell_km2=(ca / 1e6).astype(np.float32),
             enpl=np.zeros(R * C, np.float32), shape=np.array([R, C]))
    summ = {"slug": slug, "regions": len(reg), "grid": [R, C], "cell_km2_median": float(np.median(ca) / 1e6), "min_cover": float(cover.min()),
            "empty_place_layers": empty, "airports": hits, "median_area_km2": float(np.median(area) / 1e6)}
    json.dump(summ, open(f"{out}/summary_features.json", "w"), indent=1)
    print(f"{slug}: features {X.shape}, grid {R}x{C}, cell {summ['cell_km2_median']:.3f} km2, min cover {cover.min():.2f}, "
          f"empty layers {empty}, airports {hits}", flush=True)
    return summ


if __name__ == "__main__":
    slug, stage = sys.argv[1], sys.argv[2]
    if stage in ("centre", "all"):
        stage_centre(slug)
    if stage in ("regions", "all"):
        stage_regions(slug)
    if stage in ("features", "all"):
        stage_features(slug)
