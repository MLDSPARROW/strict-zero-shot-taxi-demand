"""Rank-1 static covariates per region (NO taxi demand used anywhere):
  jobs   : LODES8 2021 WAC total jobs (C000) by workplace tract  -> log1p(jobs per km2)
  resid  : LODES8 2021 RAC total workers by home tract (resident proxy) -> log1p(residents per km2)
  rail   : OSM railway=station nodes (heavy/subway/commuter rail stations) -> log1p(stations per km2)
  airport: OurAirports large_airport with scheduled service whose point lies in the region AND whose listed
           municipality is the city itself (taxi jurisdiction rule; excludes Newark EWR for NYC -- NOTE this rule was
           chosen after seeing that EWR has ~0 yellow-taxi pickups, flagged as a post-hoc domain choice)
Tract totals are spread to regions by land-area overlap (cartographic-boundary tracts, water clipped).
Output: improve/data/extra_<city>.npz  (E: regions x 4, names)"""
import os, gzip, numpy as np, pandas as pd, geopandas as gpd, requests
from shapely.geometry import Point
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = f"{HERE}/raw"; OUT = f"{HERE}/data"; os.makedirs(OUT, exist_ok=True)
B77 = f"{HERE}/../bench77/data"; MC = f"{HERE}/../multicity/data"
ST = {"austin": ("tx", "48", "Austin"), "chicago": ("il", "17", "Chicago"), "nyc": ("ny", "36", "New York"), "sf": ("ca", "06", "San Francisco"), "dc": ("dc", "11", "Washington")}

def regions(city):
    if city == "chicago":
        g = gpd.read_file(f"{B77}/chicago_community_areas.geojson"); g["rid"] = g["area_numbe"].astype(int)
    elif city == "nyc":
        g = gpd.read_file(f"{HERE}/../retune_out/taxi_zones_extracted/taxi_zones/taxi_zones.shp")
        g["rid"] = g["LocationID"].astype(int); g = g.dissolve(by="rid", as_index=False)
    elif city == "sf":
        g = gpd.read_file(f"{MC}/sf_tracts_2020.geojson").to_crs(epsg=4326)
        cen = g.to_crs(g.estimate_utm_crs()).geometry.centroid.to_crs(epsg=4326)
        g = g[cen.x.values > -122.6].dissolve(by="name", as_index=False).sort_values("name").reset_index(drop=True)
        g["rid"] = np.arange(len(g))
    elif city == "austin":
        g = gpd.read_file(f"{MC}/austin_regions.geojson")
    else:
        g = gpd.read_file(f"zip://{MC}/dc/tl_2020_11_tract.zip").to_crs(epsg=4326)
        g = g[g["ALAND"] > 0].sort_values("GEOID").reset_index(drop=True); g["rid"] = np.arange(len(g))
    return g.to_crs(epsg=4326).sort_values("rid").reset_index(drop=True)[["rid", "geometry"]]

def lodes(st, kind, col):
    df = pd.read_csv(f"{RAW}/{st}_{kind}_S000_JT00_2021.csv.gz", usecols=[col, "C000"], dtype={col: str})
    return df.groupby(df[col].str[:11])["C000"].sum()

ap = pd.read_csv(f"{RAW}/airports.csv", usecols=["type", "name", "iata_code", "latitude_deg", "longitude_deg", "municipality", "scheduled_service", "iso_country"])
ap = ap[(ap.type == "large_airport") & (ap.scheduled_service == "yes") & (ap.iso_country == "US")]

def rail(bounds):
    s, w, n, e = bounds[1], bounds[0], bounds[3], bounds[2]
    q = f'[out:json][timeout:180];node["railway"="station"]({s},{w},{n},{e});out;'
    import time
    for url in ["https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass-api.de/api/interpreter"] * 3:
        try:
            r = requests.post(url, data={"data": q}, timeout=240, headers={"User-Agent": "urbanmind-research/1.0", "Accept": "*/*"}); r.raise_for_status()
            el = r.json()["elements"]; return gpd.GeoDataFrame(geometry=[Point(x["lon"], x["lat"]) for x in el], crs="EPSG:4326")
        except Exception as ex:
            print("  overpass fail", url, ex); time.sleep(20)
    raise RuntimeError("rail fetch failed")

import sys
for city, (st, fips, muni) in ST.items():
    if os.path.exists(f"{OUT}/extra_{city}.npz"): continue
    reg = regions(city); utm = reg.estimate_utm_crs(); regp = reg.to_crs(utm); area = regp.area.values / 1e6
    tr = gpd.read_file(f"zip://{RAW}/cb_2021_{fips}_tract_500k.zip").to_crs(utm)
    tr["jobs"] = tr["GEOID"].map(lodes(st, "wac", "w_geocode")).fillna(0)
    tr["res"] = tr["GEOID"].map(lodes(st, "rac", "h_geocode")).fillna(0)
    tr["tarea"] = tr.area
    inter = gpd.overlay(tr[["GEOID", "jobs", "res", "tarea", "geometry"]], regp, how="intersection")
    w = inter.area / inter["tarea"]
    jobs = np.bincount(inter["rid"].map({r: k for k, r in enumerate(reg.rid)}).values, weights=inter["jobs"] * w, minlength=len(reg))
    res = np.bincount(inter["rid"].map({r: k for k, r in enumerate(reg.rid)}).values, weights=inter["res"] * w, minlength=len(reg))
    stn = rail(reg.total_bounds).to_crs(utm)
    ns = np.bincount(gpd.sjoin(stn, regp, predicate="within")["rid"].map({r: k for k, r in enumerate(reg.rid)}).values, minlength=len(reg))
    a = ap[ap.municipality == muni]; apg = gpd.GeoDataFrame(a, geometry=gpd.points_from_xy(a.longitude_deg, a.latitude_deg), crs="EPSG:4326").to_crs(utm)
    hit = gpd.sjoin(apg, regp, predicate="within")
    air = np.zeros(len(reg)); air[hit["rid"].map({r: k for k, r in enumerate(reg.rid)}).values] = 1
    E = np.stack([np.log1p(jobs / area), np.log1p(res / area), np.log1p(ns / area), air], 1).astype(np.float32)
    np.savez(f"{OUT}/extra_{city}.npz", E=E, names=np.array(["log_jobs_km2", "log_resid_km2", "log_rail_km2", "airport"]), jobs=jobs, res=res, rail=ns)
    print(f"{city}: {len(reg)} regions | jobs {jobs.sum():,.0f} resid {res.sum():,.0f} rail stations {ns.sum()} | airports in: "
          f"{[(int(reg.rid[k]), n) for k, n in zip(hit['rid'].map({r: k for k, r in enumerate(reg.rid)}).values, hit['iata_code'])]}", flush=True)
