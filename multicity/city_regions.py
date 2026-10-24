"""PROTOCOL_CITIES100.md, "Study area and regions": the official regions of each country, cut to a bounding box.
One reader per country; each returns a GeoDataFrame in EPSG:4326 with columns `uid` (stable id), `name` (the region name
the page shows) and `geometry`. The national or state boundary file is read once from ~/taxi-demand/raw/ (downloaded by
~/taxi-demand/scripts/download.py; listed in miliroute/sources.md) and cached as GeoParquet in ~/taxi-demand/cache/.
Downloaded files are untrusted data: they are only read (python -I), never executed.

  US  : ZIP Code Tabulation Areas 2020, cartographic boundary file 1:500k (US Census Bureau)  -> "ZIP 12345"
  UK  : Wards (December 2024), generalised, clipped to the coastline (ONS)                    -> ward name
  CA  : Forward sortation areas 2021, cartographic (Statistics Canada)                         -> "FSA M5V"
  AU  : Geoscape gazetted localities, GDA2020 (data.gov.au; as for Sydney)                     -> locality name
Residents and jobs are not read (display only and optional; null)."""
import glob
import os
import zipfile

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

RAW = os.path.expanduser("~/taxi-demand/raw")
CACHE = os.path.expanduser("~/taxi-demand/cache")
AU_STATE = {"NSW": "nsw", "VIC": "vic", "QLD": "qld", "WA": "wa", "SA": "sa", "ACT": "act", "TAS": "tas", "NT": "nt"}


def _cached(key, build):
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{key}.parquet"
    if not os.path.exists(path):
        g = build()
        g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
        g.to_parquet(path)
    return gpd.read_parquet(path)


def _zip_layer(zpath):
    shp = [n for n in zipfile.ZipFile(zpath).namelist() if n.lower().endswith(".shp")]
    assert len(shp) == 1, f"{zpath}: expected one shapefile, found {shp}"
    return gpd.read_file(f"zip://{zpath}!{shp[0]}")


def us_national():
    def build():
        g = _zip_layer(f"{RAW}/us_zcta/cb_2020_us_zcta520_500k.zip").to_crs(4326)
        return gpd.GeoDataFrame({"uid": g["ZCTA5CE20"].astype(str), "name": "ZIP " + g["ZCTA5CE20"].astype(str)},
                                geometry=g.geometry, crs=4326)
    return _cached("us_zcta2020", build)


def uk_national():
    def build():
        parts = [gpd.read_file(p) for p in sorted(glob.glob(f"{RAW}/uk_wards/wards_dec2024_bgc_*.geojson"))]
        g = pd.concat(parts, ignore_index=True).drop_duplicates("WD24CD").reset_index(drop=True)
        assert len(g) == 8396, f"UK wards: {len(g)} features, expected 8,396"
        return gpd.GeoDataFrame({"uid": g["WD24CD"], "name": g["WD24NM"], "district": g["LAD24NM"]},
                                geometry=g.geometry, crs=4326)
    return _cached("uk_wards_dec2024", build)


def ca_national():
    def build():
        z = glob.glob(f"{RAW}/ca_fsa/lfsa000*21a_e.zip")
        if not z:
            raise FileNotFoundError("Canada: the Statistics Canada FSA file is missing (statcan.gc.ca refuses this server)")
        g = _zip_layer(z[0]).to_crs(4326)
        return gpd.GeoDataFrame({"uid": g["CFSAUID"], "name": "FSA " + g["CFSAUID"]}, geometry=g.geometry, crs=4326)
    return _cached("ca_fsa2021", build)


def au_state(state):
    s = AU_STATE[state]

    def build():
        g = _zip_layer(f"{RAW}/au_localities/{s}_loc_gda2020.zip")
        g = g.set_crs(4326, allow_override=True)          # GDA2020 is within about 1.5 m of WGS84 (as for Sydney)
        return gpd.GeoDataFrame({"uid": g["LOC_PID"], "name": g["LOC_NAME"]}, geometry=g.geometry, crs=4326)
    return _cached(f"au_localities_{s}", build)


def national(country, region):
    if country == "United States":
        return us_national()
    if country == "United Kingdom":
        return uk_national()
    if country == "Canada":
        return ca_national()
    if country == "Australia":
        return au_state(region)
    raise ValueError(country)


def regions_in_box(country, region, bbox):
    """Regions that intersect bbox = (minx, miny, maxx, maxy) in degrees. For Australia the neighbouring state files are
    added when the box crosses a border (Gold Coast / Tweed, Canberra / Queanbeyan)."""
    sources = [national(country, region)]
    if country == "Australia":
        near = {"QLD": ["NSW"], "NSW": ["QLD", "ACT", "VIC"], "ACT": ["NSW"], "VIC": ["NSW", "SA"]}.get(region, [])
        sources += [au_state(s) for s in near]
    b = box(*bbox)
    out = pd.concat([g[g.intersects(b)] for g in sources], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=4326).drop_duplicates("uid").reset_index(drop=True)
    out["geometry"] = out.geometry.buffer(0)
    if "district" in out.columns:                         # UK: a ward name that appears twice in the box gets its district
        dup = out["name"].duplicated(keep=False)
        out.loc[dup, "name"] = out.loc[dup, "name"] + " (" + out.loc[dup, "district"] + ")"
    else:
        dup = out["name"].duplicated(keep=False)          # Australia: same locality name in two states
        if dup.any():
            out.loc[dup, "name"] = out.loc[dup, "name"] + " (" + out.loc[dup, "uid"].astype(str) + ")"
    return out
