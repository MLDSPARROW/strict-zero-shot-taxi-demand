"""PROTOCOL_CITIES100.md, keep rule: T = the first percentile of the density of OpenStreetMap places (the eight layers other than water, per km2)
over the 401 Sydney suburbs of PROTOCOL_SYDNEY.md, computed once from Sydney Overpass data with the code of prep_city.py, then used for every city.
Also reports (descriptive) how many of the localities that PROTOCOL_SYDNEY.md dropped lie below T. No demand data of any kind is read.
Writes miliroute/keep_rule.json and miliroute/keep_rule.md. Usage (from multicity/): python -I keep_rule_sydney.py"""
import json
import os
import sys
import time

import numpy as np
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prep_city as pc                       # noqa: E402
from city_regions import au_state            # noqa: E402

UTM = "EPSG:7856"                            # as prep_sydney.py
OUT = os.path.join(HERE, "..", "miliroute")

kept = gpd.read_file(f"{HERE}/data/sydney/sydney_regions.geojson").set_crs(4326, allow_override=True).sort_values("rid").reset_index(drop=True)
assert len(kept) == 401
centre_path = f"{HERE}/data/sydney/downtown.json"
if not os.path.exists(centre_path):          # the downtown point of prep_sydney.py (its downtown.json was not archived): same query, Nominatim
    import requests
    time.sleep(1.1)
    r = requests.get("https://nominatim.openstreetmap.org/search", params={"q": "Sydney Town Hall, Sydney, NSW, Australia", "format": "jsonv2", "limit": 1},
                     headers=pc.UA, timeout=60)
    r.raise_for_status(); h = r.json()[0]
    json.dump([float(h["lat"]), float(h["lon"])], open(centre_path, "w"))
dlat, dlon = json.load(open(centre_path))
nsw = au_state("NSW")
town = gpd.GeoSeries(gpd.points_from_xy([dlon], [dlat]), crs=4326).to_crs(UTM).iloc[0]
d = nsw.to_crs(UTM).geometry.centroid.distance(town) / 1000
cand = nsw[d.values <= 25.0].reset_index(drop=True)
kept_ids = set(kept["LOC_PID"])
dropped = cand[~cand["uid"].isin(kept_ids)].reset_index(drop=True)
missing = kept_ids - set(cand["uid"])
print(f"Sydney: {len(cand)} NSW localities with centroid within 25 km of ({dlat:.4f}, {dlon:.4f}); kept by PROTOCOL_SYDNEY {len(kept)}, "
      f"dropped {len(dropped)} ({', '.join(dropped['name'])}); kept but not found among the candidates now: {len(missing)}", flush=True)

minx, miny, maxx, maxy = cand.total_bounds
layers = pc.all_layers("sydney", (miny, minx, maxy, maxx))
dens_kept = pc.place_density(kept.to_crs(UTM), layers, UTM)
dens_drop = pc.place_density(dropped.to_crs(UTM), layers, UTM)
T = float(np.percentile(dens_kept, 1))
below = int((dens_drop < T).sum())
box_syd = pc.box_km2(kept)
rule = {"T": T, "T_definition": "first percentile (numpy default, linear) of places per km2 over the 401 Sydney suburbs; places = elements of the "
                                "eight notebook layers (Overpass, out center), each layer counted separately, inside the suburb polygon",
        "box_max_km2": 1900.0, "box_rule": "PROTOCOL_CITIES100.md addendum 1 (2026-10-24): 1,900 km2 = 0.95 km2 x 2,000 cells",
        "sydney_box_km2": box_syd, "dropped_by_protocol_sydney": len(dropped), "dropped_below_T": below,
        "computed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overpass": {name: json.load(open(f"{pc.RAW}/overpass/sydney/{name}.meta.json")) | {"elements": int(len(layers[name]))} for name, _ in pc.LAYERS}}
json.dump(rule, open(f"{OUT}/keep_rule.json", "w"), indent=1)
lines = [
    "# Keep rule of PROTOCOL_CITIES100.md: the threshold T, computed from Sydney", "",
    f"Computed {rule['computed_utc']} with `multicity/keep_rule_sydney.py` (Overpass data of Sydney downloaded the same day; no demand data).", "",
    f"- Places per km2 of the 401 Sydney suburbs: minimum {dens_kept.min():.1f}, first percentile **T = {T:.2f}**, median {np.median(dens_kept):.1f}, maximum {dens_kept.max():.0f}.",
    f"- Localities with centroid within 25 km of Sydney Town Hall ({dlat:.4f}, {dlon:.4f}): {len(cand)}; kept by PROTOCOL_SYDNEY.md: {len(kept)}; dropped: {len(dropped)}.",
    f"- Dropped localities and their density: " + "; ".join(f"{n} {v:.1f}" for n, v in zip(dropped['name'], dens_drop)) + ".",
    f"- Dropped localities below T: **{below} of {len(dropped)}** (descriptive).",
    f"- Box of the 401 Sydney suburbs, measured as the notebook measures its grid: {box_syd:.0f} km2 (the protocol text says 1,924).",
    f"- Box limit used for the cities: {rule['box_max_km2']:.0f} km2 (addendum 1 of PROTOCOL_CITIES100.md).", "",
    "Elements per layer (Sydney box): " + ", ".join(f"{k} {v['elements']:,}" for k, v in rule["overpass"].items()) + ".",
]
open(f"{OUT}/keep_rule.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
