"""PROTOCOL_SYDNEY.md: builds the interactive page of the Sydney estimate, results/sydney_page.html, from results/sydney_week.json (sydney_results.py), the suburb outlines of
multicity/data/sydney/ and improve/sydney_page_template.html. The outlines are simplified (60 m) and written as SVG paths in units of 20 m; no demand data is involved.
Usage (from improve/): python build_sydney_page.py [--partial]      (--partial: the test files written by sydney_results.py --partial; never published)"""
import os, sys, json, geopandas as gpd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/../results" + ("/partial" if "--partial" in sys.argv else "")
reg = gpd.read_file(f"{HERE}/../multicity/data/sydney/sydney_regions.geojson").set_crs(4326, allow_override=True).sort_values("rid").reset_index(drop=True)
g = reg.to_crs("EPSG:7856"); g["geometry"] = g.geometry.simplify(60, preserve_topology=True)
minx, miny, maxx, maxy = g.total_bounds; U = 20.0
ring = lambda c: "M" + "L".join(f"{round((x - minx) / U)},{round((maxy - y) / U)}" for x, y in c.coords[:-1]) + "Z"
paths = []
for geom in g.geometry:
    parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    paths.append("".join(ring(p.exterior) for p in parts if p.area > 1e4))
cen = g.geometry.representative_point()
data = json.load(open(f"{RES}/sydney_week.json"))
assert data["names"] == reg["LOC_NAME"].tolist(), "suburb order differs between the outlines and the results"
data.update({"paths": paths, "w": round((maxx - minx) / U), "h": round((maxy - miny) / U), "cx": [round((p.x - minx) / U) for p in cen], "cy": [round((maxy - p.y) / U) for p in cen]})
tpl = open(f"{HERE}/sydney_page_template.html", encoding="utf-8").read(); assert "/*DATA*/null" in tpl
html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
open(f"{RES}/sydney_page.html", "w", encoding="utf-8").write(html)
print(f"page: {len(html) / 1e6:.2f} MB, {len(paths)} suburbs, map {data['w']} x {data['h']} units, path characters {sum(map(len, paths)):,}", flush=True)
