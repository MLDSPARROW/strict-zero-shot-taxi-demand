"""Approach B, step 1 (2026-10-01): fine-grid inputs for the change-of-support model. Map data only, no taxi data.
Per city: the 40x50 fine grid of 16 static map features (same grid as everywhere else), the fraction of every grid cell
lying inside every region (frac[r, c], exact polygon intersection in a local metric projection), cell area in km2, and
log(1 + FAA enplanements in millions) per cell (airport reference point inside the cell).
Region order = the order used by the demand arrays / region_static files.
Output: grid/data/grid_<city>.npz"""
import os, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box
HERE = os.path.dirname(os.path.abspath(__file__)); S = f"{HERE}/.."; os.makedirs(f"{HERE}/data", exist_ok=True)
B77, MC, S2 = f"{S}/bench77/data", f"{S}/multicity/data", f"{S}/step2/cache"
core = np.load(f"{S2}/static_core.npz")
def fine(c):
    if c in ("chicago", "nyc"):
        z = np.load(f"{S2}/{c}_dense.npz"); return core["STATIC_CHI" if c == "chicago" else "STATIC_NYC"], z["lat_edges"], z["lon_edges"]
    z = np.load(f"{MC}/{c}_fine_static.npz"); return z["static"], z["lat_edges"], z["lon_edges"]
def regions(c):
    if c == "chicago":
        g = gpd.read_file(f"{B77}/chicago_community_areas.geojson"); g["rid"] = g["area_numbe"].astype(int)
    elif c == "nyc":
        g = gpd.read_file(f"{S}/retune_out/taxi_zones_extracted/taxi_zones/taxi_zones.shp"); g["rid"] = g["LocationID"].astype(int); g = g.dissolve(by="rid", as_index=False)
    elif c == "sf":
        g = gpd.read_file(f"{MC}/sf_tracts_2020.geojson").to_crs(epsg=4326)
        cen = g.to_crs(g.estimate_utm_crs()).geometry.centroid.to_crs(epsg=4326)
        g = g[cen.x.values > -122.6].dissolve(by="name", as_index=False).sort_values("name").reset_index(drop=True); g["rid"] = np.arange(len(g))
    else:
        g = gpd.read_file(f"zip://{MC}/dc/tl_2020_11_tract.zip").to_crs(epsg=4326)
        g = g[g["ALAND"] > 0].sort_values("GEOID").reset_index(drop=True); g["rid"] = np.arange(len(g))
    return g.to_crs(epsg=4326).sort_values("rid").reset_index(drop=True)[["rid", "geometry"]]
ENPL = {"chicago": "ORD MDW", "nyc": "JFK LGA", "sf": "", "dc": ""}
ap = pd.read_csv(f"{S}/improve/raw/airports.csv", usecols=["iata_code", "latitude_deg", "longitude_deg"])
faa = pd.read_excel(f"{S}/improve/raw/cy21.xlsx").set_index("Locid")["CY 21 Enplanements"]
for c in ["chicago", "nyc", "sf", "dc"]:
    st, la, lo = fine(c); R, C = st.shape[:2]
    cells = gpd.GeoDataFrame({"cell": np.arange(R * C)}, geometry=[box(lo[j], la[i], lo[j + 1], la[i + 1]) for i in range(R) for j in range(C)], crs="EPSG:4326")
    reg = regions(c); utm = reg.estimate_utm_crs(); rp, cp = reg.to_crs(utm), cells.to_crs(utm)
    inter = gpd.overlay(cp, rp, how="intersection"); ca = cp.area.values
    frac = np.zeros((len(reg), R * C), np.float32); k = {r: i for i, r in enumerate(reg.rid)}
    np.add.at(frac, (inter.rid.map(k).values, inter.cell.values), (inter.area.values / ca[inter.cell.values]).astype(np.float32))
    enpl = np.zeros(R * C, np.float32)
    for code in ENPL[c].split():
        a = ap[ap.iata_code == code].iloc[0]; i = np.searchsorted(la, a.latitude_deg) - 1; j = np.searchsorted(lo, a.longitude_deg) - 1
        enpl[i * C + j] += faa[code] / 1e6
    np.savez(f"{HERE}/data/grid_{c}.npz", X=st.reshape(R * C, -1).astype(np.float32), frac=frac, cell_km2=(ca / 1e6).astype(np.float32),
             enpl=np.log1p(enpl), shape=np.array([R, C]))
    cov = frac.sum(0) > 0
    print(f"{c:8s}: {len(reg)} regions, {cov.sum()} of {R*C} cells touch a region, cell {np.median(ca)/1e6:.3f} km2, "
          f"region area check (sum frac x cell area / true area) median {np.median((frac*ca).sum(1)/rp.area.values):.3f}, airports in cells {int((enpl>0).sum())}", flush=True)
