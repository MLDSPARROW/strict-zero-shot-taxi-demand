"""PROTOCOL_CITIES100.md + miliroute/JOB1_COMPUTE.md section 7: from the allocator runs of one city to the files the website reads.
  shares  : equal average of the eh and eo ensembles (each run normalised per slot), exactly as sydney_results.py
  rhythm  : the mean weekday x half-hour profile of Chicago, NYC and SF is the same for every city (PROTOCOL_CITIES100.md); it and the day counts nd are
            read from results/sydney_week.json (identical by construction: same helpers, same calendar year 2023)
  checks  : 1 to 8 of the protocol (descriptive; none is an accuracy test). A city that fails one is exported as status "draft" with the reason.
  output  : miliroute/export/cities/<country_slug>/<slug>.json, <slug>_areas.csv, <slug>_checks.txt; miliroute/export/_meta/<slug>.json;
            miliroute/export/cities.csv (rebuilt from all _meta files); miliroute/export/_pics/<slug>.png (one picture to look at, check 7)
  cleanup : only after a published or draft export is written, the per-run arrays of the city are deleted (--keep to skip)
Sydney (`python -I export_city.py sydney-nsw`) is not recomputed: it is converted from results/sydney_* (the laptop's 20 runs).
Usage (from improve/): python -I export_city.py <slug> [--runs 10] [--keep]"""
import csv
import json
import math
import os
import shutil
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..")
EXPORT = f"{ROOT}/miliroute/export"
MC = f"{ROOT}/multicity"
START = date(2023, 1, 1)
MODEL_VERSION = "v1"
UA = {"User-Agent": "strict-zero-shot-taxi-demand research (github.com/MLDSPARROW/strict-zero-shot-taxi-demand; contact miladghalejughi@gmail.com)"}

COUNTRY_SLUG = {"United States": "united-states", "United Kingdom": "united-kingdom", "Canada": "canada", "Australia": "australia"}
AREA_LABEL = {"United States": "ZIP code area", "United Kingdom": "ward", "Canada": "postal-code area", "Australia": "suburb"}
LOCALE = {"United States": "en-US", "United Kingdom": "en-GB", "Canada": "en-CA", "Australia": "en-AU"}
REGION_NAME = {"England": "England", "Scotland": "Scotland", "Wales": "Wales", "Northern Ireland": "Northern Ireland"}
ATTRIBUTION = {
    "United States": "US Census Bureau, ZIP Code Tabulation Areas 2020, cartographic boundary file (public domain); airport boardings: FAA calendar year 2019 enplanements (public domain)",
    "United Kingdom": "Contains public sector information licensed under the Open Government Licence v3.0. Wards (December 2024) Boundaries UK BGC: Office for National Statistics; contains OS data © Crown copyright and database right 2024; airport passengers: UK Civil Aviation Authority, 2019",
    "Canada": "Statistics Canada, Forward Sortation Area Boundary File, 2021 (Open Government Licence - Canada); airport passengers 2019 from each airport authority's published statistics",
    "Australia": "Suburb/locality boundaries: Geoscape Administrative Boundaries (Geoscape Australia and the Department of Industry, Science and Resources), CC BY 4.0; airport passengers: BITRE and airport reports, 2019",
}
TZ_STATE = {"CT": "America/New_York", "DE": "America/New_York", "FL": "America/New_York", "GA": "America/New_York", "MA": "America/New_York", "MD": "America/New_York",
            "NC": "America/New_York", "NY": "America/New_York", "OH": "America/New_York", "PA": "America/New_York", "RI": "America/New_York", "VA": "America/New_York",
            "AL": "America/Chicago", "IL": "America/Chicago", "LA": "America/Chicago", "MN": "America/Chicago", "MO": "America/Chicago", "NE": "America/Chicago",
            "OK": "America/Chicago", "TX": "America/Chicago", "WI": "America/Chicago", "CO": "America/Denver", "NM": "America/Denver", "UT": "America/Denver",
            "AZ": "America/Phoenix", "CA": "America/Los_Angeles", "NV": "America/Los_Angeles", "OR": "America/Los_Angeles", "WA": "America/Los_Angeles",
            "HI": "Pacific/Honolulu", "MI": "America/Detroit", "IN": "America/Indiana/Indianapolis", "KY": "America/Kentucky/Louisville", "TN": "America/Chicago"}
TZ_SLUG = {"el-paso-tx": "America/Denver", "nashville-tn": "America/Chicago", "memphis-tn": "America/Chicago", "ottawa-on": "America/Toronto"}
TZ_OTHER = {"ON": "America/Toronto", "QC": "America/Toronto", "BC": "America/Vancouver", "AB": "America/Edmonton", "MB": "America/Winnipeg", "NS": "America/Halifax",
            "SK": "America/Regina", "NSW": "Australia/Sydney", "ACT": "Australia/Sydney", "VIC": "Australia/Melbourne", "QLD": "Australia/Brisbane",
            "WA": "Australia/Perth", "SA": "Australia/Adelaide", "TAS": "Australia/Hobart", "NT": "Australia/Darwin"}
THRESH_RUN_TO_RUN = 0.56            # PROTOCOL_CITIES100.md, check 5 as changed by addendum 1


def city_row(slug):
    df = pd.read_csv(f"{MC}/cities100.csv")
    return df[df["slug"] == slug].iloc[0].to_dict()


def timezone(r):
    if r["country"] == "United Kingdom":
        return "Europe/London"
    if r["slug"] in TZ_SLUG:
        return TZ_SLUG[r["slug"]]
    if r["country"] == "United States":
        return TZ_STATE[r["region"]]
    return TZ_OTHER[r["region"]]


def region_display(r):
    return r["region"]


def run_files(slug, runs, srcs_fn):
    return {(rr, h): f"{HERE}/out_cal/multi_{slug}_from_{'+'.join(srcs_fn(slug, rr))}_v{h}_seed{rr - 1}.npy" for rr in runs for h in ("eh", "eo")}


# ---------------------------------------------------------------- map paths (build_sydney_page.py, generalised)
def map_paths(reg, utm):
    g = reg.to_crs(utm)
    g["geometry"] = g.geometry.buffer(0).simplify(60, preserve_topology=True)
    minx, miny, maxx, maxy = g.total_bounds
    U = 20.0
    ring = lambda c: "M" + "L".join(f"{round((x - minx) / U)},{round((maxy - y) / U)}" for x, y in c.coords[:-1]) + "Z"
    paths = []
    for geom in g.geometry:
        parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        s = "".join(ring(p.exterior) for p in parts if p.area > 1e4)
        if not s:                                          # a region smaller than 0.01 km2 after simplification: keep its largest part so that the map has no hole
            big = max(parts, key=lambda p: p.area)
            s = ring(big.exterior)
        paths.append(s)
    cen = g.geometry.representative_point()
    return {"paths": paths, "w": round((maxx - minx) / U), "h": round((maxy - miny) / U), "cx": [round((p.x - minx) / U) for p in cen],
            "cy": [round((maxy - p.y) / U) for p in cen]}, g


def wikidata_id(name, lat, lon):
    """Wikidata item of the city, accepted only if its coordinate lies within 30 km of the city's centre and it is a city-like place."""
    import requests
    try:
        s = requests.get("https://www.wikidata.org/w/api.php", params={"action": "wbsearchentities", "search": name, "language": "en", "limit": 8, "format": "json"},
                         headers=UA, timeout=60).json().get("search", [])
        ids = [x["id"] for x in s]
        if not ids:
            return ""
        time.sleep(1)
        ents = requests.get("https://www.wikidata.org/w/api.php", params={"action": "wbgetentities", "ids": "|".join(ids), "props": "claims|descriptions", "languages": "en", "format": "json"},
                            headers=UA, timeout=60).json()["entities"]
        for i in ids:
            e = ents.get(i, {})
            cl = e.get("claims", {}).get("P625")
            if not cl:
                continue
            v = cl[0]["mainsnak"]["datavalue"]["value"]
            d = 6371 * 2 * math.asin(math.sqrt(math.sin(math.radians(v["latitude"] - lat) / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(v["latitude"])) * math.sin(math.radians(v["longitude"] - lon) / 2) ** 2))
            desc = e.get("descriptions", {}).get("en", {}).get("value", "").lower()
            if d <= 30 and any(w in desc for w in ("city", "town", "municipality", "metropolitan", "borough", "capital", "urban")):
                return i
    except Exception as ex:
        print(f"  wikidata lookup failed: {type(ex).__name__}", flush=True)
    return ""


# ---------------------------------------------------------------- checks
def pooled_feature_range():
    cols = []
    for p in (f"{ROOT}/bench77/data/region_static_chicago.npz", f"{ROOT}/bench77/data/region_static_nyc.npz",
              f"{MC}/data/region_static_sf.npz", f"{MC}/data/region_static_dc.npz"):
        cols.append(np.load(p)["X"][:, :18])
    X = np.concatenate(cols)
    return X.min(0), X.max(0)


def common_checks(slug, reg, info, feats, enpl, rows):
    """Checks 1 to 3 of the protocol (also used before the allocator runs). Returns (list of lines, list of failures)."""
    lines, fails = [], []
    R = len(reg)
    ok = 60 <= R <= 1500
    lines.append(f"check 1: {R} regions (60 to 1,500: {'ok' if ok else 'FAIL'}); box {info['box_km2']:.0f} km2 (at most 1,900: {'ok' if info['box_km2'] <= 1900 else 'FAIL'}); "
                 f"grid cell {feats['cell_km2_median']:.3f} km2 (0.07 to 0.95: {'ok' if 0.07 <= feats['cell_km2_median'] <= 0.95 else 'FAIL'})")
    if not ok:
        fails.append(f"{R} regions, outside 60 to 1,500")
    if info["box_km2"] > 1900:
        fails.append(f"box {info['box_km2']:.0f} km2 above 1,900 (radius {info['radius_km']:.0f} km)")
    if not 0.07 <= feats["cell_km2_median"] <= 0.95:
        fails.append(f"grid cell {feats['cell_km2_median']:.3f} km2 outside 0.07 to 0.95")
    X = np.load(f"{MC}/data/region_static_{slug}.npz")["X"][:, :18]
    lo, hi = pooled_feature_range()
    outside = float(((X < lo - 1e-9) | (X > hi + 1e-9)).mean())
    nan = bool(np.isnan(X).any())
    ok2 = feats["empty_place_layers"] == 0 and not nan and outside <= 0.05
    lines.append(f"check 2: empty map layers {feats['empty_place_layers']}; NaN {nan}; share of the 18 map columns outside the pooled helper range {100 * outside:.2f}% (at most 5%: {'ok' if outside <= 0.05 else 'FAIL'}; Sydney 0.2%)")
    if not ok2:
        fails.append(f"map features: empty layers {feats['empty_place_layers']}, NaN {nan}, outside range {100 * outside:.1f}%")
    ap = rows
    ap_ok = True
    for iata, region, v in feats["airports"]:
        pass
    want = [a for a in ap if a["value"] > 0]
    got = {a["iata"] for a in want if any(h[0] == a["iata"] and h[2] > 0 for h in feats["airports"])}
    miss = [a["iata"] for a in want if a["iata"] not in got and a["in_box"]]
    lines.append(f"check 3: airports with a value above 0 inside a kept region: {sorted(got) or 'none'}; listed with a value above 0 and inside the study box but in no kept region: {miss or 'none'}")
    if miss:
        fails.append(f"airport {', '.join(miss)} (value above 0) lies in the box but in no kept region")
    return lines, fails


def airport_rows(slug, reg):
    p = f"{MC}/airports_cities100.csv"
    if not os.path.exists(p):
        return []
    a = pd.read_csv(p)
    a = a[a["slug"] == slug]
    minx, miny, maxx, maxy = reg.total_bounds
    out = []
    for _, r in a.iterrows():
        v = float(r["enplanements_m"])
        out.append({"iata": r["iata"], "value": v if v >= 5.0 else 0.0,
                    "in_box": bool(minx <= r["lon"] <= maxx and miny <= r["lat"] <= maxy)})
    return out


# ---------------------------------------------------------------- shares from the runs
def norm(p):
    return p / p.sum(1, keepdims=True)


def accumulate(slug, runs, srcs_fn, R):
    """Returns P (mean shares per slot, both heads averaged), per-head sd of per-run slot-mean share, and the stability / sum checks."""
    files = run_files(slug, runs, srcs_fn)
    miss = [f for f in files.values() if not os.path.exists(f)]
    if miss:
        raise FileNotFoundError(f"{len(miss)} run files missing, first: {os.path.basename(miss[0])}")
    from slots import calendar_slots
    sl = calendar_slots(slug, START)
    sums, sds, bad_sum, nan = {}, {}, 0.0, False
    for h in ("eh", "eo"):
        acc, means = None, []
        for rr in runs:
            p = np.load(files[(rr, h)]).astype(np.float64)
            assert p.shape == (len(sl), R), f"{os.path.basename(files[(rr, h)])}: shape {p.shape}, expected {(len(sl), R)}"
            nan = nan or bool(np.isnan(p).any())
            bad_sum = max(bad_sum, float(np.abs(p.sum(1) - 1).max()))
            p = norm(p)
            acc = p if acc is None else acc + p
            means.append(p.mean(0))
        sums[h] = acc / len(runs)
        sds[h] = np.std(means, 0) if len(runs) > 1 else np.zeros(R)
    P = norm(0.5 * sums["eh"] + 0.5 * sums["eo"])
    return P, sds, sl, bad_sum, nan


def week_tables(P, sl, rh, nd):
    di, si = np.array([a for a, _ in sl]), np.array([b for _, b in sl])
    dow = np.array([(START + timedelta(days=int(d))).weekday() for d in range(365)])
    w_t = dow[di]
    assert (np.bincount(dow, minlength=7) == np.array(nd)).all(), "the day counts differ from results/sydney_week.json"
    Tt = rh[w_t, si] / np.array(nd)[w_t] * 365.0
    ann = (P * Tt[:, None]).sum(0)
    share = ann / ann.sum()
    R = P.shape[1]
    S = np.zeros((7, 48, R))
    for w in range(7):
        for s in range(48):
            S[w, s] = P[(w_t == w) & (si == s)].mean(0)
    return share, S


# ---------------------------------------------------------------- the export
def write_city_files(meta, d, areas_rows, checks_lines):
    cs = meta["country_slug"]
    folder = f"{EXPORT}/cities/{cs}"
    os.makedirs(folder, exist_ok=True)
    json.dump(d, open(f"{folder}/{meta['slug']}.json", "w"), separators=(",", ":"))
    with open(f"{folder}/{meta['slug']}_areas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "area", "share_pct", "jobs", "residents", "run_to_run_sd_pct"])
        w.writerows(areas_rows)
    open(f"{folder}/{meta['slug']}_checks.txt", "w", encoding="utf-8").write("\n".join(checks_lines) + "\n")


def write_meta(meta):
    os.makedirs(f"{EXPORT}/_meta", exist_ok=True)
    json.dump(meta, open(f"{EXPORT}/_meta/{meta['slug']}.json", "w"), indent=1)
    rebuild_cities_csv()


CSV_COLUMNS = ["slug", "name", "country", "country_slug", "region", "area_label", "lat", "lon", "timezone", "locale", "calendar_year", "areas", "median_area_km2",
               "runs", "heads", "model_version", "built_on", "status", "reason", "total_reliable", "default_total_per_day", "attribution", "wikidata_id"]


def rebuild_cities_csv():
    metas = [json.load(open(p)) for p in sorted(f"{EXPORT}/_meta/{f}" for f in os.listdir(f"{EXPORT}/_meta") if f.endswith(".json"))]
    order = {s: k for k, s in enumerate(pd.read_csv(f"{MC}/cities100.csv")["slug"])}
    metas.sort(key=lambda m: order.get(m["slug"], 999))
    with open(f"{EXPORT}/cities.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(metas)


def picture(slug, g, share, d):
    from PIL import Image, ImageDraw
    minx, miny, maxx, maxy = g.total_bounds
    W = 900
    s = W / max(maxx - minx, maxy - miny)
    img = Image.new("RGB", (int((maxx - minx) * s) + 20, int((maxy - miny) * s) + 20), (244, 248, 249))
    dr = ImageDraw.Draw(img)
    ramp = [(238, 246, 248), (205, 230, 236), (159, 207, 219), (100, 176, 196), (46, 140, 168), (10, 101, 133), (7, 63, 92)]
    vmax = np.sort(share)[int(0.97 * (len(share) - 1))] or 1
    for i, geom in enumerate(g.geometry):
        t = math.sqrt(min(1, share[i] / vmax)) * 6
        k = min(5, int(t))
        col = tuple(round(ramp[k][j] + (ramp[k + 1][j] - ramp[k][j]) * (t - k)) for j in range(3))
        parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in parts:
            dr.polygon([((x - minx) * s + 10, (maxy - y) * s + 10) for x, y in p.exterior.coords], fill=col, outline=(255, 255, 255))
    os.makedirs(f"{EXPORT}/_pics", exist_ok=True)
    img.save(f"{EXPORT}/_pics/{slug}.png")


def export_city(slug, runs, keep=False):
    from runner_city import city_srcs
    r = city_row(slug)
    t0 = time.time()
    out = f"{MC}/data/{slug}"
    reg = gpd.read_file(f"{out}/{slug}_regions.geojson").set_crs(4326, allow_override=True).sort_values("rid").reset_index(drop=True)
    info = json.load(open(f"{out}/summary_regions.json"))
    feats = json.load(open(f"{out}/summary_features.json"))
    centre = json.load(open(f"{out}/centre.json"))
    R = len(reg)
    syd = json.load(open(f"{ROOT}/results/sydney_week.json"))
    rh, nd = np.array(syd["rhythm"]), syd["nd"]
    lines = [f"{slug}: {r['city']}, {r['region']}, {r['country']}; centre {centre['lat']:.5f}, {centre['lon']:.5f} ({centre['display_name'][:80]})"]
    aps = airport_rows(slug, reg)
    cl, fails = common_checks(slug, reg, info, feats, load_enpl(slug), aps)
    lines += cl
    try:
        P, sds, sl, bad_sum, nan = accumulate(slug, runs, city_srcs, R)
    except FileNotFoundError as e:
        fails.append(f"check 4: {e}")
        P = None
    meta = {"slug": slug, "name": r["city"], "country": r["country"], "country_slug": COUNTRY_SLUG[r["country"]], "region": region_display(r),
            "area_label": AREA_LABEL[r["country"]], "lat": round(centre["lat"], 5), "lon": round(centre["lon"], 5), "timezone": timezone(r), "locale": LOCALE[r["country"]],
            "calendar_year": 2023, "areas": R, "median_area_km2": round(info["median_area_km2"], 3), "runs": 2 * len(runs), "heads": "eh+eo", "model_version": MODEL_VERSION,
            "built_on": time.strftime("%Y-%m-%d", time.gmtime()), "status": "published", "reason": "", "total_reliable": "false", "default_total_per_day": "",
            "attribution": ATTRIBUTION[r["country"]], "wikidata_id": ""}
    if P is not None:
        sl_ok = bad_sum <= 1e-6 and not nan
        lines.append(f"check 4: {2 * len(runs)} runs present; largest deviation of a slot's shares from 1 {bad_sum:.2e} (at most 1e-6); NaN {nan}: {'ok' if sl_ok else 'FAIL'}")
        if not sl_ok:
            fails.append(f"check 4: shares sum deviates by {bad_sum:.1e} or NaN")
        share, S = week_tables(P, sl, rh, nd)
        sd = np.sqrt(sds["eh"] ** 2 / 2 + sds["eo"] ** 2 / 2)
        stab = float(sd.mean() / share.mean() * (1 if True else 1))
        stab = float(sd.mean() / (1.0 / R))
        ok5 = stab <= THRESH_RUN_TO_RUN
        lines.append(f"check 5: run-to-run: mean sd of a region's single-run share / mean share = {stab:.3f} (at most {THRESH_RUN_TO_RUN}; Sydney 0.28): {'ok' if ok5 else 'FAIL'}")
        if not ok5:
            fails.append(f"check 5: run-to-run {stab:.2f} above {THRESH_RUN_TO_RUN}")
        from scipy.stats import spearmanr
        dens = reg["density"].values
        rho = float(spearmanr(share, dens)[0])
        lines.append(f"check 6: Spearman of the yearly share with the OpenStreetMap place density = {rho:.3f} (at least 0.3): {'ok' if rho >= 0.3 else 'FAIL'}")
        if not rho >= 0.3:
            fails.append(f"check 6: Spearman with place density {rho:.2f} below 0.3")
        mp, g = map_paths(reg, info["utm"])
        inside = all(0 <= x <= mp["w"] for x in mp["cx"]) and all(0 <= y <= mp["h"] for y in mp["cy"]) and all(p.startswith("M") for p in mp["paths"])
        lines.append(f"check 7: map paths {len(mp['paths'])}, none empty, all label points inside the box {mp['w']} x {mp['h']} units (20 m), north up: {'ok' if inside else 'FAIL'}")
        if not inside:
            fails.append("check 7: map paths empty or outside the box")
        # areas, labels, indices
        order = np.argsort(-share)
        enpl = load_enpl(slug)
        airport_i = [int(i) for i in np.where(enpl > 0)[0]]
        cm = reg.to_crs(info["utm"])
        cpt = gpd.GeoSeries(gpd.points_from_xy([centre["lon"]], [centre["lat"]]), crs=4326).to_crs(info["utm"]).iloc[0]
        dist = cm.geometry.centroid.distance(cpt).values / 1000
        centre_i = [int(i) for i in np.where((dist <= 2.5) | cm.geometry.contains(cpt).values)[0]]
        labels, seen = [], set()
        for i in airport_i + [int(i) for i in order[:8]]:
            if i not in seen and len(labels) < 8:
                seen.add(i)
                labels.append({"i": i, "text": "Airport" if i in airport_i else str(reg.loc[i, "name"])[:22]})
        d = {"names": reg["name"].tolist(), "pid": reg["uid"].astype(str).tolist(), "area": reg["area_km2"].round(3).tolist(), "jobs": None, "res": None,
             "rhythm": syd["rhythm"], "nd": nd, "level_year_strict": None, "share_year": [round(float(x), 8) for x in (share / share.sum())],
             "share_ws": np.round(S * 1e6).astype(int).transpose(2, 0, 1).tolist(), **mp, "labels": labels, "airport_i": airport_i, "centre_i": centre_i,
             "heads": "eh+eo", "runs": 2 * len(runs), "area_label": AREA_LABEL[r["country"]]}
        d["share_year"][int(np.argmax(d["share_year"]))] += round(1 - sum(d["share_year"]), 8)
        areas_rows = [[k + 1, reg.loc[i, "name"], f"{100 * share[i]:.4f}", "", "", f"{100 * sd[i]:.4f}"] for k, i in enumerate(order)]
        lines.append(f"top areas: " + "; ".join(f"{reg.loc[i, 'name']} {100 * share[i]:.2f}%" for i in order[:10]))
        lines.append(f"top-10 areas hold {100 * share[order[:10]].sum():.1f}% of the demand; airport areas {[reg.loc[i, 'name'] for i in airport_i]} hold {100 * share[airport_i].sum():.1f}%")
        meta["wikidata_id"] = wikidata_id(f"{r['city']}", centre["lat"], centre["lon"])
        picture(slug, g, share, d)
    if fails:
        meta["status"], meta["reason"] = "draft", "; ".join(fails)[:600]
        lines.append("check 8: FAILED, status draft: " + meta["reason"])
    else:
        lines.append("check 8: all checks passed, status published")
    if P is not None and not fails:
        write_city_files(meta, d, areas_rows, lines)
    elif P is not None:
        write_city_files(meta, d, areas_rows, lines)       # the files of a draft are written too (the page ignores drafts), so that the owner can look
    else:
        os.makedirs(f"{EXPORT}/cities/{meta['country_slug']}", exist_ok=True)
        open(f"{EXPORT}/cities/{meta['country_slug']}/{slug}_checks.txt", "w").write("\n".join(lines) + "\n")
    write_meta(meta)
    os.makedirs(f"{ROOT}/miliroute/status", exist_ok=True)
    print("\n".join(lines), flush=True)
    if P is not None and not keep:
        for f in run_files(slug, runs, city_srcs).values():
            if os.path.exists(f):
                os.remove(f)
        for f in os.listdir(f"{HERE}/out_cal"):
            if f.startswith(f"multi_{slug}_from_") and f.endswith(".json"):
                os.remove(f"{HERE}/out_cal/{f}")
    print(f"{slug}: exported as {meta['status']} in {time.time() - t0:.0f} s", flush=True)
    return meta


def load_enpl(slug):
    return np.load(f"{HERE}/data/enpl_{slug}.npy")


# ---------------------------------------------------------------- Sydney from the laptop's result (results/sydney_*)
def export_sydney():
    slug = "sydney-nsw"
    syd = json.load(open(f"{ROOT}/results/sydney_week.json"))
    reg = gpd.read_file(f"{MC}/data/sydney/sydney_regions.geojson").set_crs(4326, allow_override=True).sort_values("rid").reset_index(drop=True)
    assert syd["names"] == reg["LOC_NAME"].tolist()
    mp, g = map_paths(reg, "EPSG:7856")
    centre = json.load(open(f"{MC}/data/sydney/downtown.json"))
    names = syd["names"]
    idx = lambda n: names.index(n)
    enpl = np.load(f"{HERE}/data/enpl_sydney.npy")
    airport_i = [int(i) for i in np.where(enpl > 0)[0]]
    cm = g
    cpt = gpd.GeoSeries(gpd.points_from_xy([centre[1]], [centre[0]]), crs=4326).to_crs("EPSG:7856").iloc[0]
    dist = reg.to_crs("EPSG:7856").geometry.centroid.distance(cpt).values / 1000
    centre_i = [int(i) for i in np.where((dist <= 2.5) | reg.to_crs("EPSG:7856").geometry.contains(cpt).values)[0]]
    labels = [{"i": idx("Sydney"), "text": "City"}, {"i": airport_i[0], "text": "Airport"}] + [{"i": idx(n), "text": n} for n in ("Parramatta", "Bondi", "Manly", "Chatswood", "Bankstown", "Hurstville")]
    d = {k: syd[k] for k in ("names", "pid", "area", "jobs", "res", "rhythm", "nd", "level_year_strict", "share_year", "share_ws")}
    d.update(mp)
    d.update({"labels": labels, "airport_i": airport_i, "centre_i": centre_i, "heads": "eh+eo", "runs": 20, "area_label": "suburb"})
    sub = pd.read_csv(f"{ROOT}/results/sydney_suburbs.csv")
    sub = sub.sort_values("rank")
    areas_rows = [[int(x.rank), x.suburb, f"{x.share_pct:.4f}", int(x.jobs), int(x.residents), f"{x.run_to_run_sd_of_slot_share_pct:.4f}"] for x in sub.itertuples()]
    meta = {"slug": slug, "name": "Sydney", "country": "Australia", "country_slug": "australia", "region": "NSW", "area_label": "suburb", "lat": round(centre[0], 5), "lon": round(centre[1], 5),
            "timezone": "Australia/Sydney", "locale": "en-AU", "calendar_year": 2023, "areas": len(names), "median_area_km2": round(float(np.median(syd["area"])), 3), "runs": 20,
            "heads": "eh+eo", "model_version": MODEL_VERSION, "built_on": time.strftime("%Y-%m-%d", time.gmtime()), "status": "published", "reason": "", "total_reliable": "false",
            "default_total_per_day": "", "attribution": "Suburb outlines: Geoscape Administrative Boundaries (CC BY 4.0). Jobs and residents in 2021: Transport for NSW Travel Zone Projections (CC BY). "
            "Commuters: ABS Census 2021 (CC BY 4.0). Airport boardings: half of the 2019 passengers of Sydney Airport", "wikidata_id": ""}
    meta["wikidata_id"] = wikidata_id("Sydney", centre[0], centre[1])
    lines = [open(f"{ROOT}/results/sydney_checks.txt").read().strip(), "Converted from results/sydney_week.json and results/sydney_suburbs.csv (the laptop's 20 runs, PROTOCOL_SYDNEY.md); not recomputed."]
    errs = check_city_data_fields(d)
    assert not errs, errs
    write_city_files(meta, d, areas_rows, lines)
    picture(slug, g, np.array(syd["share_year"]), d)
    write_meta(meta)
    print(f"sydney-nsw exported: {len(names)} suburbs, wikidata {meta['wikidata_id'] or 'none'}", flush=True)


def check_city_data_fields(d):
    for k in ("names", "area", "rhythm", "nd", "share_year", "share_ws", "paths", "w", "h", "cx", "cy"):
        if k not in d:
            return [f"missing {k}"]
    R = len(d["names"])
    errs = []
    for k in ("area", "paths", "cx", "cy", "share_ws", "share_year"):
        if len(d[k]) != R:
            errs.append(k)
    return errs


if __name__ == "__main__":
    slug = sys.argv[1]
    runs = list(range(1, (int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 10) + 1))
    if slug == "sydney-nsw":
        export_sydney()
    else:
        export_city(slug, runs, keep="--keep" in sys.argv)
