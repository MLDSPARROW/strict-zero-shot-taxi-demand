"""EXPLORATION (geometry and public statistics only; no taxi demand is read except the published yearly trip totals of Table 1 and
Austin's span total). Two facts used to assess external ideas (2026-10-21):
 (1) alternative summaries of region size as a multiple of the median area, for every target (the block-size sweep of the paper is
     flat between 0.5 and 2 times the median; its best values were 1.0 for Chicago and 1.5 for New York City);
 (2) yearly trips per transit commuter (the level unit chosen for every target) for the taxi cities and for RideAustin.
Usage (from improve/): python explore_scale_stats.py"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); F = f"{HERE}/../features"; MC = f"{HERE}/../multicity/data"
P = {"Chicago": f"{F}/region_static_chicago.npz", "NYC": f"{F}/region_static_nyc.npz", "SF tracts": f"{F}/region_static_sf.npz",
     "DC tracts": f"{F}/region_static_dc.npz", "SF ZIP areas": f"{MC}/region_static_sf_zcta.npz", "DC clusters": f"{MC}/region_static_dc_cluster.npz",
     "Austin cells": f"{MC}/region_static_austin.npz"}
print("(1) region-size summaries as a multiple of the median area (mean = total area / number of regions = inverse region density)")
print(f"{'target':14s} {'median km2':>10s} {'geo-mean':>9s} {'mean':>7s} {'area-wtd mean':>13s} {'25% / median':>13s} {'75% / median':>13s}")
for k, p in P.items():
    a = np.load(p)["area_km2"]; m = np.median(a)
    print(f"{k:14s} {m:10.2f} {np.exp(np.log(a).mean()) / m:9.2f} {a.mean() / m:7.2f} {(a ** 2).sum() / a.sum() / m:13.2f} {np.quantile(a, .25) / m:13.2f} {np.quantile(a, .75) / m:13.2f}")
acs = json.load(open(f"{HERE}/raw/acs_city.json"))["data"]; acs.update(json.load(open(f"{HERE}/raw/acs_austin.json"))["data"])
G = {"chicago": "16000US1714000", "nyc": "16000US3651000", "sf": "16000US0667000", "dc": "16000US1150000", "austin": "16000US4805000"}
trips = {"chicago": 3639126, "nyc": 30533607, "sf": 1703882, "dc": 1195111, "austin": 900643 * 365 / 248}     # Austin: 900,643 rides in 248 days, annualised
print("\n(2) yearly trips per transit commuter (Austin: rides in the 248 evaluated days scaled to a year)")
for c, g in G.items():
    tc = acs[g]["B08301"]["estimate"]["B08301010"]; print(f"   {c:8s} transit commuters {tc:>9,.0f}  trips per commuter {trips[c] / tc:6.1f}")
