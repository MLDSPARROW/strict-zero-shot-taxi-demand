"""PROTOCOL_CITIES100.md: scale-matched re-aggregation of the taxi helper cities Chicago, NYC and SF for one city (copy of build_sydney_partitions.py with the city as argument),
with exactly the function build_partitions of build_coarse_targets.py (K from k-means with random state 0 so that the median block area is closest to the city's median region
area on the logarithmic scale; ten partitions with the random states 1 to 10; a helper whose median region area is already at least the target's is left native).
Taxi helper data and the city's map data only; no demand of the city exists. Usage (from improve/): python -I build_city_partitions.py <slug>
Output: improve/data/agg/<H>_for_<slug>_p<r>_{Y.npy, static.npz}, improve/data/extra_<H>_for_<slug>_p<r>.npz, improve/data/enpl_<H>_for_<slug>_p<r>.npy"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
slug = sys.argv[1]
src = open("build_coarse_targets.py").read(); ns = {}
exec(src[:src.index('a_sf = build_target("sf", "sf_zcta")')], ns)                  # the checks of the trip totals and the function build_partitions
area = np.load(f"../multicity/data/region_static_{slug}.npz")["area_km2"]; aT = float(np.median(area))
print(f"target: {slug}, {len(area)} regions, median area {aT:.3f} km2", flush=True)
for H in ("chicago", "nyc", "sf"): ns["build_partitions"](H, slug, aT, slug)
print("done", flush=True)
