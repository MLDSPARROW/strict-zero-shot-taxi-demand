"""PROTOCOL_SYDNEY.md: scale-matched re-aggregation of the taxi helper cities Chicago, NYC and SF for Sydney, with exactly the function build_partitions of
build_coarse_targets.py (K from k-means with random state 0 so that the median block area is closest to the median suburb area of Sydney on the
logarithmic scale; ten partitions with the random states 1 to 10; a helper whose median region area is already at least the target's is left native).
Taxi helper data and Sydney map data only; no Sydney demand exists. Usage (from improve/): python build_sydney_partitions.py
Output: improve/data/agg/<H>_for_sydney_p<r>_{Y.npy, static.npz}, improve/data/extra_<H>_for_sydney_p<r>.npz, improve/data/enpl_<H>_for_sydney_p<r>.npy"""
import os, numpy as np
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
src = open("build_coarse_targets.py").read(); ns = {}
exec(src[:src.index('a_sf = build_target("sf", "sf_zcta")')], ns)                  # the checks of the trip totals and the function build_partitions
aT = float(np.median(np.load("../multicity/data/region_static_sydney.npz")["area_km2"]))
print(f"target: Sydney, {len(np.load('../multicity/data/region_static_sydney.npz')['area_km2'])} suburbs, median area {aT:.3f} km2", flush=True)
for H in ("chicago", "nyc", "sf"): ns["build_partitions"](H, "sydney", aT, "sydney")
print("done", flush=True)
