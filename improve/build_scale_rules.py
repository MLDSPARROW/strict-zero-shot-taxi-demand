"""PROTOCOL_V3.md (written before this script ran): helper blocks for the DC block-size sweep and for the mean rule. Geometry and helper
data only; no target demand is read. Outputs data/agg/<tag>_{Y.npy,static.npz}, data/extra_<tag>.npz, data/enpl_<tag>.npy for
  sf_for_dc_x{075,150,200,400,mean}p{r}, r = 1..5   (target Washington DC; helper San Francisco re-aggregated to goal q x median area;
                                                      the mean rule uses goal = mean region area of DC)
  chicago_for_sf_zcta_mp{r}, nyc_for_sf_zcta_mp{r}, r = 1..10   (target SF ZIP-code areas; mean rule, goal = their mean area)
Same k-means, same K search and same block-feature formula as aggregate.py / aggregate_sweep.py. Run from improve/."""
import os, numpy as np
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from sklearn.cluster import KMeans
src = open("aggregate_control.py").read(); ns = {}; exec(src[:src.index("for T, H in")], ns)
choose_k, write, SP, YP, G = ns["choose_k"], ns["write"], ns["SP"], ns["YP"], ns["G"]
M, F = "../multicity/data", "../features"
area_dc = np.load(f"{F}/region_static_dc.npz")["area_km2"]; area_zc = np.load(f"{M}/region_static_sf_zcta.npz")["area_km2"]
JOBS = [("sf", np.median(area_dc) * q, f"sf_for_dc_x{code}p", 5, "dc") for code, q in (("150", 1.5), ("200", 2.0), ("075", 0.75), ("400", 4.0))]
JOBS = [("sf", area_dc.mean(), "sf_for_dc_xmeanp", 5, "dc")] + JOBS + [("chicago", area_zc.mean(), "chicago_for_sf_zcta_mp", 10, "sf_zcta"),
                                                                     ("nyc", area_zc.mean(), "nyc_for_sf_zcta_mp", 10, "sf_zcta")]
for H, goal, prefix, nr, T in JOBS:
    S = np.load(SP[H]); area = S["area_km2"]; g = np.load(f"{G}/grid_{H}.npz"); Y = np.load(YP[H]).astype(np.float32); E = np.load(f"data/extra_{H}.npz")
    if np.median(area) >= goal:
        print(f"{prefix}: helper {H} native median {np.median(area):.3f} >= goal {goal:.3f} km2 -> native scale (nothing built)", flush=True); continue
    K = choose_k(S, goal); meds, ns_ = [], []
    for r in range(1, nr + 1):
        lab = KMeans(K, n_init=4, random_state=r).fit(S["centroid_km"], sample_weight=area).labels_
        n, med, _ = write(H, T, f"{prefix}{r}", lab, S, area, g, Y, E); meds.append(med); ns_.append(n)
    print(f"{prefix}1..{nr}: target {T}, helper {H} ({len(area)} regions), goal {goal:.3f} km2, K {K} -> {min(ns_)} to {max(ns_)} blocks, "
          f"median block area {np.mean(meds):.3f} km2 (range {min(meds):.3f} to {max(meds):.3f})", flush=True)
print("done")
