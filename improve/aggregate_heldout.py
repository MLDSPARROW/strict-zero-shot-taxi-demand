"""Held-out protocol step 4: SF re-aggregated to DC's median region area (partitions 1..10). Same feature formula as aggregate.py."""
import os, numpy as np
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from sklearn.cluster import KMeans
src = open("aggregate.py", encoding="utf-8").read(); ns = {}; exec(src[:src.index("# check: the formula")], ns)
feats, G, M = ns["feats"], ns["G"], ns["M"]
aT = np.median(np.load(f"{M}/region_static_dc.npz")["area_km2"])
S = np.load(f"{M}/region_static_sf.npz"); area = S["area_km2"]; g = np.load(f"{G}/grid_sf.npz")
Y = np.load(f"{M}/sf_2023_tract_30min.npy").astype(np.float32); E = np.load("data/extra_sf.npz"); en = np.load("data/enpl_sf.npy")
best = None
for K in range(4, len(area)):
    lab = KMeans(K, n_init=4, random_state=0).fit(S["centroid_km"], sample_weight=area).labels_
    med = np.median(np.bincount(lab, weights=area, minlength=K))
    if best is None or abs(np.log(med / aT)) < abs(np.log(best[1] / aT)): best = (K, med)
    if med < aT * 0.8: break
K = best[0]; print(f"DC median {aT:.2f} km2 -> SF K = {K}")
for r in range(1, 11):
    lab = KMeans(K, n_init=4, random_state=r).fit(S["centroid_km"], sample_weight=area).labels_
    Mm = np.zeros((lab.max() + 1, len(area))); Mm[lab, np.arange(len(area))] = 1; Mm = Mm[Mm.sum(1) > 0]
    frac = Mm @ g["frac"]; a2 = Mm @ area; cen = (Mm @ (S["centroid_km"] * area[:, None])) / a2[:, None]
    X2 = feats(frac, g["cell_km2"], g["X"], a2, cen); Y2 = (Y.reshape(-1, Y.shape[2]) @ Mm.T.astype(np.float32)).reshape(Y.shape[0], 48, -1)
    jobs, res, rail = Mm @ E["jobs"], Mm @ E["res"], Mm @ E["rail"]; enpl = Mm @ en
    E2 = np.stack([np.log1p(jobs / a2), np.log1p(res / a2), np.log1p(rail / a2), (enpl > 0).astype(float)], 1).astype(np.float32)
    tag = f"sf_for_dc_p{r}"
    np.save(f"data/agg/{tag}_Y.npy", Y2); np.savez(f"data/agg/{tag}_static.npz", X=X2, rid=np.arange(len(a2)), centroid_km=cen, area_km2=a2, cover=X2[:, 18])
    np.savez(f"data/extra_{tag}.npz", E=E2, jobs=jobs, res=res, rail=rail); np.save(f"data/enpl_{tag}.npy", enpl.astype(np.float32))
    print(f"{tag}: {len(a2)} blocks, median {np.median(a2):.2f} km2")
