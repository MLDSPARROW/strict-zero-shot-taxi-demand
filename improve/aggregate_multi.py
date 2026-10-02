"""Partition-diverse scale-matched helpers (declared 2026-10-02 before running). Same as aggregate.py, but each helper is
re-aggregated with 5 more k-means random partitions (random_state 1..5) at the SAME number of blocks K found by aggregate.py
(median block area matched to the target's median region area). Model run r (r = 1..5) trains on partition r of every helper;
the 5 runs are averaged. Compared with 5 runs (seeds 0-4) on the single partition 0. Variants eh and eo. Fully strict scoring."""
import os, numpy as np
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from sklearn.cluster import KMeans
src = open("aggregate.py").read()
ns = {}; exec(src[:src.index("# check: the formula")], ns); feats, YP, SP, G = ns["feats"], ns["YP"], ns["SP"], ns["G"]
for T, H in [("chicago", "nyc"), ("chicago", "sf"), ("nyc", "sf")]:
    base = np.load(f"data/agg/{H}_for_{T}_static.npz"); K = len(base["area_km2"])
    S = np.load(SP[H]); area = S["area_km2"]; g = np.load(f"{G}/grid_{H}.npz"); Y = np.load(YP[H]).astype(np.float32); E = np.load(f"data/extra_{H}.npz")
    for r in range(1, 6):
        lab = KMeans(K, n_init=4, random_state=r).fit(S["centroid_km"], sample_weight=area).labels_
        K2 = lab.max() + 1; Mm = np.zeros((K2, len(area))); Mm[lab, np.arange(len(area))] = 1; Mm = Mm[Mm.sum(1) > 0]
        frac = Mm @ g["frac"]; a2 = Mm @ area; cen = (Mm @ (S["centroid_km"] * area[:, None])) / a2[:, None]
        X2 = feats(frac, g["cell_km2"], g["X"], a2, cen); Y2 = (Y.reshape(-1, Y.shape[2]) @ Mm.T.astype(np.float32)).reshape(Y.shape[0], 48, -1)
        jobs, res, rail = Mm @ E["jobs"], Mm @ E["res"], Mm @ E["rail"]; enpl = Mm @ np.load(f"data/enpl_{H}.npy")
        E2 = np.stack([np.log1p(jobs / a2), np.log1p(res / a2), np.log1p(rail / a2), (enpl > 0).astype(float)], 1).astype(np.float32)
        tag = f"{H}_for_{T}_p{r}"
        np.save(f"data/agg/{tag}_Y.npy", Y2); np.savez(f"data/agg/{tag}_static.npz", X=X2, rid=np.arange(len(a2)), centroid_km=cen, area_km2=a2, cover=X2[:, 18])
        np.savez(f"data/extra_{tag}.npz", E=E2, jobs=jobs, res=res, rail=rail); np.save(f"data/enpl_{tag}.npy", enpl.astype(np.float32))
        print(f"{tag}: {len(a2)} blocks, median {np.median(a2):.2f} km2")
