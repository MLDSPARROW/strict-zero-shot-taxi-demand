"""Aggregation control for scale matching (PROTOCOL_REVISION.md, part B). Builds three control representations of each
merged helper, with exactly the block-feature formula of aggregate.py, 10 partitions each (r = 1..10):
  R  random grouping with the same number of blocks K as the proposed scale-matched helper (balanced, not spatially coherent)
  H  area-weighted k-means with K chosen so the median block area is about half of the target median area
  D  area-weighted k-means with K chosen so the median block area is about twice the target median area
Output: data/agg/<H>_for_<T>_<R|H|D><r>_{Y.npy,static.npz}, data/extra_<tag>.npz, data/enpl_<tag>.npy. No target demand is read."""
import os, numpy as np
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from sklearn.cluster import KMeans
src = open("aggregate.py").read()
ns = {}; exec(src[:src.index("# check: the formula")], ns); feats, YP, SP, G = ns["feats"], ns["YP"], ns["SP"], ns["G"]


def choose_k(S, goal):                      # same search as aggregate.py, with a different goal area
    area = S["area_km2"]; best = None
    for K in range(2, len(area)):
        lab = KMeans(K, n_init=4, random_state=0).fit(S["centroid_km"], sample_weight=area).labels_
        med = np.median(np.bincount(lab, weights=area, minlength=K))
        if best is None or abs(np.log(med / goal)) < abs(np.log(best[1] / goal)): best = (K, med)
        if med < goal * 0.8: break
    return best[0]


def write(H, T, tag, lab, S, area, g, Y, E):
    K2 = lab.max() + 1; Mm = np.zeros((K2, len(area))); Mm[lab, np.arange(len(area))] = 1; Mm = Mm[Mm.sum(1) > 0]
    frac = Mm @ g["frac"]; a2 = Mm @ area; cen = (Mm @ (S["centroid_km"] * area[:, None])) / a2[:, None]
    X2 = feats(frac, g["cell_km2"], g["X"], a2, cen); Y2 = (Y.reshape(-1, Y.shape[2]) @ Mm.T.astype(np.float32)).reshape(Y.shape[0], 48, -1)
    jobs, res, rail = Mm @ E["jobs"], Mm @ E["res"], Mm @ E["rail"]; enpl = Mm @ np.load(f"data/enpl_{H}.npy")
    E2 = np.stack([np.log1p(jobs / a2), np.log1p(res / a2), np.log1p(rail / a2), (enpl > 0).astype(float)], 1).astype(np.float32)
    np.save(f"data/agg/{tag}_Y.npy", Y2); np.savez(f"data/agg/{tag}_static.npz", X=X2, rid=np.arange(len(a2)), centroid_km=cen, area_km2=a2, cover=X2[:, 18])
    np.savez(f"data/extra_{tag}.npz", E=E2, jobs=jobs, res=res, rail=rail); np.save(f"data/enpl_{tag}.npy", enpl.astype(np.float32))
    return len(a2), np.median(a2), int((enpl > 0).sum())


for T, H in [("chicago", "nyc"), ("chicago", "sf"), ("nyc", "sf")]:
    aT = np.median(np.load(SP[T])["area_km2"]); K_M = len(np.load(f"data/agg/{H}_for_{T}_static.npz")["area_km2"])
    S = np.load(SP[H]); area = S["area_km2"]; g = np.load(f"{G}/grid_{H}.npz"); Y = np.load(YP[H]).astype(np.float32); E = np.load(f"data/extra_{H}.npz")
    K = {"R": K_M, "H": choose_k(S, 0.5 * aT), "D": choose_k(S, 2.0 * aT)}
    print(f"{H} for {T}: target median {aT:.2f} km2, {len(area)} helper regions, K: M {K_M}, R {K['R']}, H {K['H']}, D {K['D']}", flush=True)
    for cond in ("R", "H", "D"):
        meds = []
        for r in range(1, 11):
            if cond == "R":
                lab = np.random.default_rng(r).permutation(np.arange(len(area)) % K["R"])
            else:
                lab = KMeans(K[cond], n_init=4, random_state=r).fit(S["centroid_km"], sample_weight=area).labels_
            n, med, nair = write(H, T, f"{H}_for_{T}_{cond}{r}", lab, S, area, g, Y, E); meds.append(med)
        print(f"   {cond}: {n} blocks, median block area over 10 partitions {np.mean(meds):.2f} km2 "
              f"(range {min(meds):.2f} to {max(meds):.2f}), airport blocks in last partition {nair}", flush=True)
