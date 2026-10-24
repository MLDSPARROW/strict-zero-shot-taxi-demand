"""PROTOCOL_V9.md: scale-matched re-aggregation of the bike-share cities for a target, exactly the procedure of build_coarse_targets.build_partitions
(K from k-means with random state 0 so that the median block area is closest to the median region area of the target on the logarithmic scale; ten
partitions with the random states 1 to 10; block features recomputed with the area-weighted formula of aggregate.py from the 40x50 fine grid).
A bike city whose median tract area is already at least the target's is left native. Map data and bike-share trips only; no taxi data.
Usage (from improve/): python build_bike_partitions.py <target> [<target> ...]     targets: chicago nyc sf dc sf_zcta dc_cluster austin
Output: improve/data/agg/bike_<city>_for_<target>_p<r>_{Y.npy, static.npz}, improve/data/enpl_bike_<city>_for_<target>_p<r>.npy (zeros)"""
import os, sys, numpy as np
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from sklearn.cluster import KMeans
HERE = os.path.dirname(os.path.abspath(__file__)); B77, MC, G = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data", f"{HERE}/../grid/data"
BIKE = ["boston", "philadelphia", "la", "minneapolis", "columbus"]
TSTATIC = {"chicago": f"{B77}/region_static_chicago.npz", "nyc": f"{B77}/region_static_nyc.npz", "sf": f"{MC}/region_static_sf.npz", "dc": f"{MC}/region_static_dc.npz",
           "sf_zcta": f"{MC}/region_static_sf_zcta.npz", "dc_cluster": f"{MC}/region_static_dc_cluster.npz", "austin": f"{MC}/region_static_austin.npz"}
os.makedirs(f"{HERE}/data/agg", exist_ok=True)


def feats(frac, cell_km2, Xf, area, cen):                                  # as aggregate.py
    W = frac * cell_km2[None]; Wn = W / np.maximum(W.sum(1, keepdims=True), 1e-9); f = Xf.astype(np.float64)
    dens = Wn @ np.expm1(f[:, :8]); other = Wn @ f[:, 8:]; busy = dens.sum(1)
    pr = dens / np.clip(busy[:, None], 1e-9, None); mix = -np.sum(np.where(pr > 0, pr * np.log(pr + 1e-9), 0.0), 1)
    F = np.concatenate([np.log1p(dens), other[:, 0:3], np.log1p(busy)[:, None], mix[:, None], other[:, 5:8]], 1)
    nbr = np.argsort(np.linalg.norm(cen[:, None] - cen[None], axis=2), 1)[:, 1:7]; con = F[:, 11] - F[:, 11][nbr].mean(1)
    cover = W.sum(1) / area
    return np.concatenate([F, np.log(area / np.median(area))[:, None], con[:, None], np.clip(cover, 0, 1)[:, None]], 1).astype(np.float32)


for T in sys.argv[1:]:
    aT = float(np.median(np.load(TSTATIC[T])["area_km2"]))
    for H in BIKE:
        S = np.load(f"{MC}/bike/{H}/region_static_bike_{H}.npz"); area = S["area_km2"]; g = np.load(f"{G}/grid_bike_{H}.npz")
        Xr = feats(g["frac"], g["cell_km2"], g["X"], area, S["centroid_km"]); d = np.abs(Xr[:, :18] - S["X"][:, :18])
        assert d.max() < 1e-3, f"{H}: the formula does not reproduce the stored features (max abs diff {d.max():.4f})"
        if np.median(area) >= aT: print(f"bike_{H} for {T}: tracts already >= target size ({np.median(area):.2f} >= {aT:.2f} km2), left native", flush=True); continue
        best = None
        for K in range(4, len(area)):
            lab_ = KMeans(K, n_init=4, random_state=0).fit(S["centroid_km"], sample_weight=area).labels_
            med = np.median(np.bincount(lab_, weights=area, minlength=K))
            if best is None or abs(np.log(med / aT)) < abs(np.log(best[1] / aT)): best = (K, med)
            if med < aT * 0.8: break
        K = best[0]; Y = np.load(f"{MC}/bike/{H}/bike_{H}_2021_tract_30min.npy").astype(np.float32); meds = []
        for r in range(1, 11):
            lab = KMeans(K, n_init=4, random_state=r).fit(S["centroid_km"], sample_weight=area).labels_
            Mm = np.zeros((lab.max() + 1, len(area))); Mm[lab, np.arange(len(area))] = 1; Mm = Mm[Mm.sum(1) > 0]
            frac = Mm @ g["frac"]; a2 = Mm @ area; cen = (Mm @ (S["centroid_km"] * area[:, None])) / a2[:, None]
            X2 = feats(frac, g["cell_km2"], g["X"], a2, cen); Y2 = (Y.reshape(-1, Y.shape[2]) @ Mm.T.astype(np.float32)).reshape(Y.shape[0], 48, -1)
            tag = f"bike_{H}_for_{T}_p{r}"
            np.save(f"{HERE}/data/agg/{tag}_Y.npy", Y2)
            np.savez(f"{HERE}/data/agg/{tag}_static.npz", X=X2, rid=np.arange(len(a2)), centroid_km=cen, area_km2=a2, cover=X2[:, 18])
            np.save(f"{HERE}/data/enpl_{tag}.npy", np.zeros(len(a2), np.float32)); meds.append(np.median(a2))
        print(f"bike_{H} for {T}: {len(area)} tracts -> K={K} blocks, median block area {np.mean(meds):.2f} km2 over 10 partitions (target {aT:.2f}), trips kept {Y2.sum() / Y.sum():.4f}", flush=True)
print("done", flush=True)
