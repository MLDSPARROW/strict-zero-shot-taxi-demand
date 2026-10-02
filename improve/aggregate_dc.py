"""Scale-matched helper cities (declared 2026-10-01 before running). For a target T with median region area a_T (public polygons
only), each helper city H is re-aggregated into K = round(total area of H / a_T) blocks by k-means on region centroids (km),
seeded, so its median block area is close to a_T. Block features are recomputed with EXACTLY the region-feature formula
(area-weighted fine-grid map features), demand = sum of member regions, jobs/residents/rail/airport passengers = sums.
Output per (helper, target): improve/data/agg/<H>_for_<T>_{Y.npy, static.npz}, data/extra_<H>_for_<T>.npz, data/enpl_<H>_for_<T>.npy"""
import os, numpy as np
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
from sklearn.cluster import KMeans
B, M, G = "../bench77/data", "../multicity/data", "../grid/data"
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy", "sf": f"{M}/sf_2023_tract_30min.npy", "dc": f"{M}/dc_2021_tract_30min.npy"}
SP = {"chicago": f"{B}/region_static_chicago.npz", "nyc": f"{B}/region_static_nyc.npz", "sf": f"{M}/region_static_sf.npz", "dc": f"{M}/region_static_dc.npz"}
def feats(frac, cell_km2, Xf, area, cen):
    W = frac * cell_km2[None]; Wn = W / np.maximum(W.sum(1, keepdims=True), 1e-9); f = Xf.astype(np.float64)
    dens = Wn @ np.expm1(f[:, :8]); other = Wn @ f[:, 8:]; busy = dens.sum(1)
    pr = dens / np.clip(busy[:, None], 1e-9, None); mix = -np.sum(np.where(pr > 0, pr * np.log(pr + 1e-9), 0.0), 1)
    F = np.concatenate([np.log1p(dens), other[:, 0:3], np.log1p(busy)[:, None], mix[:, None], other[:, 5:8]], 1)
    nbr = np.argsort(np.linalg.norm(cen[:, None] - cen[None], axis=2), 1)[:, 1:7]; con = F[:, 11] - F[:, 11][nbr].mean(1)
    cover = W.sum(1) / area
    return np.concatenate([F, np.log(area / np.median(area))[:, None], con[:, None], np.clip(cover, 0, 1)[:, None]], 1).astype(np.float32)
# check: the formula reproduces the stored native features
for c in ["dc"]:
    g = np.load(f"{G}/grid_{c}.npz"); S = np.load(SP[c])
    Xr = feats(g["frac"], g["cell_km2"], g["X"], S["area_km2"], S["centroid_km"])
    d = np.abs(Xr[:, :18] - S["X"][:, :18]); print(f"formula check {c}: max abs diff over 18 features {d.max():.4f} (median {np.median(d):.5f})")
PLAN = {"chicago": ["dc"], "nyc": ["dc"]}   # rule R1 extension (2026-10-02): DC scale-matched too
for T, Hs in PLAN.items():
    aT = np.median(np.load(SP[T])["area_km2"])
    for H in Hs:
        S = np.load(SP[H]); area = S["area_km2"]
        if np.median(area) >= aT: print(f"{H} for {T}: helper regions already >= target size, skipped"); continue
        best = None                                   # choose K so the MEDIAN block area is closest to the target median area
        for K in range(4, len(area)):
            lab_ = KMeans(K, n_init=4, random_state=0).fit(S["centroid_km"], sample_weight=area).labels_
            med = np.median(np.bincount(lab_, weights=area, minlength=K))
            if best is None or abs(np.log(med / aT)) < abs(np.log(best[1] / aT)): best = (K, med, lab_)
            if med < aT * 0.8: break
        K, _, lab = best
        Mm = np.zeros((K, len(area))); Mm[lab, np.arange(len(area))] = 1
        g = np.load(f"{G}/grid_{H}.npz"); frac = Mm @ g["frac"]; a2 = Mm @ area; cen = (Mm @ (S["centroid_km"] * area[:, None])) / a2[:, None]
        X2 = feats(frac, g["cell_km2"], g["X"], a2, cen)
        Y = np.load(YP[H]).astype(np.float32); Y2 = (Y.reshape(-1, Y.shape[2]) @ Mm.T.astype(np.float32)).reshape(Y.shape[0], 48, K)
        E = np.load(f"data/extra_{H}.npz"); jobs, res, rail = Mm @ E["jobs"], Mm @ E["res"], Mm @ E["rail"]
        enpl = Mm @ np.load(f"data/enpl_{H}.npy")
        E2 = np.stack([np.log1p(jobs / a2), np.log1p(res / a2), np.log1p(rail / a2), (enpl > 0).astype(float)], 1).astype(np.float32)
        tag = f"{H}_for_{T}"
        np.save(f"data/agg/{tag}_Y.npy", Y2); np.savez(f"data/agg/{tag}_static.npz", X=X2, rid=np.arange(K), centroid_km=cen, area_km2=a2, cover=X2[:, 18])
        np.savez(f"data/extra_{tag}.npz", E=E2, jobs=jobs, res=res, rail=rail); np.save(f"data/enpl_{tag}.npy", enpl.astype(np.float32))
        print(f"{tag}: {len(area)} regions -> {K} blocks, median block area {np.median(a2):.2f} km2 (target {aT:.2f}), trips kept {Y2.sum()/Y.sum():.4f}, airport blocks {int((enpl>0).sum())}")
