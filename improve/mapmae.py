"""PROTOCOL_V4.md, Way 1: scale-matched masked map pre-training. Map data only: no taxi data of any city is read here.
For a target T, with median region area a_T from its public polygons, and a seed s:
  cells    the 40 x 50 map grids of Chicago, New York City, San Francisco, Washington, DC and Austin, 13 features per cell
           (8 log place densities, 3 log distances, log busyness, place-mix entropy; the 3 missing-data flags are dropped),
           standardised with the median and inter-quartile range pooled over all cells of the five cities and clipped to [-8, 8];
           cell sizes in km from the grid edges (local equirectangular projection; all cells of a grid have the same size).
  masks    the scale-matching rule applied to map cells: per city, 20 partitions of its cells into superpixels by k-means on the
           cell centres in km, K = round(grid area / a_T) (at most one superpixel per cell), random_state 1000 s + j. A training
           mask hides a random half of the superpixels of one partition, so that every hidden patch has about the size of a
           region of the target.
  encoder  masked normalised convolution in physical units: 3 residual layers; in each, every cell receives the Gaussian-weighted
           mean of the states of the VISIBLE cells at the bandwidths 0.5, 1 and 2 x sqrt(a_T) km and the visible fraction of the
           kernel mass at each bandwidth; hidden size 64. A hidden cell starts from a learned mask token and is never a source,
           so its state depends only on its visible surroundings. Bottleneck of 8 numbers per cell, decoder = 2-layer perceptron
           to the 13 features. Loss: mean squared error on the hidden cells, averaged over the five cities (one mask per city and
           step). AdamW, learning rate 1e-3, weight decay 1e-4, 2,000 steps, gradient norm clipped at 1.
  output   the context latent of every cell = its bottleneck while it is hidden, averaged over the 20 partitions, each used once
           with a random half of its superpixels hidden and once with the other half; for every region or block, the mean over
           its cells weighted by area (the region-feature formula). Blocks of re-aggregated helpers are rebuilt with the original
           k-means seeds and checked against the stored jobs sums.
  check    reconstruction error on the hidden cells of 5 further partitions that are not used in training, against two
           references without training: the city mean of the visible cells and the Gaussian-weighted mean of the visible cells
           at the bandwidth sqrt(a_T).
Usage (from improve/): python mapmae.py <target> <seed>
Output: data/mapmae/<unit>__<target>__s<seed>.npy (regions or blocks x 8) for the target and every share helper of the target."""
import os, sys, time, numpy as np, torch, torch.nn as nn
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from sklearn.cluster import KMeans
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = f"{HERE}/.."
B77, MC, G, S2 = f"{CODE}/bench77/data", f"{CODE}/multicity/data", f"{CODE}/grid/data", f"{CODE}/step2/cache"
torch.set_num_threads(max(1, os.cpu_count() // 2))
TGT, SEED = sys.argv[1], int(sys.argv[2])
CITIES = ["chicago", "nyc", "sf", "dc", "austin"]
SP = {"chicago": f"{B77}/region_static_chicago.npz", "nyc": f"{B77}/region_static_nyc.npz", "sf": f"{MC}/region_static_sf.npz",
      "dc": f"{MC}/region_static_dc.npz", "austin": f"{MC}/region_static_austin.npz"}
R10 = range(1, 11)
HELP = {"chicago": [("nyc", r) for r in R10] + [("sf", r) for r in R10], "nyc": [("chicago", None)] + [("sf", r) for r in R10]}
KM = np.pi / 180 * 6371.0088
NSTEP, NPART, NCHECK, HID, LAT = 2000, 20, 5, 64, 8
SMOKE = os.environ.get("MAPMAE_SMOKE") == "1"     # code test only: 20 steps, output to data/mapmae_smoke
if SMOKE: NSTEP = int(os.environ.get("MAPMAE_STEPS", 20))
OUT = f"{HERE}/data/{'mapmae_smoke' if SMOKE else 'mapmae'}"
t0 = time.time()


def cell_size(c):
    """Cell height and width in km. Austin: from the edges stored with its map grid. The archived grids of the other cities (the
    reference features of the paper) do not store their edges, and the edge files rebuilt after the loss of the local data have
    slightly different extents, so the cell size is recovered from the archived grid itself: the region centroids (km) are
    regressed on their positions in grid units (overlap-weighted mean cell index), and the result is rescaled to the exact cell
    area of the grid file, keeping the fitted aspect ratio."""
    if c == "austin":
        z = np.load(f"{MC}/austin_fine_static.npz"); la, lo = z["lat_edges"], z["lon_edges"]
        return (la[1] - la[0]) * KM, (lo[1] - lo[0]) * KM * np.cos(np.deg2rad(la.mean()))
    g = np.load(f"{G}/grid_{c}.npz"); fr = g["frac"].astype(np.float64); cen = np.load(SP[c])["centroid_km"]
    ii, jj = np.meshgrid(np.arange(40) + 0.5, np.arange(50) + 0.5, indexing="ij"); w = fr.sum(1); ok = w > 0.5 * np.median(w)
    A = np.c_[np.ones(ok.sum()), (fr @ ii.ravel())[ok] / w[ok], (fr @ jj.ravel())[ok] / w[ok]]
    dy = np.linalg.lstsq(A, cen[ok, 1], rcond=None)[0][1]; dx = np.linalg.lstsq(A, cen[ok, 0], rcond=None)[0][2]
    k = np.sqrt(np.median(g["cell_km2"]) / (dy * dx)); assert abs(k - 1) < 0.03, (c, k)
    return dy * k, dx * k


def feats16(c):
    if c == "austin": return np.load(f"{MC}/austin_fine_static.npz")["static"].reshape(2000, 16)
    return np.load(f"{G}/grid_{c}.npz")["X"]


GEO = {c: cell_size(c) for c in CITIES}
X16 = {c: feats16(c).astype(np.float64) for c in CITIES}
for c in CITIES: assert X16[c].shape == (2000, 16), c
pool = np.concatenate([X16[c][:, :13] for c in CITIES])
q25, med, q75 = np.percentile(pool, [25, 50, 75], axis=0); sc = np.where(q75 - q25 < 1e-5, 1.0, q75 - q25)
XS = {c: torch.tensor(np.clip((X16[c][:, :13] - med) / sc, -8, 8), dtype=torch.float32).reshape(40, 50, 13) for c in CITIES}
aT = float(np.median(np.load(SP[TGT])["area_km2"])); ell = np.sqrt(aT); SIG = (0.5 * ell, ell, 2 * ell)


def kern(n, step, s):
    d = (np.arange(n)[:, None] - np.arange(n)[None]) * step; return torch.tensor(np.exp(-0.5 * (d / s) ** 2), dtype=torch.float32)


KY = [torch.stack([kern(40, GEO[c][0], s) for c in CITIES]) for s in SIG]       # per bandwidth: city x 40 x 40
KX = [torch.stack([kern(50, GEO[c][1], s) for c in CITIES]) for s in SIG]       # per bandwidth: city x 50 x 50
XA, ALL = torch.stack([XS[c] for c in CITIES]), torch.arange(len(CITIES))


def blur(A, ci, k):                               # A: B x 40 x 50 x d, ci: city index of each row; separable Gaussian sum over the cells
    return torch.matmul(torch.matmul(KY[k][ci][:, None], A.permute(0, 3, 1, 2)), KX[k][ci][:, None].transpose(-1, -2)).permute(0, 2, 3, 1)


MASS = [blur(torch.ones(len(CITIES), 40, 50, 1), ALL, k) for k in range(len(SIG))]   # kernel mass with every cell visible


def partitions(c, base, n):
    dy, dx = GEO[c]; ii, jj = np.meshgrid(np.arange(40), np.arange(50), indexing="ij")
    P = np.stack([(ii.ravel() + 0.5) * dy, (jj.ravel() + 0.5) * dx], 1); K = int(min(2000, max(2, round(2000 * dy * dx / aT))))
    return [KMeans(K, n_init=1, random_state=base + j).fit(P).labels_ for j in range(n)]


PART = {c: partitions(c, 1000 * SEED, NPART) for c in CITIES}
CHECK = {c: partitions(c, 1000 * SEED + 500, NCHECK) for c in CITIES}
print(f"target {TGT}: a_T {aT:.3f} km2, bandwidths {', '.join(f'{s:.2f}' for s in SIG)} km; superpixels per city "
      + ", ".join(f"{c} {PART[c][0].max() + 1} (cell {GEO[c][0] * GEO[c][1]:.3f} km2)" for c in CITIES) + f" [{time.time() - t0:.0f}s]", flush=True)


class Enc(nn.Module):
    def __init__(s, fin=13, hid=HID, lat=LAT, L=3, ns=3):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(fin, hid), nn.GELU(), nn.Linear(hid, hid)); s.tok = nn.Parameter(torch.zeros(hid))
        s.mix = nn.ModuleList([nn.Sequential(nn.Linear(hid * (1 + ns) + ns, hid), nn.GELU(), nn.Linear(hid, hid)) for _ in range(L)])
        s.norm = nn.ModuleList([nn.LayerNorm(hid) for _ in range(L)])
        s.lat = nn.Linear(hid, lat); s.dec = nn.Sequential(nn.Linear(lat, hid), nn.GELU(), nn.Linear(hid, fin))

    def forward(s, ci, vis):                      # ci: city index of each row (B), vis: B x 40 x 50 bool, True = visible
        v = vis.float()[..., None]
        h = torch.where(vis[..., None], s.inp(XA[ci]), s.tok)
        den = [blur(v, ci, k) for k in range(len(SIG))]; fr = [d / MASS[k][ci] for k, d in enumerate(den)]
        for mix, norm in zip(s.mix, s.norm):
            h = norm(h + mix(torch.cat([h] + [blur(v * h, ci, k) / (den[k] + 1e-6) for k in range(len(SIG))] + fr, -1)))
        z = s.lat(h); return z, s.dec(z)


def hide(lab, rng):                               # a random half of the superpixels of one partition
    K = lab.max() + 1; return np.isin(lab, rng.permutation(K)[:K // 2]).reshape(40, 50)


def masked_mse(xr, X, hid):                       # mean over the rows of the mean squared error on each row's hidden cells
    err = ((xr - X) ** 2).mean(-1); return torch.stack([err[b][hid[b]].mean() for b in range(len(hid))])


torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
enc = Enc(); opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)
for step in range(1, NSTEP + 1):
    hid = torch.tensor(np.stack([hide(PART[c][rng.integers(NPART)], rng) for c in CITIES]))
    _, xr = enc(ALL, ~hid); loss = masked_mse(xr, XA, hid).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step()
    if step % 250 == 0 or step == 1: print(f"  step {step}: masked reconstruction loss {loss.item():.4f} [{time.time() - t0:.0f}s]", flush=True)
enc.eval(); rng_c = np.random.default_rng(10_000 + SEED)
with torch.no_grad():
    print("  check on 5 unused partitions, mean squared error on the hidden cells (standardised features): model | city mean of visible | Gaussian mean of visible", flush=True)
    for b, c in enumerate(CITIES):
        hid = torch.tensor(np.stack([hide(lab, rng_c) for lab in CHECK[c]])); vis = ~hid; ci = torch.full((NCHECK,), b)
        X, v = XA[ci], vis.float()[..., None]; _, xr = enc(ci, vis)
        mu = (X * v).sum((1, 2), keepdim=True) / v.sum((1, 2), keepdim=True); gm = blur(v * X, ci, 1) / (blur(v, ci, 1) + 1e-6)
        e = [masked_mse(p, X, hid).mean().item() for p in (xr, mu.expand_as(X), gm)]
        print(f"    {c:8s} {e[0]:.4f} | {e[1]:.4f} | {e[2]:.4f}   (model / Gaussian {e[0] / e[2]:.2f})", flush=True)
    Z = {}
    for b, c in enumerate(CITIES):                # each cell is hidden once per partition: a random half, then the other half
        H1 = np.stack([hide(lab, rng_c) for lab in PART[c]]); hid = torch.tensor(np.concatenate([H1, ~H1]))
        z, _ = enc(torch.full((len(hid),), b), ~hid); hf = hid[..., None].float(); n = hf.sum(0)
        assert torch.all(n == NPART); Z[c] = ((z * hf).sum(0) / n).reshape(2000, LAT).numpy().astype(np.float64)


def membership(H, r, n_blocks, jobs_stored):    # as build_acs.membership_kmeans: original k-means seeds, verified on the jobs sums
    S = np.load(SP[H]); area, cen = S["area_km2"], S["centroid_km"]; jobs = np.load(f"{HERE}/data/extra_{H}.npz")["jobs"]
    for K in range(n_blocks, n_blocks + 8):
        lab = KMeans(K, n_init=4, random_state=r).fit(cen, sample_weight=area).labels_
        Mm = np.zeros((lab.max() + 1, len(area))); Mm[lab, np.arange(len(area))] = 1; Mm = Mm[Mm.sum(1) > 0]
        if Mm.shape[0] == n_blocks and np.allclose(Mm @ jobs, jobs_stored, rtol=1e-9, atol=1e-6): return Mm
    raise RuntimeError(f"membership of {H} partition {r} with {n_blocks} blocks not reproduced")


def pool_units(frac, Zc):                         # the region-feature formula: equal cell areas within a grid, so area weights = frac
    w = frac.sum(1); E = (frac @ Zc) / np.maximum(w, 1e-9)[:, None]
    if (w < 1e-6).any(): E[w < 1e-6] = np.median(E[w >= 1e-6], axis=0)
    return E.astype(np.float32)


os.makedirs(OUT, exist_ok=True)
units = [(TGT, TGT, None)] + [(f"{H}_for_{TGT}_p{r}" if r else H, H, r) for H, r in HELP[TGT]]
for key, H, r in units:
    frac = np.load(f"{G}/grid_{H}.npz")["frac"].astype(np.float64)
    if r is not None:
        js = np.load(f"{HERE}/data/extra_{key}.npz")["jobs"]; frac = membership(H, r, len(js), js) @ frac
    E = pool_units(frac, Z[H]); n_rows = np.load(f"{HERE}/data/agg/{key}_static.npz" if r else SP[H])["X"].shape[0]
    assert E.shape == (n_rows, LAT), (key, E.shape, n_rows)
    np.save(f"{OUT}/{key}__{TGT}__s{SEED}.npy", E)
open(f"{OUT}/_done__{TGT}__s{SEED}", "w").write(f"{len(units)} unit sets\n")    # marker: every file of this target and seed is written
print(f"  saved context latents for {len(units)} unit sets of target {TGT}, seed {SEED} [{time.time() - t0:.0f}s]", flush=True)
