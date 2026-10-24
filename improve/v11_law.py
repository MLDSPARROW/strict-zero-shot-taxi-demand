"""DESCRIPTIVE check (helper data and development targets only): is the cross-sectional law log(yearly taxi share) = a + beta * log(people share)
(people = LODES jobs + ACS population of the region, airport regions left out) stable across taxi cities and scales? And where do the predictions
of the final method and its heads sit against it on Chicago and NYC? Weighted least squares with weights = people share (so that tiny regions
do not dominate). Usage (from improve/): python v11_law.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
B, MC, AG, D = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data", f"{HERE}/data/agg", f"{HERE}/data"
NAT = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy", "sf": f"{MC}/sf_2023_tract_30min.npy"}


def people(tag):
    j = np.load(f"{D}/extra_{tag}.npz")["jobs"].astype(np.float64); p = np.load(f"{D}/acs_{tag}.npz")["pop"].astype(np.float64)
    e = np.load(f"{D}/enpl_{tag}.npy").astype(np.float64) if os.path.exists(f"{D}/enpl_{tag}.npy") else np.zeros_like(j)
    return (j + p) / (j + p).sum(), e > 0


def law(s, p, air):
    m = (s > 0) & (p > 0) & ~air; x, y, w = np.log(p[m]), np.log(s[m]), p[m]
    X = np.c_[np.ones(m.sum()), x]; W = np.diag(w / w.sum()) if m.sum() < 400 else None
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    r = y - X @ beta; r2 = 1 - np.sum(w * r ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return beta[1], r2, m.sum()


def yearly(tag):
    Y = np.load(NAT[tag]) if tag in NAT else np.load(f"{AG}/{tag}_Y.npy"); s = Y.sum((0, 1)).astype(np.float64); return s / s.sum()


print("===== true yearly taxi shares against people shares, airports left out (beta = slope, weighted R2, regions used) =====")
rows = {"native": ["chicago", "nyc", "sf"], "for Chicago (scale of the Chicago target)": [f"nyc_for_chicago_p{r}" for r in range(1, 11)] + [f"sf_for_chicago_p{r}" for r in range(1, 11)],
        "for NYC (scale of the NYC target)": [f"sf_for_nyc_p{r}" for r in range(1, 11)]}
for g, tags in rows.items():
    print(f"   -- {g}")
    by = {}
    for t in tags:
        try:
            p, air = people(t); b, r2, n = law(yearly(t), p, air); by.setdefault(t.split("_p")[0], []).append((b, r2, n))
        except FileNotFoundError as e: print(f"      {t}: missing {os.path.basename(str(e).split(chr(39))[-2]) if chr(39) in str(e) else e}")
    for k, v in by.items():
        v = np.array(v); print(f"      {k:16s} beta {v[:, 0].mean():.2f}" + (f" (range {v[:, 0].min():.2f} to {v[:, 0].max():.2f} over {len(v)} partitions)" if len(v) > 1 else "") + f"  R2 {v[:, 1].mean():.2f}  regions {int(v[:, 2].mean())}")
# predictions of the final method and its heads on the development targets (10-run ensembles of out/, demand-reading mode)
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
norm = lambda q: q / q.sum(1, keepdims=True)
print("\n===== predicted yearly shares of the development targets against the same law =====")
for T in ("chicago", "nyc"):
    Y = np.load(NAT[T]).astype(np.float64); tot = Y.sum(2); di, si = np.nonzero(tot > 0); p, air = people(T)
    b, r2, n = law(yearly(T), p, air); print(f"   {T}: TRUE beta {b:.2f} (R2 {r2:.2f})")
    H = {h: np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0) for h in ("eh", "eo", "ep")}
    H["v1"] = 0.5 * H["eh"] + 0.5 * H["eo"]
    for h in ("v1", "eh", "eo", "ep"):
        sy = (H[h] * tot[di, si][:, None]).sum(0); sy /= sy.sum(); b, r2, n = law(sy, p, air)
        print(f"      predicted by {h:3s}: beta {b:.2f} (R2 {r2:.2f}); airport share predicted {100 * sy[air].sum():.1f}% vs true {100 * yearly(T)[air].sum():.1f}%")
