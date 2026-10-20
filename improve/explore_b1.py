"""EXPLORATION (development targets Chicago and NYC only; geometry plus the already-existing development runs; no new training, no held-out city).
Question for the idea 'match the distribution of region areas, not only the median' (an external suggestion): (1) how far is the
area distribution of our scale-matched blocks from that of the target at other quantiles; (2) does that distance, per partition,
predict how good the run trained on that partition is? Run quality = mean absolute share error (MAE_share) of run r, mean of heads
eh and eo, from the confirmation runs of PROTOCOL_V2 (demand-reading mode). Usage (from improve/): python explore_b1.py"""
import os, numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); B = f"{HERE}/../bench77/data"; AG = f"{HERE}/data/agg"
Q = [0.1, 0.25, 0.5, 0.75, 0.9]
area = lambda p: np.load(p)["area_km2"]
TG = {"chicago": dict(y=f"{B}/chicago_2021_ca_30min.npy", tgt=area(f"{B}/region_static_chicago.npz"),
                      helpers=lambda r: [f"{AG}/nyc_for_chicago_p{r}_static.npz", f"{AG}/sf_for_chicago_p{r}_static.npz"], src=lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}"),
      "nyc": dict(y=f"{B}/nyc_2021_zone_30min.npy", tgt=area(f"{B}/region_static_nyc.npz"),
                  helpers=lambda r: [f"{AG}/sf_for_nyc_p{r}_static.npz"], src=lambda r: f"chicago+sf_for_nyc_p{r}")}
sig = lambda a: np.quantile(np.log(a), Q)
for T, c in TG.items():
    Y = np.load(c["y"]).astype(np.float64); tot = Y.sum(2); idx = [(d, s) for d in range(Y.shape[0]) for s in range(48) if tot[d, s] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Sh = Y[di, si] / tot[di, si][:, None]
    ts = sig(c["tgt"]); print(f"\n===== {T}: target regions {len(c['tgt'])}, area quantiles 10/25/50/75/90 % (km2): " + " ".join(f"{np.exp(q):.2f}" for q in ts) + f"; sd of log area {np.log(c['tgt']).std():.2f}")
    J, MS = [], []
    for r in range(1, 11):
        js = []
        for hp in c["helpers"](r):
            a = area(hp); js.append(np.abs(sig(a) - ts));
            if r == 1: print(f"   partition 1, {os.path.basename(hp)[:-11]:24s} {len(a):4d} blocks, quantiles: " + " ".join(f"{np.exp(q):.2f}" for q in sig(a)) + f"; sd of log area {np.log(a).std():.2f}")
        J.append(np.mean(js))                       # mean absolute log-quantile discrepancy, averaged over the helpers of this run
        ms = [np.abs(norm - Sh).mean() for norm in [np.load(f"{HERE}/out/multi_{T}_from_{c['src'](r)}_v{h}_seed{r - 1}.npy") for h in ("eh", "eo")]]
        MS.append(np.mean(ms))
    J, MS = np.array(J), np.array(MS)
    print(f"   discrepancy J per partition (mean |log-quantile difference|): " + " ".join(f"{x:.2f}" for x in J))
    print(f"   run MAE_share per partition (mean of eh, eo, x1000):         " + " ".join(f"{1000 * x:.2f}" for x in MS))
    rho, p = spearmanr(J, MS); print(f"   Spearman(J, MAE_share) over the 10 partitions: rho = {rho:+.2f}, p = {p:.2f}; MAE_share spread across partitions: {100 * MS.std() / MS.mean():.1f}% (sd / mean)")
print("\ndone")
