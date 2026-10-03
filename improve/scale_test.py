"""Evaluation of each step of the method separately (development targets), with trip errors only.
Steps: native helpers -> scale-matched helpers, one partition -> ten partitions -> average of the two heads.
For each step and head: 10-run ensembles, trip MAE and RMSE, mean daily MAE difference with a 7-day block-bootstrap
95% interval (2,000 draws), and the number of the 10 paired single runs (same seed) in which the trip MAE is lower.
Uses the strict city totals saved by predict_final.py. Usage: python scale_test.py"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); B = f"{HERE}/../bench77/data"
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy"}
NAT = {"chicago": "nyc+sf", "nyc": "chicago+sf"}
ONE = {"chicago": "nyc_for_chicago+sf_for_chicago", "nyc": "chicago+sf_for_nyc"}
TEN = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}


def boot(x, rng):
    nb = len(x) // 7
    return [x[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(x) - 7, nb)])].mean() for _ in range(2000)]


for T in ("chicago", "nyc"):
    rng = np.random.default_rng(0); P = np.load(f"{HERE}/out_cal/pred_{T}.npz"); di, si, Tt = P["di"], P["si"], P["total"]
    Y = np.load(YP[T]).astype(np.float64); Yc = Y[di, si]
    norm = lambda p: p / p.sum(1, keepdims=True)
    def daily(p):
        A = np.abs(norm(p) * Tt[:, None] - Yc).mean(1); return np.bincount(di, A) / np.bincount(di)
    def score(p):
        E = norm(p) * Tt[:, None] - Yc; return np.abs(E).mean(), np.sqrt((E ** 2).mean())
    files = {h: {"native": [f"{HERE}/out/multi_{T}_from_{NAT[T]}_v{h}_seed{s}.npy" for s in range(10)],
                 "one partition": [f"{HERE}/out/multi_{T}_from_{ONE[T]}_v{h}_seed{s}.npy" for s in range(10)],
                 "ten partitions": [f"{HERE}/out/multi_{T}_from_{TEN[T](r)}_v{h}_seed{r-1}.npy" for r in range(1, 11)]} for h in ("eh", "eo")}
    print(f"\n===== {T.upper()}: trip MAE / RMSE of 10-run ensembles =====")
    E = {h: {k: norm(np.mean([np.load(f) for f in fs], 0)) for k, fs in d.items()} for h, d in files.items()}
    for h in E:
        for k in E[h]: print(f"   {h} {k:15s} {score(E[h][k])[0]:.2f} / {score(E[h][k])[1]:.2f}")
    for nm in ("native", "one partition", "ten partitions"):
        a = 0.5 * E["eh"][nm] + 0.5 * E["eo"][nm]; print(f"   average of heads, {nm:15s} {score(a)[0]:.2f} / {score(a)[1]:.2f}")
    print("   step tests (mean daily MAE difference, 95% block-bootstrap interval; paired single runs with lower MAE)")
    for h in ("eh", "eo"):
        for a, b in (("native", "one partition"), ("one partition", "ten partitions")):
            d = daily(E[h][b]) - daily(E[h][a]); bs = boot(d, rng)
            ra = np.array([score(np.load(f))[0] for f in files[h][a]]); rb = np.array([score(np.load(f))[0] for f in files[h][b]])
            print(f"   {h}: {a} -> {b}: {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}], "
                  f"better on {100*(d<0).mean():.0f}% of days; single runs better {(rb < ra).sum()}/10, mean run MAE {ra.mean():.2f} -> {rb.mean():.2f}")
    for h in ("eh", "eo"):
        d = daily(0.5 * E["eh"]["ten partitions"] + 0.5 * E["eo"]["ten partitions"]) - daily(E[h]["ten partitions"]); bs = boot(d, rng)
        print(f"   head {h} -> average of heads (ten partitions): {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
