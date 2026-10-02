"""FULLY STRICT scoring: no oracle city total. Declared 2026-10-01 before running.
City total per slot predicted from sources only:
  level  : target yearly trips = target LODES jobs x geometric mean over sources of (source yearly trips / source jobs)
  rhythm : share of the yearly total falling in each (day-of-week, 30-min slot), averaged over sources
  total[d,s] = level x rhythm[dow(d), s] / (# days with that dow in the target year)
Shares: our models' 10-seed (or 3-seed) ensembles. Diagnostics (use target info, NOT strict): (b) true yearly level +
source rhythm, (c) oracle total (our usual scoring)."""
import os, numpy as np
from datetime import date, timedelta
HERE = os.path.dirname(os.path.abspath(__file__)); MC = f"{HERE}/../multicity"; B = f"{HERE}/../bench77/data"
Ypath = {"chicago": (f"{B}/chicago_2021_ca_30min.npy", date(2021, 1, 1)), "nyc": (f"{B}/nyc_2021_zone_30min.npy", date(2021, 1, 1)),
         "sf": (f"{MC}/data/sf_2023_tract_30min.npy", date(2023, 1, 1))}
jobs = {c: np.load(f"{HERE}/data/extra_{c}.npz")["jobs"].sum() for c in Ypath}
Y = {c: np.load(p).astype(np.float64) for c, (p, _) in Ypath.items()}
dow = {c: np.array([(st + timedelta(days=d)).weekday() for d in range(Y[c].shape[0])]) for c, (_, st) in Ypath.items()}
def rhythm(c):
    tot = Y[c].sum(2); r = np.zeros((7, 48))
    for k in range(7): r[k] = tot[dow[c] == k].sum(0)
    return r / r.sum()
for c in Ypath: print(f"{c}: yearly trips {Y[c].sum():,.0f}, jobs {jobs[c]:,.0f}, trips per job {Y[c].sum()/jobs[c]:.2f}")
CFG = {"chicago": ("nyc+sf", ["nyc", "sf"]), "nyc": ("chicago+sf", ["chicago", "sf"])}
for tgt, (src, S) in CFG.items():
    Yt = Y[tgt]; D = Yt.shape[0]; tot = Yt.sum(2)
    idx = [(d, s) for d in range(D) for s in range(48) if tot[d, s] > 0]
    di, si = np.array([d for d, _ in idx]), np.array([s for _, s in idx]); Yc = Yt[di, si]
    rate = np.exp(np.mean([np.log(Y[c].sum() / jobs[c]) for c in S])); level = jobs[tgt] * rate
    rh = np.mean([rhythm(c) for c in S], 0); ndow = np.bincount(dow[tgt], minlength=7)
    Tstrict = level * rh[dow[tgt][di], si] / ndow[dow[tgt][di]]
    Tlevel = Yt.sum() * rh[dow[tgt][di], si] / ndow[dow[tgt][di]]
    Toracle = tot[di, si]
    print(f"\n===== {tgt.upper()} <- {src}: predicted yearly trips {level:,.0f} vs true {Yt.sum():,.0f} ({100*(level/Yt.sum()-1):+.0f}%) =====")
    tl = np.abs(Tstrict - Toracle).mean() / Toracle.mean()
    print(f"city-total error per slot: {100*tl:.0f}% (strict) | {100*np.abs(Tlevel-Toracle).mean()/Toracle.mean():.0f}% (true level, source rhythm)")
    models = {"uniform": None, "previous best (10)": f"{MC}/multi_{tgt}_from_{src}_seed{{}}.npy",
              "eh (10)": f"{HERE}/out/multi_{tgt}_from_{src}_veh_seed{{}}.npy", "eho (10)": f"{HERE}/out/multi_{tgt}_from_{src}_veho_seed{{}}.npy"}
    print(f"{'model':20s} | {'FULLY STRICT  MAE  RMSE  MAPE':30s} | {'true yearly level  MAE RMSE MAPE':33s} | {'oracle total  MAE RMSE MAPE':28s}")
    for n, pat in models.items():
        if pat is None: p = np.full_like(Yc, 1.0 / Yt.shape[2])
        else:
            ks = range(10) if "(10)" in n else range(3)
            p = np.mean([np.load(pat.format(k)) for k in ks], 0); p /= p.sum(1, keepdims=True)
        out = []
        for T in (Tstrict, Tlevel, Toracle):
            E = p * T[:, None] - Yc; A = np.abs(E); m = Yc >= 5
            out.append((A.mean(), np.sqrt((E ** 2).mean()), 100 * np.mean(A[m] / Yc[m])))
        print(f"{n:20s} | " + " | ".join(f"{a:8.2f} {b:6.2f} {c:6.1f}%      " for a, b, c in out))
