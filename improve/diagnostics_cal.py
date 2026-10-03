"""EVALUATION diagnostics on the calendar slots (uses target data; not part of the method).
(1) Sources of error at 30 minutes: final strict; final shares with the true city total; perfect allocation (true weekday
    or weekend x slot shares of the target) with the strict or the true city total; Poisson reference level (counts drawn
    from a Poisson distribution around the perfect expected counts).
(2) Allocation-only metrics: share MAE, mean Jensen-Shannon divergence per slot, Spearman correlation of annual region
    totals, recall of the ten busiest regions. True shares are undefined for slots without any trip, which are left out
    of (2) only. Usage: python diagnostics_cal.py chicago nyc sf dc"""
import os, sys, numpy as np
from datetime import timedelta
from scipy.stats import spearmanr
from predict_final import YP, START
HERE = os.path.dirname(os.path.abspath(__file__))
for T in sys.argv[1:]:
    rng = np.random.default_rng(0); P = np.load(f"{HERE}/out_cal/pred_{T}.npz"); di, si, Ts = P["di"], P["si"], P["total"]
    Y = np.load(YP[T]).astype(np.float64); Yc = Y[di, si]; To = Yc.sum(1)
    we = np.array([(START[T] + timedelta(days=d)).weekday() >= 5 for d in range(Y.shape[0])])
    tru = np.zeros((2, 48, Y.shape[2]))
    for k in (0, 1):
        for s in range(48): v = Y[we == bool(k), s].sum(0); tru[k, s] = v / max(v.sum(), 1e-9)
    pt = tru[we[di].astype(int), si]
    sh = {m: P[m] / np.maximum(P[m].sum(1, keepdims=True), 1e-12) for m in ("jobs", "jobs_res", "base", "final")}
    def sc(p, TT): E = p * TT[:, None] - Yc; return f"{np.abs(E).mean():.2f} / {np.sqrt((E**2).mean()):.2f}"
    lam = pt * To[:, None]; sim = rng.poisson(lam)
    print(f"\n===== {T.upper()} ({len(di)} calendar slots) =====")
    print(f"   (1) final strict {sc(sh['final'], Ts)} | final shares + true total {sc(sh['final'], To)} | perfect allocation + strict total "
          f"{sc(pt, Ts)} | perfect allocation + true total {sc(pt, To)} | Poisson reference level {np.abs(lam - sim).mean():.2f} / {np.sqrt(((lam - sim)**2).mean()):.2f}")
    ok = To > 0; Sh = Yc[ok] / To[ok][:, None]
    for m, nm in (("jobs", "Jobs-proportional"), ("jobs_res", "Jobs and residents"), ("base", "Base allocator"), ("final", "Final")):
        p = sh[m][ok]; q = np.clip(p, 1e-12, 1); s_ = np.clip(Sh, 1e-12, 1); mM = 0.5 * (q + s_)
        js = 0.5 * (np.sum(q * np.log(q / mM), 1) + np.sum(np.where(Sh > 0, s_ * np.log(s_ / mM), 0), 1))
        ptot, ttot = (p * To[ok][:, None]).sum(0), Yc[ok].sum(0); top = set(np.argsort(-ttot)[:10])
        print(f"   (2) {nm:20s} share MAE {np.abs(p - Sh).mean():.5f} | JS {js.mean():.4f} | Spearman {spearmanr(ptot, ttot)[0]:.3f} | "
              f"top-10 recall {len(top & set(np.argsort(-ptot)[:10]))/10:.1f}")
