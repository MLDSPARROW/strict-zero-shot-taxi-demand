"""CHEATING ceilings (use target demand; for diagnosis only): how good could ANY static-map + calendar allocator be?
A: true per-region share averaged by (weekday/weekend, 30-min slot) over the whole year, x oracle total
B: true per-region share by (day-of-week, slot, month)  -- even richer calendar
C: true yearly-average share per region (no time variation)
Also the Poisson noise floor: if the model knew each region-slot's exact expected count, MAE from randomness alone."""
import numpy as np
from datetime import date, timedelta
B = "../bench77/data"
for tgt, f in [("chicago", "chicago_2021_ca_30min.npy"), ("nyc", "nyc_2021_zone_30min.npy")]:
    Y = np.load(f"{B}/{f}").astype(np.float64); D, S, R = Y.shape; tot = Y.sum(2)
    dow = np.array([(date(2021, 1, 1) + timedelta(days=d)).weekday() for d in range(D)])
    mon = np.array([(date(2021, 1, 1) + timedelta(days=d)).month for d in range(D)])
    def score(P):
        ok = tot > 0; E = (P - Y)[ok]; Yo = Y[ok]; A = np.abs(E); m = Yo >= 5
        return A.mean(), np.sqrt((E ** 2).mean()), 100 * np.mean(A[m] / Yo[m])
    def alloc(keyfn):
        P = np.zeros_like(Y)
        keys = {}
        for d in range(D): keys.setdefault(keyfn(d), []).append(d)
        for k, ds in keys.items():
            for s in range(S):
                sh = Y[ds, s].sum(0); sh = sh / max(sh.sum(), 1e-9)
                P[ds, s] = tot[ds, s][:, None] * sh[None]
        return P
    print(f"\n=== {tgt.upper()} ({R} regions) -- CHEATING ceilings, oracle total, MAE / RMSE / MAPE>=5 ===")
    print("A weekday/weekend x slot shares : %.2f / %.2f / %.1f%%" % score(alloc(lambda d: dow[d] >= 5)))
    print("B dow x month x slot shares     : %.2f / %.2f / %.1f%%" % score(alloc(lambda d: (dow[d], mon[d]))))
    sh = Y.sum((0, 1)); sh /= sh.sum(); print("C yearly share, no time         : %.2f / %.2f / %.1f%%" % score(tot[:, :, None] * sh[None, None]))
    rng = np.random.default_rng(0); lam = alloc(lambda d: (dow[d], mon[d])); Ys = rng.poisson(lam)
    ok = tot > 0; A = np.abs(lam - Ys)[ok]; m = Ys[ok] >= 5
    print("Poisson noise floor (perfect expected counts, simulated): MAE %.2f, MAPE>=5 %.1f%%" % (A.mean(), 100 * np.mean(A[m] / Ys[ok][m])))
