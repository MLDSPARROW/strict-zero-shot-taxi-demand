"""Two no-training ideas (declared 2026-10-01 before running).
(1) Model averaging: shares = 0.5 * eh + 0.5 * eo (10-seed ensembles each); equal weights fixed in advance (no target data).
(2) RHYTHM diagnosis (uses target data, diagnosis only): split the strict per-slot city-total error into
    day-level error (wrong daily totals) and within-day shape error, and test a strict holiday rule:
    US federal holidays use the helpers' Sunday shape and a holiday day-level factor = mean helper (holiday day / same-weekday mean)."""
import numpy as np, io, contextlib
from datetime import timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
src = open("strict_v3.py").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(src[:src.index("for tgt, (srcs, S) in st.CFG.items():")], ns)
st, v2, U, trips, TAXI = ns["st"], ns["v2"], ns["U"], ns["trips"], ns["TAXI"]
def hol_days(c):
    s = st.Ypath[c][1]; n = st.Y[c].shape[0]
    h = set(USFederalHolidayCalendar().holidays(start=str(s), end=str(s + timedelta(days=n))).date)
    return np.array([(s + timedelta(days=d)) in h for d in range(n)])
def strict_T(tgt, S, di, si, holiday=False):
    H = [c for c in TAXI if c != tgt]
    loo = {u: np.mean([abs(np.exp(np.mean([np.log(trips[c] / U[c][u]) for c in H if c != h])) * U[h][u] / trips[h] - 1) for h in H]) for u in U[tgt]}
    unit = min(loo, key=loo.get)
    T = v2.strict_T(tgt, S, di, si, 2)
    if holiday:
        hd = hol_days(tgt); dow = st.dow[tgt]
        fac, sun = [], []
        for c in S:
            tot = st.Y[c].sum(2); hc = hol_days(c); dc = st.dow[c]; day = tot.sum(1)
            fac.append(np.mean([day[d] / day[(dc == dc[d]) & ~hc].mean() for d in np.where(hc)[0]]))
            s_ = tot[dc == 6].sum(0); sun.append(s_ / s_.sum())
        fac, sun = np.mean(fac), np.mean(sun, 0)
        T = T.copy()
        for d in np.where(hd)[0]:
            m = di == d; daytot = T[m].sum() * fac; T[m] = daytot * sun[si[m]]
    return T / T.sum() * U[tgt][unit] * np.exp(np.mean([np.log(trips[c] / U[c][unit]) for c in H]))
for tgt, (srcs, S) in st.CFG.items():
    Yt = st.Y[tgt]; tot = Yt.sum(2); idx = [(a, b) for a in range(Yt.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Yt[di, si]; To = tot[di, si]
    P = {n: np.mean([np.load(f"out/multi_{tgt}_from_{srcs}_v{n}_seed{s}.npy") for s in range(10)], 0) for n in ("eh", "eo")}
    for n in P: P[n] /= P[n].sum(1, keepdims=True)
    P["avg(eh,eo)"] = 0.5 * P["eh"] + 0.5 * P["eo"]
    print(f"\n===== {tgt.upper()} FULLY STRICT, MAE / RMSE / MAPE =====")
    for hol in (False, True):
        T = strict_T(tgt, S, di, si, hol)
        dayT = np.bincount(di, T); dayO = np.bincount(di, To)
        shapeonly = T / dayT[di] * dayO[di]                                     # true daily totals, our within-day shape (diagnosis)
        print(f"-- holiday rule {'ON ' if hol else 'OFF'}: per-slot total error {100*np.abs(T-To).mean()/To.mean():.1f}% | "
              f"daily-total error {100*np.abs(dayT-dayO).mean()/dayO.mean():.1f}% | within-day shape error alone {100*np.abs(shapeonly-To).mean()/To.mean():.1f}%")
        for n, p in P.items():
            E = p * T[:, None] - Yc; A = np.abs(E); m = Yc >= 5
            print(f"     {n:11s} {A.mean():.2f} / {np.sqrt((E**2).mean()):.2f} / {100*np.mean(A[m]/Yc[m]):.1f}%")
