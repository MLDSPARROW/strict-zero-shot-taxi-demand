"""[FINAL METHOD VERSION, paper v2] COST OF HISTORY FOR OUR OWN METHOD (declared 2026-10-02). Test month: December 2021. MAE / RMSE per area per 30 min.
 A  strict          : our best strict model (partition-diverse scale-matched helpers; eo Chicago / eh NYC), strict LEVEL x RHYTHM.
 B  long history    : the SAME model trained on the target's own Jan-Nov 2021 data (3 runs); city total per slot = the target's
                      mean city total for the same weekday x slot over the last 4 weeks (Nov 3-30). No December data used.
 C  recent history  : A's prediction corrected with the target's recent counts: XGBoost (Tweedie) on [A's predicted count,
                      lags t-1, t-2, t-3, t-48, t-336, mean of last 12 slots, calendar], trained Jan-Oct, early stop Nov, 1 step ahead.
 Cost = A minus B / C."""
import numpy as np, io, contextlib, os, xgboost as xgb
from datetime import date, timedelta
src = open("idea_avg_rhythm.py").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(src[:src.rindex("for tgt, (srcs, S) in st.CFG.items():")], ns)
st, strict_T = ns["st"], ns["strict_T"]
CFG = {"chicago": ("eo", ["nyc_for_chicago", "sf_for_chicago"]), "nyc": ("eh", ["chicago", "sf_for_nyc"])}
def err(P, Yd): E = P - Yd; return np.abs(E).mean(), np.sqrt((E ** 2).mean())
for tgt, (fl, H) in CFG.items():
    srcs, S = st.CFG[tgt]; Y = st.Y[tgt]; D, _, R = Y.shape; tot = Y.sum(2)
    days = [date(2021, 1, 1) + timedelta(days=d) for d in range(D)]; mon = np.array([d.month for d in days]); dow = np.array([d.weekday() for d in days])
    dec = np.where(mon == 12)[0]; Yd = Y[dec]
    idx = [(a, b) for a in range(D) for b in range(48) if tot[a, b] > 0]; di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx])
    pA = np.load(f"out/FINAL_{tgt}_shares.npy"); runs = range(10); T = strict_T(tgt, S, di, si); A = np.zeros_like(Y); A[di, si] = pA * T[:, None]
    rows = [("A. OUR METHOD (final), strict", "none", *err(A[dec], Yd))]
    fb = {h: [f"out/multi_{tgt}_from_{tgt}_hist_v{h}_seed{s}.npy" for s in range(3)] for h in ("eh", "eo")}
    if all(os.path.exists(f) for v in fb.values() for f in v):
        pB = 0.0
        for h in ("eh", "eo"):
            q = np.mean([np.load(f) for f in fb[h]], 0); pB = pB + 0.5 * q / q.sum(1, keepdims=True)
        last4 = np.where((mon == 11) & (np.array([d.day for d in days]) >= 3))[0]
        tot4 = np.stack([tot[last4[dow[last4] == k]].mean(0) for k in range(7)])
        Bm = np.zeros_like(Y); Bm[di, si] = pB * tot4[dow[di], si][:, None]
        rows.append(("B. OUR METHOD + long history (3 runs per head)", "Jan-Nov of target", *err(Bm[dec], Yd)))
    flat = Y.reshape(-1, R); Af = A.reshape(-1, R); tt = np.arange(336, flat.shape[0]); tm = mon[tt // 48]
    def feats(ts):
        L = [Af[ts]] + [flat[ts - k] for k in (1, 2, 3, 48, 336)] + [np.stack([flat[ts - k] for k in range(1, 13)]).mean(0)]
        cal = np.stack([np.sin(2*np.pi*(ts % 48)/48), np.cos(2*np.pi*(ts % 48)/48), dow[ts // 48] / 6, (dow[ts // 48] >= 5).astype(float)], 1)
        return np.c_[np.stack(L, 2).reshape(-1, len(L)), np.repeat(cal, R, 0)], flat[ts].reshape(-1)
    rng = np.random.default_rng(0); tr = rng.choice(tt[tm <= 10], 6000 if R > 100 else 12000, replace=False); va = tt[tm == 11]; td = np.arange(dec[0] * 48, (dec[-1] + 1) * 48)
    Xtr, ytr = feats(tr); Xva, yva = feats(va); Xte, _ = feats(td)
    m = xgb.XGBRegressor(n_estimators=1500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:tweedie", n_jobs=8, early_stopping_rounds=50)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    rows.append(("C. OUR METHOD + recent history (30 min ahead)", "Jan-Oct + last 6 h", *err(m.predict(Xte).reshape(Yd.shape), Yd)))
    a = rows[0]
    print(f"\n===== {tgt.upper()} — December 2021, MAE / RMSE per area per 30 min =====")
    for r in rows:
        print(f"{r[0]:48s} target taxi data: {r[1]:20s} MAE {r[2]:5.2f}  RMSE {r[3]:6.2f} | cost of NOT using history: MAE {a[2]-r[2]:+5.2f} ({100*(a[2]/r[2]-1):+4.0f}%)  RMSE {a[3]-r[3]:+6.2f} ({100*(a[3]/r[3]-1):+4.0f}%)")
