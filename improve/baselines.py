"""Simple strict zero-shot BASELINES for the paper (declared 2026-10-01 before running). No target taxi data anywhere.
  jobs     : area share = LODES jobs in area / city jobs (same for every slot)
  resid    : same with LODES residents
  jobs+res : share proportional to jobs + residents
  gravity  : share_r ∝ jobs_r * sum_s resid_s * exp(-d_rs / lam); lam chosen on SOURCE cities only (best mean source MAE_share)
  xgboost  : gradient-boosted trees on the 18 static map features + 9 calendar features, trained on source (area, slot)
             relative shares (share x #areas), Tweedie loss, 800 random slots per source city; target predictions
             normalised per slot. Feature scaling pooled over source areas (as in the main model).
Scored two ways: with help (true city total per slot) and fully strict (city total guessed as in strict_total.py)."""
import os, numpy as np, xgboost as xgb
from datetime import date, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
HERE = os.path.dirname(os.path.abspath(__file__)); MC = f"{HERE}/../multicity"; B = f"{HERE}/../bench77/data"
CITY = {"chicago": (f"{B}/chicago_2021_ca_30min.npy", f"{B}/region_static_chicago.npz", date(2021, 1, 1)),
        "nyc": (f"{B}/nyc_2021_zone_30min.npy", f"{B}/region_static_nyc.npz", date(2021, 1, 1)),
        "sf": (f"{MC}/data/sf_2023_tract_30min.npy", f"{MC}/data/region_static_sf.npz", date(2023, 1, 1))}
def cal(start):
    hol = set(USFederalHolidayCalendar().holidays(start=str(start), end=str(start + timedelta(days=400))).date)
    def f(d, s):
        dd = start + timedelta(days=int(d)); w = dd.weekday()
        return [np.sin(2*np.pi*s/48), np.cos(2*np.pi*s/48), np.sin(2*np.pi*w/7), np.cos(2*np.pi*w/7), float(w >= 5), float(dd in hol),
                np.sin(2*np.pi*dd.month/12), np.cos(2*np.pi*dd.month/12), float(s // 2 in (7, 8, 9, 16, 17, 18))]
    return f
C = {}
for c, (yp, sp, st) in CITY.items():
    Y = np.load(yp).astype(np.float64); S = np.load(sp); E = np.load(f"{HERE}/data/extra_{c}.npz"); tot = Y.sum(2)
    idx = [(d, s) for d in range(Y.shape[0]) for s in range(48) if tot[d, s] > 0]
    di, si = np.array([d for d, _ in idx]), np.array([s for _, s in idx])
    C[c] = dict(Y=Y, X=S["X"][:, :18].astype(np.float64), cen=S["centroid_km"], jobs=E["jobs"], res=E["res"], di=di, si=si,
                Sh=Y[di, si] / tot[di, si][:, None], T=tot[di, si], st=st, dow=np.array([(st + timedelta(days=d)).weekday() for d in range(Y.shape[0])]))
def gravity(c, lam):
    d = np.sqrt(((C[c]["cen"][:, None] - C[c]["cen"][None]) ** 2).sum(2))
    g = C[c]["jobs"] * (np.exp(-d / lam) @ C[c]["res"]); return g / g.sum()
def static_err(c, p): return np.abs(p[None] - C[c]["Sh"]).mean()
def strict_T(tgt, S):
    rate = np.exp(np.mean([np.log(C[c]["Y"].sum() / C[c]["jobs"].sum()) for c in S])); level = C[tgt]["jobs"].sum() * rate
    rh = []
    for c in S:
        tot = C[c]["Y"].sum(2); r = np.array([tot[C[c]["dow"] == k].sum(0) for k in range(7)]); rh.append(r / r.sum())
    rh = np.mean(rh, 0); w = C[tgt]["dow"][C[tgt]["di"]]; nd = np.bincount(C[tgt]["dow"], minlength=7)
    return level * rh[w, C[tgt]["si"]] / nd[w]
def score(tgt, p, T):
    Yc = C[tgt]["Y"][C[tgt]["di"], C[tgt]["si"]]; E = p * T[:, None] - Yc; A = np.abs(E); m = Yc >= 5
    return A.mean(), np.sqrt((E ** 2).mean()), 100 * np.mean(A[m] / Yc[m])
lams = [0.5, 1, 2, 3, 5, 8, 12]
for tgt, S in [("chicago", ["nyc", "sf"]), ("nyc", ["chicago", "sf"])]:
    R = C[tgt]["Y"].shape[2]; n = len(C[tgt]["di"]); preds = {"uniform": np.full(R, 1 / R)}
    preds["jobs"] = C[tgt]["jobs"] / C[tgt]["jobs"].sum(); preds["resid"] = C[tgt]["res"] / C[tgt]["res"].sum()
    jr = C[tgt]["jobs"] + C[tgt]["res"]; preds["jobs+res"] = jr / jr.sum()
    lam = min(lams, key=lambda l: np.mean([static_err(c, gravity(c, l)) for c in S])); preds[f"gravity (lam={lam} km)"] = gravity(tgt, lam)
    pool = np.concatenate([C[c]["X"] for c in S]); q25, med, q75 = np.percentile(pool, [25, 50, 75], 0); sc = np.where(q75 - q25 < 1e-5, 1, q75 - q25)
    Xs, ys = [], []; rng = np.random.default_rng(0)
    for c in S:
        Xc = np.clip((C[c]["X"] - med) / sc, -8, 8); f = cal(C[c]["st"]); Rc = Xc.shape[0]
        for k in rng.choice(len(C[c]["di"]), 800, replace=False):
            cf = np.array(f(C[c]["di"][k], C[c]["si"][k]))
            Xs.append(np.hstack([Xc, np.repeat(cf[None], Rc, 0)])); ys.append(C[c]["Sh"][k] * Rc)
    mdl = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:tweedie", n_jobs=4)
    mdl.fit(np.vstack(Xs), np.concatenate(ys))
    Xt = np.clip((C[tgt]["X"] - med) / sc, -8, 8); f = cal(C[tgt]["st"]); P = np.zeros((n, R))
    keys = {}
    for k in range(n): keys.setdefault((C[tgt]["dow"][C[tgt]["di"][k]], C[tgt]["si"][k], (C[tgt]["st"] + timedelta(days=int(C[tgt]["di"][k]))).month,
                                        tuple(f(C[tgt]["di"][k], C[tgt]["si"][k]))), []).append(k)
    for key, ks in keys.items():
        cf = np.array(key[3]); pr = np.clip(mdl.predict(np.hstack([Xt, np.repeat(cf[None], R, 0)])), 1e-9, None); P[ks] = pr / pr.sum()
    Ts, To = strict_T(tgt, S), C[tgt]["T"]
    print(f"\n===== {tgt.upper()} <- {'+'.join(S)}  (MAE / RMSE / MAPE>=5) =====")
    print(f"{'baseline':24s} | {'with help':24s} | fully strict")
    for name, p in list(preds.items()) + [("xgboost (map+calendar)", P)]:
        p2 = np.repeat(p[None], n, 0) if p.ndim == 1 else p
        a, b = score(tgt, p2, To), score(tgt, p2, Ts)
        print(f"{name:24s} | {a[0]:5.2f} / {a[1]:6.2f} / {a[2]:5.1f}% | {b[0]:5.2f} / {b[1]:6.2f} / {b[2]:5.1f}%", flush=True)
