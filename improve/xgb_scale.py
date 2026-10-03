"""Non-neural scale-matching check (PROTOCOL_XGB_SCALE.md). The XGBoost baseline of the paper, trained on native helpers
and on scale-matched helpers, 10 runs each. Training uses helper data only; the target's demand is read only for scoring.
Usage: python xgb_scale.py"""
import os, numpy as np, xgboost as xgb
from datetime import date, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
HERE = os.path.dirname(os.path.abspath(__file__)); B = f"{HERE}/../bench77/data"; MC = f"{HERE}/../multicity/data"; AG = f"{HERE}/data/agg"
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy", "sf": f"{MC}/sf_2023_tract_30min.npy"}
SP = {"chicago": f"{B}/region_static_chicago.npz", "nyc": f"{B}/region_static_nyc.npz", "sf": f"{MC}/region_static_sf.npz"}


def cal(start):
    hol = set(USFederalHolidayCalendar().holidays(start=str(start), end=str(start + timedelta(days=400))).date)
    def f(d, s):
        dd = start + timedelta(days=int(d)); w = dd.weekday()
        return [np.sin(2*np.pi*s/48), np.cos(2*np.pi*s/48), np.sin(2*np.pi*w/7), np.cos(2*np.pi*w/7), float(w >= 5), float(dd in hol),
                np.sin(2*np.pi*dd.month/12), np.cos(2*np.pi*dd.month/12), float(s // 2 in (7, 8, 9, 16, 17, 18))]
    return f


def helper(name):                                   # name: native city or scale-matched tag in data/agg
    if name in YP: Y, S = np.load(YP[name]), np.load(SP[name])
    else: Y, S = np.load(f"{AG}/{name}_Y.npy"), np.load(f"{AG}/{name}_static.npz")
    Y = Y.astype(np.float64); tot = Y.sum(2); di, si = np.nonzero(tot > 0)
    return dict(X=S["X"][:, :18].astype(np.float64), di=di, si=si, Sh=Y[di, si] / tot[di, si][:, None],
                st=date(2023, 1, 1) if name.startswith("sf") else date(2021, 1, 1))


def boot(x, rng):
    nb = len(x) // 7
    return [x[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(x) - 7, nb)])].mean() for _ in range(2000)]


HEL = {"chicago": {"native": lambda r: ["nyc", "sf"], "scale-matched": lambda r: [f"nyc_for_chicago_p{r}", f"sf_for_chicago_p{r}"]},
       "nyc": {"native": lambda r: ["chicago", "sf"], "scale-matched": lambda r: ["chicago", f"sf_for_nyc_p{r}"]}}
for T in ("chicago", "nyc"):
    P = np.load(f"{HERE}/out_cal/pred_{T}.npz"); di, si, Tt = P["di"], P["si"], P["total"]
    Yc = np.load(YP[T]).astype(np.float64)[di, si]; Xt0 = np.load(SP[T])["X"][:, :18].astype(np.float64); R = Xt0.shape[0]
    ft = cal(date(2021, 1, 1)); keys = {}
    for k in range(len(di)): keys.setdefault(tuple(ft(di[k], si[k])), []).append(k)
    res = {}
    for cond, H in HEL[T].items():
        runs = []
        for r in range(1, 11):
            hs = [helper(h) for h in H(r)]
            pool = np.concatenate([h["X"] for h in hs]); q25, med, q75 = np.percentile(pool, [25, 50, 75], 0); scl = np.where(q75 - q25 < 1e-5, 1, q75 - q25)
            Xs, ys = [], []; rng = np.random.default_rng(r - 1)
            for h in hs:
                Xc = np.clip((h["X"] - med) / scl, -8, 8); f = cal(h["st"]); Rc = Xc.shape[0]
                for k in rng.choice(len(h["di"]), 800, replace=False):
                    Xs.append(np.hstack([Xc, np.repeat(np.array(f(h["di"][k], h["si"][k]))[None], Rc, 0)])); ys.append(h["Sh"][k] * Rc)
            mdl = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   objective="reg:tweedie", n_jobs=4, random_state=r - 1).fit(np.vstack(Xs), np.concatenate(ys))
            Xt = np.clip((Xt0 - med) / scl, -8, 8); p = np.zeros((len(di), R))
            for key, ks in keys.items():
                pr = np.clip(mdl.predict(np.hstack([Xt, np.repeat(np.array(key)[None], R, 0)])), 1e-9, None); p[ks] = pr / pr.sum()
            runs.append(p)
        ens = np.mean(runs, 0); E = ens * Tt[:, None] - Yc; A = np.abs(E)
        single = [np.abs(q * Tt[:, None] - Yc).mean() for q in runs]
        res[cond] = (A.mean(), np.sqrt((E ** 2).mean()), np.bincount(di, A.mean(1)) / np.bincount(di), np.mean(single), np.std(single, ddof=1))
        print(f"{T:8s} XGBoost {cond:14s} MAE {res[cond][0]:.2f} RMSE {res[cond][1]:.2f} | single runs MAE {res[cond][3]:.2f} +- {res[cond][4]:.2f}", flush=True)
    d = res["scale-matched"][2] - res["native"][2]; b = boot(d, np.random.default_rng(0))
    print(f"{T:8s} scale-matched minus native: mean daily MAE difference {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], "
          f"scale-matched better on {100*(d<0).mean():.0f}% of days", flush=True)
