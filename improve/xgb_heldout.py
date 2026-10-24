"""PROTOCOL_V10.md: the XGBoost baseline of the paper (configuration of xgb_scale.py, unchanged) trained for the five held-out targets on the same
helpers as the final method (run r: partition r of the merged helpers, XGBoost and slot-sampling seed r - 1; San Francisco: native helpers, seeds
0-9). Helper data only: the demand of the target is never read. Predictions on the target's calendar slots (the slots of coarse_eval.py and
austin_eval.py); the 10-run ensemble shares are saved to out_xgb/xgb_sm_<target>.npy.
Usage (from improve/): python xgb_heldout.py [target ...]      default: sf dc sf_zcta dc_cluster austin"""
import os, sys, time, ctypes, numpy as np, xgboost as xgb
from datetime import date
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); MC = f"{HERE}/../multicity/data"
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)    # below normal: keeps the laptop usable
except Exception: pass
s = open(f"{HERE}/xgb_scale.py").read(); nx = {"__file__": f"{HERE}/xgb_scale.py"}; exec(s[:s.index("HEL = {")], nx); cal, helper = nx["cal"], nx["helper"]
s = open(f"{HERE}/coarse_eval.py").read(); nc = {"__file__": f"{HERE}/coarse_eval.py"}; exec(s[:s.index("norm = lambda")], nc); totals = nc["totals"]
TG = {"sf": ("sf", date(2023, 1, 1), lambda r: ["chicago", "nyc"]),
      "dc": ("dc", date(2021, 1, 1), lambda r: ["chicago", "nyc", f"sf_for_dc_p{r}"]),
      "sf_zcta": ("sf", date(2023, 1, 1), lambda r: ["chicago", f"nyc_for_sf_zcta_p{r}"]),
      "dc_cluster": ("dc", date(2021, 1, 1), lambda r: ["chicago", f"nyc_for_dc_cluster_p{r}", f"sf_for_dc_cluster_p{r}"]),
      "austin": ("austin", date(2016, 6, 4), lambda r: ["chicago", "nyc", f"sf_for_austin_p{r}"])}


def slots(tag, city):
    if tag == "austin":                                   # level and rhythm of austin_eval.py (helper data and public data only)
        a = open(f"{HERE}/austin_eval.py").read(); na = {"__file__": f"{HERE}/austin_eval.py"}
        exec(a[:a.index('Y = np.load(f"{MC}/austin_2016_cell_30min.npy")')], na); return na["di"], na["si"]
    di, si, Tt, unit, lev = totals(city, tag); return di, si


for tag in (sys.argv[1:] or list(TG)):
    city, start, H = TG[tag]; t0 = time.time()
    di, si = slots(tag, city); Xt0 = np.load(f"{MC}/region_static_{tag}.npz")["X"][:, :18].astype(np.float64); R = Xt0.shape[0]
    ft = cal(start); keys = {}
    for k in range(len(di)): keys.setdefault(tuple(ft(di[k], si[k])), []).append(k)
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
        Xt = np.clip((Xt0 - med) / scl, -8, 8); p = np.zeros((len(di), R), np.float32)
        for key, ks in keys.items():
            pr = np.clip(mdl.predict(np.hstack([Xt, np.repeat(np.array(key)[None], R, 0)])), 1e-9, None); p[ks] = pr / pr.sum()
        runs.append(p); print(f"   {tag} run {r} done ({time.time() - t0:.0f} s)", flush=True)
    np.save(f"{HERE}/out_xgb/xgb_sm_{tag}.npy", np.mean(runs, 0).astype(np.float32))
    print(f"{tag}: {R} regions, {len(di)} calendar slots, helpers {H(1)} ... saved ({time.time() - t0:.0f} s)", flush=True)
