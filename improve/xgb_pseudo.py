"""PROTOCOL_V15.md: pseudo-target predictions of the XGBoost baseline of the paper (configuration of xgb_scale.py, unchanged: 400 trees, depth 6, learning rate 0.05, subsample 0.8, column sample 0.8,
800 random slots per helper, Tweedie loss, median/IQR scaling pooled over the helper regions; run r: partition r of the merged helpers, seed r - 1; 5 runs) for every pseudo-target configuration of
v14_gate.py (development and held-out). Each pseudo-target is a share helper of a target, predicted by models trained on the OTHER share helpers only, on the observed slots of the pseudo-target
(as the pseudo-target predictions of v1). Saved to out_xgb/xgbps__<tag>__<sources of run 1>.npy. Restartable. Usage (from improve/): python xgb_pseudo.py [dev|heldout|all]"""
import os, sys, time, ctypes, numpy as np, xgboost as xgb
from datetime import date
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); MC = f"{HERE}/../multicity/data"
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception: pass
s = open(f"{HERE}/xgb_scale.py").read(); nx = {"__file__": f"{HERE}/xgb_scale.py"}; exec(s[:s.index("HEL = {")], nx); cal, helper = nx["cal"], nx["helper"]
g = open(f"{HERE}/v14_gate.py").read(); ng = {"np": np, "os": os, "HERE": HERE}; exec(g[g.index("S = lambda s:"):g.index("SRC = {")], ng); PSEUDO = ng["PSEUDO"]
B77 = f"{HERE}/../bench77/data"; AG = f"{HERE}/data/agg"
START = {"chicago": date(2021, 1, 1), "nyc": date(2021, 1, 1), "sf": date(2023, 1, 1)}


def static(tag):
    if tag in ("chicago", "nyc"): return np.load(f"{B77}/region_static_{tag}.npz")
    if os.path.exists(f"{MC}/region_static_{tag}.npz"): return np.load(f"{MC}/region_static_{tag}.npz")
    return np.load(f"{AG}/{tag}_static.npz")


def start_of(tag): return START["sf"] if tag.startswith("sf") else START["nyc"]


which = sys.argv[1] if len(sys.argv) > 1 else "all"
targets = {"dev": ["chicago", "nyc"], "heldout": ["sf", "sf_zcta", "dc", "dc_cluster", "austin"], "all": ["chicago", "nyc", "sf", "sf_zcta", "dc", "dc_cluster", "austin"]}[which]
done = set()
for T in targets:
    for tag, srcs in PSEUDO[T]:
        key = f"xgbps__{tag}__{srcs(1)}"
        if key in done or os.path.exists(f"{HERE}/out_xgb/{key}.npy"): done.add(key); continue
        t0 = time.time()
        hh = helper(tag)                                              # the pseudo-target as a "helper": observed slots, shares, calendar index
        di, si = hh["di"], hh["si"]; Xt0 = hh["X"]; R = Xt0.shape[0]; ft = cal(hh["st"]); keys = {}
        for k in range(len(di)): keys.setdefault(tuple(ft(di[k], si[k])), []).append(k)
        runs = []
        for r in range(1, 6):
            hs = [helper(h) for h in srcs(r).split(",")]
            pool = np.concatenate([h["X"] for h in hs]); q25, med, q75 = np.percentile(pool, [25, 50, 75], 0); scl = np.where(q75 - q25 < 1e-5, 1, q75 - q25)
            Xs, ys = [], []; rng = np.random.default_rng(r - 1)
            for h in hs:
                Xc = np.clip((h["X"] - med) / scl, -8, 8); f = cal(h["st"]); Rc = Xc.shape[0]
                for k in rng.choice(len(h["di"]), 800, replace=False):
                    Xs.append(np.hstack([Xc, np.repeat(np.array(f(h["di"][k], h["si"][k]))[None], Rc, 0)])); ys.append(h["Sh"][k] * Rc)
            mdl = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:tweedie", n_jobs=4, random_state=r - 1).fit(np.vstack(Xs), np.concatenate(ys))
            Xt = np.clip((Xt0 - med) / scl, -8, 8); p = np.zeros((len(di), R), np.float32)
            for kk, ks in keys.items():
                pr = np.clip(mdl.predict(np.hstack([Xt, np.repeat(np.array(kk)[None], R, 0)])), 1e-9, None); p[ks] = pr / pr.sum()
            runs.append(p)
        np.save(f"{HERE}/out_xgb/{key}.npy", np.mean(runs, 0).astype(np.float32)); done.add(key)
        print(f"{key}: {R} regions, {len(di)} slots, 5 runs saved ({time.time() - t0:.0f} s)", flush=True)
