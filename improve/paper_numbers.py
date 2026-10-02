"""ALL numbers for paper version 2 (2026-10-02), recomputed from saved predictions. Fully strict unless stated.
Final method = equal-weight average of the two allocator heads (eh, eo), each a 10-run partition-diverse scale-matched ensemble."""
import json, numpy as np, io, contextlib, xgboost as xgb
from datetime import date, timedelta
src = open("idea_avg_rhythm.py", encoding="utf-8").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(src[:src.rindex("for tgt, (srcs, S) in st.CFG.items():")], ns)
st, strict_T, v2, U, trips, TAXI = ns["st"], ns["strict_T"], ns["v2"], ns["U"], ns["trips"], ns["TAXI"]
B, M = "../bench77/data", "../multicity"
SCALE = {"chicago": ["nyc_for_chicago", "sf_for_chicago"], "nyc": ["chicago", "sf_for_nyc"]}
def ens(files): p = np.mean([np.load(f) for f in files], 0); return p / p.sum(1, keepdims=True)
def multi(tgt, n): return [f"out/multi_{tgt}_from_{'+'.join(h if h == 'chicago' else f'{h}_p{r}' for h in SCALE[tgt])}_v{n}_seed{r-1}.npy" for r in range(1, 11)]
OUT = {}
def line(*a): s = " ".join(str(x) for x in a); print(s); OUT.setdefault("log", []).append(s)
line("=== TABLE 1: data ===")
for c, y in [("chicago", 2021), ("nyc", 2021), ("sf", 2023), ("dc", 2021)]:
    S = np.load(f"{B}/region_static_{c}.npz") if c in ("chicago", "nyc") else np.load(f"{M}/data/region_static_{c}.npz")
    yp = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy", "sf": f"{M}/data/sf_2023_tract_30min.npy", "dc": f"{M}/data/dc_2021_tract_30min.npy"}[c]
    ntr = int(round(float(np.load(yp).astype(np.float64).sum())))   # DC halves are fractional (hourly split), so sum in float64
    line(f"{c}: year {y}, regions {len(S['area_km2'])}, median area {np.median(S['area_km2']):.2f} km2, trips {ntr:,}, jobs {U[c]['jobs']:,.0f}, transit commuters {U[c]['transit commuters']:,.0f}, trips per transit commuter {ntr/U[c]['transit commuters']:.2f}")
for tgt in ["chicago", "nyc"]:
    srcs, S = st.CFG[tgt]; Y = st.Y[tgt]; tot = Y.sum(2); R = Y.shape[2]
    idx = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]; di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Y[di, si]
    dec = np.array([(date(2021, 1, 1) + timedelta(days=int(d))).month == 12 for d in di]); T = strict_T(tgt, S, di, si)
    def sc(p, TT=T):
        E = p * TT[:, None] - Yc; A = np.abs(E); m = Yc >= 5
        return (A.mean(), np.sqrt((E**2).mean()), 100*np.mean(A[m]/Yc[m]), A[dec].mean(), np.sqrt((E[dec]**2).mean()))
    rows = []
    E_ = np.load(f"data/extra_{tgt}.npz"); n = len(di)
    rows.append(("Uniform split", np.full((n, R), 1 / R)))
    rows.append(("Share by LODES jobs", np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0)))
    jr = E_["jobs"] + E_["res"]; rows.append(("Share by jobs + residents", np.repeat((jr / jr.sum())[None], n, 0)))
    # gravity and XGBoost exactly as baselines.py (native helpers)
    gsrc = open("baselines.py", encoding="utf-8").read(); g = {"__file__": "baselines.py"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(gsrc[:gsrc.index("lams = [")], g)
    Cg = g["C"]; lams = [0.5, 1, 2, 3, 5, 8, 12]
    lam = min(lams, key=lambda l: np.mean([g["static_err"](c, g["gravity"](c, l)) for c in S])); rows.append((f"Gravity (lambda={lam} km)", np.repeat(g["gravity"](tgt, lam)[None], n, 0)))
    pool = np.concatenate([Cg[c]["X"] for c in S]); q25, med, q75 = np.percentile(pool, [25, 50, 75], 0); scl = np.where(q75 - q25 < 1e-5, 1, q75 - q25)
    Xs, ys = [], []; rng = np.random.default_rng(0)
    for c in S:
        Xc = np.clip((Cg[c]["X"] - med) / scl, -8, 8); f = g["cal"](Cg[c]["st"]); Rc = Xc.shape[0]
        for k in rng.choice(len(Cg[c]["di"]), 800, replace=False):
            cf = np.array(f(Cg[c]["di"][k], Cg[c]["si"][k])); Xs.append(np.hstack([Xc, np.repeat(cf[None], Rc, 0)])); ys.append(Cg[c]["Sh"][k] * Rc)
    mdl = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:tweedie", n_jobs=8).fit(np.vstack(Xs), np.concatenate(ys))
    Xt = np.clip((Cg[tgt]["X"] - med) / scl, -8, 8); f = g["cal"](Cg[tgt]["st"]); P = np.zeros((n, R)); keys = {}
    for k in range(n): keys.setdefault(tuple(f(di[k], si[k])), []).append(k)
    for key, ks in keys.items(): pr = np.clip(mdl.predict(np.hstack([Xt, np.repeat(np.array(key)[None], R, 0)])), 1e-9, None); P[ks] = pr / pr.sum()
    rows.append(("XGBoost on map + calendar features", P))
    src_n = "+".join(srcs.split("+"))
    rows.append(("Joint allocator (18 OSM features)", ens([f"{M}/multi_{tgt}_from_{srcs}_seed{s}.npy" for s in range(10)])))
    rows.append(("+ FAA airport branch (head eh)", ens([f"out/multi_{tgt}_from_{srcs}_veh_seed{s}.npy" for s in range(10)])))
    rows.append(("+ area offset instead (head eo)", ens([f"out/multi_{tgt}_from_{srcs}_veo_seed{s}.npy" for s in range(10)])))
    rows.append(("Head eh + scale-matched helpers", ens([f"out/multi_{tgt}_from_{'+'.join(SCALE[tgt])}_veh_seed{s}.npy" for s in range(10)])))
    rows.append(("Head eo + scale-matched helpers", ens([f"out/multi_{tgt}_from_{'+'.join(SCALE[tgt])}_veo_seed{s}.npy" for s in range(10)])))
    peh, peo = ens(multi(tgt, "eh")), ens(multi(tgt, "eo"))
    rows.append(("Head eh + 10 diverse mergings", peh)); rows.append(("Head eo + 10 diverse mergings", peo))
    pf = 0.5 * peh + 0.5 * peo; rows.append(("FINAL: average of both heads", pf))
    line(f"\n=== TABLE 2 ({tgt.upper()}): FULLY STRICT  MAE / RMSE / MAPE>=5 (full year) | MAE / RMSE (December) ===")
    for name, p in rows:
        r = sc(p); line(f"{name:40s} {r[0]:.2f} / {r[1]:.2f} / {r[2]:.1f}% | {r[3]:.2f} / {r[4]:.2f}")
    line(f"--- LEVEL / RHYTHM ablation for the FINAL method ({tgt}) ---")
    H = [c for c in TAXI if c != tgt]
    for u in ["jobs", "residents", "taxi commuters", "car-free households", "transit commuters"]:
        lev = U[tgt][u] * np.exp(np.mean([np.log(trips[c] / U[c][u]) for c in H])); TT = T / T.sum() * lev
        r = sc(pf, TT); line(f"LEVEL unit {u:20s}: yearly total error {100*(lev/Y.sum()-1):+.0f}% -> {r[0]:.2f} / {r[1]:.2f} / {r[2]:.1f}%")
    rh = np.mean([st.rhythm(c) for c in S], 0); w = st.dow[tgt][di]; nd = np.bincount(st.dow[tgt], minlength=7)
    T1 = rh[w, si] / nd[w]; T1 = T1 / T1.sum() * T.sum(); r = sc(pf, T1); line(f"RHYTHM without month factor: {r[0]:.2f} / {r[1]:.2f} / {r[2]:.1f}%  (per-slot total error {100*np.abs(T1-tot[di,si]).mean()/tot[di,si].mean():.1f}%)")
    line(f"RHYTHM with month factor (final): per-slot total error {100*np.abs(T-tot[di,si]).mean()/tot[di,si].mean():.1f}%, yearly total error {100*(T.sum()/Y.sum()-1):+.1f}%")
    r = sc(pf, tot[di, si]); line(f"FINAL shares with the TRUE city total (diagnostic, not strict): {r[0]:.2f} / {r[1]:.2f} / {r[2]:.1f}% | Dec {r[3]:.2f} / {r[4]:.2f}")
    np.save(f"out/FINAL_{tgt}_shares.npy", pf)
