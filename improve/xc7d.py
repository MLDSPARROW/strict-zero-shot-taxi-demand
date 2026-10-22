"""PROTOCOL_V8.md: a scale-matched cross-city forecaster that reads the previous seven days of the target ("ours + 7 days").
Trained on helper cities only (re-aggregated to the scale of the target, partition r), then applied to the target, whose own previous 336
half-hour counts are an INPUT; no target data are used for training. Poisson gradient boosting on top of a shrunken weekly-average offset.
Usage (from improve/), the interpreter of the main environment:
  python xc7d.py leak              # tests that no feature of an origin depends on counts at or after the origin (synthetic and real arrays)
  python xc7d.py check             # reproduces the weekly-average values of PROTOCOL_V7 (October and November 2021, a = 0): 1.400 / 1.955
  python xc7d.py time              # trains one model on helper data only (no target data) and reports the time
  python xc7d.py dev               # development period, candidates C1 to C4, 2 partitions, scored against Chronos-Bolt-small
  python xc7d.py final <cand>      # December 2021, the frozen candidate, 10 partitions, scale-matched and native helpers, once"""
import os, sys, time, numpy as np
from datetime import date, timedelta
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try:                                                     # below-normal priority: the laptop is shared with other runs
    import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception: pass
import xgboost as xgb
from pandas.tseries.holiday import USFederalHolidayCalendar
B = f"{HERE}/../bench77/data"; MC = f"{HERE}/../multicity/data"; AG = f"{HERE}/data/agg"; OUT = f"{HERE}/out_xc"; os.makedirs(OUT, exist_ok=True)
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy", "sf": f"{MC}/sf_2023_tract_30min.npy"}
SP = {"chicago": f"{B}/region_static_chicago.npz", "nyc": f"{B}/region_static_nyc.npz", "sf": f"{MC}/region_static_sf.npz"}
CTX, HOR, KAPPA, LAM_FLOOR = 336, 6, 1.0, 0.002
NT = int(os.environ.get("XC_THREADS", "2"))
HEL = {"chicago": {"matched": lambda r: [f"nyc_for_chicago_p{r}", f"sf_for_chicago_p{r}"], "native": lambda r: ["nyc", "sf"]},
       "nyc": {"matched": lambda r: ["chicago", f"sf_for_nyc_p{r}"], "native": lambda r: ["chicago", "sf"]}}
CAND = {"C1": dict(depth=6, static=True), "C2": dict(depth=6, static=False), "C3": dict(depth=4, static=True), "C4": dict(depth=8, static=True)}
DEV = (273, 333); DEC = (334, 364)                       # days of 2021 (0 = 1 January): October and November, December


def grid(d0, d1, half):
    """forecast origins (flat slot index): every 6 slots, or every 12 slots with offsets 0,12,24,36 on even days and 6,18,30,42 on odd days"""
    out = []
    for d in range(d0, d1 + 1):
        offs = ((0, 12, 24, 36) if d % 2 == 0 else (6, 18, 30, 42)) if half else range(0, 48, 6)
        out += [d * 48 + s for s in offs]
    return np.array(out)


def load_series(name):
    """name: a native city or a scale-matched tag of data/agg. Returns counts (nd*48, B) float32, 18 map features, start date."""
    if name in YP: Y, S = np.load(YP[name]), np.load(SP[name])
    else: Y, S = np.load(f"{AG}/{name}_Y.npy"), np.load(f"{AG}/{name}_static.npz")
    nd = Y.shape[0]
    return Y.reshape(nd * 48, -1).astype(np.float32), S["X"][:, :18].astype(np.float64), (date(2023, 1, 1) if name.startswith("sf") else date(2021, 1, 1))


def prep(Y, st):
    """everything that depends on the counts only through slots at least 48 slots (one day) before the slot concerned"""
    T, Bn = Y.shape; nd = T // 48
    days = [st + timedelta(days=d) for d in range(nd)]
    hol = set(USFederalHolidayCalendar().holidays(start=str(st - timedelta(days=2)), end=str(st + timedelta(days=nd + 2))).date)
    wk = np.array([x.weekday() for x in days]); weekend = wk >= 5
    h = np.array([x in hol for x in days]); hp = np.array([(x - timedelta(days=1)) in hol for x in days]); hn = np.array([(x + timedelta(days=1)) in hol for x in days])
    Y3 = Y.reshape(nd, 48, Bn); LW = np.zeros((T, Bn), np.float32); NS = np.zeros(nd, np.int16)
    for d in range(7, nd):                               # mean of the same slot on the days of the same type (weekday / weekend) among the last 7 days
        ids = [d - i for i in range(1, 8) if weekend[d - i] == weekend[d]]; LW[d * 48:(d + 1) * 48] = Y3[ids].mean(0); NS[d] = len(ids)
    z = np.zeros((1, Bn)); C = np.concatenate([z, np.cumsum(Y, 0, dtype=np.float64)]); CL = np.concatenate([z, np.cumsum(LW, 0, dtype=np.float64)])
    LWc = LW.sum(1, dtype=np.float64); Cc = np.concatenate([[0.], np.cumsum(Y.sum(1, dtype=np.float64))]); CLc = np.concatenate([[0.], np.cumsum(LWc)])
    return dict(Y=Y, LW=LW, NS=NS, C=C, CL=CL, LWc=LWc, Cc=Cc, CLc=CLc, wk=wk, weekend=weekend, h=h, hp=hp, hn=hn)


def features(P, Xs, O, static=True):
    """rows of all (origin, region) for the horizon steps j = 0..5 (j-major). Returns F (n, nf) float32, offset margin (n,), label (n,), weekly average (n,)"""
    Y, LW, C, CL, Cc, CLc, LWc, NS = (P[k] for k in ("Y", "LW", "C", "CL", "Cc", "CLc", "LWc", "NS"))
    O = np.asarray(O); no, Bn = len(O), Y.shape[1]; bc = lambda a: np.broadcast_to(np.asarray(a, np.float64).reshape(no, -1), (no, Bn))
    wm = (C[O] - C[O - CTX]) / CTX
    r6, r12, r48 = (C[O] - C[O - k] for k in (6, 12, 48)); e6, e48 = CL[O] - CL[O - 6], CL[O] - CL[O - 48]; y1 = Y[O - 1]
    rr6, rr48 = (r6 + .5) / (e6 + .5), (r48 + .5) / (e48 + .5)
    rho6 = np.clip((Cc[O] - Cc[O - 6] + 1) / (CLc[O] - CLc[O - 6] + 1), .5, 2.); rho48 = np.clip((Cc[O] - Cc[O - 48] + 1) / (CLc[O] - CLc[O - 48] + 1), .5, 2.)
    cwm = np.maximum((Cc[O] - Cc[O - CTX]) / CTX, 1e-6); Fs, Ms, Ys, Ws = [], [], [], []
    for j in range(HOR):
        t = O + j; lw = LW[t].astype(np.float64); lw3 = (LW[t - 1].astype(np.float64) + lw + LW[t + 1]) / 3.
        ly, ly1 = Y[t - CTX], Y[t - 48]; ns = NS[t // 48].astype(np.float64); cprof = LWc[t] / cwm; prior = wm * cprof[:, None]
        lam0 = np.maximum(rho6[:, None] * (ns[:, None] * lw + KAPPA * prior) / (ns[:, None] + KAPPA), LAM_FLOOR)
        d, s = t // 48, t % 48
        cal = [np.sin(2 * np.pi * s / 48), np.cos(2 * np.pi * s / 48), np.sin(2 * np.pi * P["wk"][d] / 7), np.cos(2 * np.pi * P["wk"][d] / 7),
               P["weekend"][d], P["h"][d], P["hp"][d], P["hn"][d]]
        cols = [np.log(lam0), np.log1p(lw), np.log1p(lw3), np.log1p(ly), np.log1p(ly1), np.log1p(wm), np.log1p(prior), bc(ns), bc(rho6), bc(rho48),
                np.log(rr6), np.log(rr48), np.log1p(y1), np.log1p(r6), np.log1p(r12), bc(cprof), np.full((no, Bn), float(j))] + [bc(c) for c in cal]
        F = np.stack([np.broadcast_to(c, (no, Bn)) for c in cols], -1)
        if static: F = np.concatenate([F, np.broadcast_to(Xs[None], (no, Bn, Xs.shape[1]))], -1)
        Fs.append(F.reshape(no * Bn, -1).astype(np.float32)); Ms.append(np.log(lam0).reshape(-1)); Ys.append(Y[t].reshape(-1)); Ws.append((rho6[:, None] * lw).reshape(-1))
    return np.concatenate(Fs), np.concatenate(Ms), np.concatenate(Ys), np.concatenate(Ws)


def scaler(pool):
    med = np.median(pool, 0); q25, q75 = np.percentile(pool, [25, 75], 0); scl = np.where(q75 - q25 < 1e-5, 1., q75 - q25)
    return lambda X: np.clip((X - med) / scl, -8, 8)


def train_set(names, cutoff, static):
    """helper data only. 2021 helpers are cut at day `cutoff` (the first day of the evaluation period); the 2023 San Francisco series is used whole"""
    hs = []
    for nm in names:
        Y, X, st = load_series(nm); last = Y.shape[0] // 48 if nm.startswith("sf") else cutoff; hs.append((prep(Y[:last * 48], st), X, last))
    sc = scaler(np.concatenate([h[1] for h in hs])); Ftr, Mtr, ytr, Fva, Mva, yva = [], [], [], [], [], []
    for P, X, last in hs:
        for va in (False, True):
            O = np.array([o for o in grid(8, last - 1, True) if o + HOR + 1 <= last * 48 and ((o // 48) % 12 == 5) == va])
            F, M, y, _ = features(P, sc(X), O, static); (Fva if va else Ftr).append(F); (Mva if va else Mtr).append(M); (yva if va else ytr).append(y)
    return [np.concatenate(a) for a in (Ftr, Mtr, ytr, Fva, Mva, yva)], sc


def fit(Ftr, Mtr, ytr, Fva, Mva, yva, depth, seed):
    prm = dict(objective="count:poisson", eval_metric="poisson-nloglik", eta=0.05, max_depth=depth, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
               reg_lambda=1.0, tree_method="hist", max_bin=128, nthread=NT, seed=seed)
    dtr, dva = xgb.DMatrix(Ftr, label=ytr, base_margin=Mtr), xgb.DMatrix(Fva, label=yva, base_margin=Mva)
    return xgb.train(prm, dtr, num_boost_round=600, evals=[(dva, "val")], early_stopping_rounds=30, verbose_eval=False)


def predict(bst, F, M): return bst.predict(xgb.DMatrix(F, base_margin=M), iteration_range=(0, bst.best_iteration + 1))


def target_rows(T, O, sc, static):
    Y, X, st = load_series(T); P = prep(Y, st); F, M, y, W = features(P, sc(X), O, static); return Y, P, F, M, y, W


def run_one(T, cond, r, cand, mode, cutoff, O):
    f = f"{OUT}/pred_{mode}_{cand}_{T}_{cond}_r{r}.npy"
    if os.path.exists(f): return np.load(f)
    t0 = time.time(); cfg = CAND[cand]; (Ftr, Mtr, ytr, Fva, Mva, yva), sc = train_set(HEL[T][cond](r), cutoff, cfg["static"])
    bst = fit(Ftr, Mtr, ytr, Fva, Mva, yva, cfg["depth"], r - 1); Y, P, F, M, y, W = target_rows(T, O, sc, cfg["static"])
    R = Y.shape[1]; p = predict(bst, F, M).astype(np.float32).reshape(HOR, len(O), R).transpose(1, 0, 2)
    np.save(f, p); print(f"   {mode} {cand} {T} {cond} r={r}: {len(ytr)} train rows, {bst.best_iteration + 1} trees, {time.time() - t0:.0f}s", flush=True); return p


def chronos(T, mode):
    """quantile-mean and median forecasts of Chronos-Bolt-small as arrays (nd*48, R) and the mask of covered slots"""
    f = f"{OUT}/fm_{mode}_{T}.npz"
    if os.path.exists(f): z = np.load(f); return z["PM"], z["PD"], z["cov"]
    assert mode == "dec"                                   # December: rebuilt from the chunks saved by fm_chronos.py (PROTOCOL_V6)
    Y = np.load(YP[T]).astype(np.float32); nd, _, R = Y.shape; orig = grid(DEC[0], DEC[1], False)
    K = np.repeat(orig, R); Rr = np.tile(np.arange(R), len(orig)); n = len(K)
    Q = np.concatenate([np.load(f"{HERE}/out_cal/fm_chunks/{T}_{c0}.npy") for c0 in range(0, n, 2048)])
    qm = np.clip(np.nan_to_num(Q).mean(1), 0, None); md = np.clip(np.nan_to_num(Q)[:, 4, :], 0, None)
    PM = np.zeros((nd * 48, R), np.float32); PD = np.zeros_like(PM); cov = np.zeros(nd * 48, bool)
    for j in range(HOR): PM[K + j, Rr] = qm[:, j]; PD[K + j, Rr] = md[:, j]; cov[K + j] = True
    np.savez(f, PM=PM, PD=PD, cov=cov); return PM, PD, cov


def poisson_median(lam):
    from scipy.stats import poisson
    return poisson.ppf(0.5, np.maximum(lam, 1e-9))


def boot(x, rng, nb=2000):
    k = len(x) // 7; return np.array([x[np.concatenate([np.arange(a, a + 7) for a in rng.integers(0, len(x) - 7, k)])].mean() for _ in range(nb)])


def perday(T, O, pred, truth_flat):
    """MAE per day over all region-slots of the forecasts `pred` (n_o, HOR, R); also overall MAE, RMSE"""
    t = (O[:, None] + np.arange(HOR)[None]); y = truth_flat[t]; e = np.abs(pred - y); d = t // 48
    num = np.bincount(d.reshape(-1), e.mean(2).reshape(-1)); den = np.bincount(d.reshape(-1)); ok = den > 0
    return (num[ok] / den[ok]), float(e.mean()), float(np.sqrt(((pred - y) ** 2).mean()))


def cmp_chronos(T, O, pred, mode):
    Y3 = np.load(YP[T]); flat = Y3.reshape(-1, Y3.shape[2]).astype(np.float32); PM, PD, cov = chronos(T, mode)
    t = O[:, None] + np.arange(HOR)[None]; assert cov[t].all()
    cm = np.stack([PM[t[:, j]] for j in range(HOR)], 1); cd = np.stack([PD[t[:, j]] for j in range(HOR)], 1)
    dn, mn, rn = perday(T, O, pred, flat); dc, mc, rc = perday(T, O, cm, flat); _, mdn, _ = perday(T, O, cd, flat)
    diff = dn - dc; b = boot(diff, np.random.default_rng(0))
    return dict(ours=(mn, rn), chronos=(mc, rc), chronos_median=mdn, diff=(diff.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)), better_days=float((diff < 0).mean()),
                median_ours=perday(T, O, poisson_median(pred), flat)[1])


def baseline_pred(T, O):
    Y, X, st = load_series(T); P = prep(Y, st); _, _, y, W = features(P, np.zeros((Y.shape[1], 1)), O, False)
    return W.reshape(HOR, len(O), Y.shape[1]).transpose(1, 0, 2), Y.reshape(-1, Y.shape[1])


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "leak":                                      # no feature of an origin may depend on counts at or after the origin
        rng = np.random.default_rng(1); st = date(2021, 1, 1)
        for label, Y in (("synthetic", rng.poisson(2.0, (60 * 48, 7)).astype(np.float32)), ("real (Chicago 2021, first 120 days)", load_series("chicago")[0][:120 * 48])):
            P = prep(Y, st); O = np.array([20 * 48 + 12, 30 * 48 + 6, 45 * 48 + 24]); X = np.random.default_rng(2).normal(size=(Y.shape[1], 18))
            F0, M0, y0, W0 = features(P, X, O); worst = 0.0
            for k, o in enumerate(O):
                Y2 = Y.copy(); Y2[o:] = rng.poisson(50, Y2[o:].shape); P2 = prep(Y2, st); F1, M1, y1, W1 = features(P2, X, O[k:k + 1])
                rows = np.concatenate([np.arange(j * len(O) * Y.shape[1] + k * Y.shape[1], j * len(O) * Y.shape[1] + (k + 1) * Y.shape[1]) for j in range(HOR)])
                worst = max(worst, float(np.abs(F0[rows] - F1).max()), float(np.abs(M0[rows] - M1).max()), float(np.abs(W0[rows] - W1).max()))
            print(f"leak test, {label}: largest change of any feature, offset or weekly average when all counts from the origin on are replaced: {worst:.2e}")
    elif mode == "check":
        for T, want in (("chicago", 1.400), ("nyc", 1.955)):
            O = grid(DEV[0], DEV[1], False); pred, flat = baseline_pred(T, O); d, m, r = perday(T, O, pred, flat)
            print(f"{T}: weekly average (a = 0 of PROTOCOL_V7) October and November, every 6 slots: MAE {m:.3f} RMSE {r:.3f} (PROTOCOL_V7: MAE {want})")
    elif mode == "time":
        t0 = time.time(); (Ftr, Mtr, ytr, Fva, Mva, yva), sc = train_set(HEL["chicago"]["matched"](1), DEV[0], True); print(f"train set {Ftr.shape}, validation {Fva.shape}, {time.time() - t0:.0f}s", flush=True)
        t0 = time.time(); bst = fit(Ftr, Mtr, ytr, Fva, Mva, yva, 6, 0); print(f"fit: {bst.best_iteration + 1} trees, {time.time() - t0:.0f}s")
        p = predict(bst, Fva, Mva); raw = bst.predict(xgb.DMatrix(Fva, base_margin=Mva), output_margin=True, iteration_range=(0, bst.best_iteration + 1))
        print("prediction equals exp(margin):", float(np.abs(p - np.exp(raw)).max()), "| helper validation mean label %.3f, mean prediction %.3f, offset-only mean %.3f" % (yva.mean(), p.mean(), np.exp(Mva).mean()))
        print("helper validation MAE: offset only %.4f, model %.4f" % (np.abs(np.exp(Mva) - yva).mean(), np.abs(p - yva).mean()))
    elif mode == "dev":
        O = grid(DEV[0], DEV[1], True); res = {}
        for T in ("chicago", "nyc"):
            for c in CAND:
                for r in (1, 2): run_one(T, "matched", r, c, "dev", DEV[0], O)
            for r in (1, 2): run_one(T, "native", r, "C1", "dev", DEV[0], O)
        print("\n===== development period, 1 October to 30 November 2021, origins every 12 slots (alternating offsets), MAE / RMSE of the mean forecast =====")
        for T in ("chicago", "nyc"):
            b, flat = baseline_pred(T, O); db, mb, rb = perday(T, O, b, flat); res[T] = {}
            for c in CAND: res[T][c] = cmp_chronos(T, O, np.mean([np.load(f"{OUT}/pred_dev_{c}_{T}_matched_r{r}.npy") for r in (1, 2)], 0), "dev")
            nat = cmp_chronos(T, O, np.mean([np.load(f"{OUT}/pred_dev_C1_{T}_native_r{r}.npy") for r in (1, 2)], 0), "dev")
            ch = res[T]["C1"]
            print(f"\n{T}: weekly average (a=0) {mb:.3f} / {rb:.3f} | Chronos-Bolt-small {ch['chronos'][0]:.3f} / {ch['chronos'][1]:.3f} (median-forecast MAE {ch['chronos_median']:.3f})")
            for c in CAND:
                q = res[T][c]; print(f"   {c} (depth {CAND[c]['depth']}, map features {CAND[c]['static']}): {q['ours'][0]:.3f} / {q['ours'][1]:.3f}  ratio to Chronos MAE {q['ours'][0] / q['chronos'][0]:.3f}  "
                                    f"daily MAE difference to Chronos {q['diff'][0]:+.4f} [{q['diff'][1]:+.4f}, {q['diff'][2]:+.4f}], better on {100 * q['better_days']:.0f}% of days | Poisson-median MAE {q['median_ours']:.3f}")
            print(f"   C1 with native helpers: {nat['ours'][0]:.3f} / {nat['ours'][1]:.3f} (for information, not used for the choice)")
        crit = {c: np.mean([res[T][c]["ours"][0] / res[T][c]["chronos"][0] for T in res]) for c in CAND}; best = min(CAND, key=lambda c: (crit[c], c))
        print("\ncriterion (mean over the two cities of MAE / MAE of Chronos-Bolt):", " ".join(f"{c}: {crit[c]:.3f}" for c in CAND), f"-> {best}")
        ok = all(res[T][best]["ours"][0] < res[T][best]["chronos"][0] for T in res)
        print(f"selected candidate {best}; lower MAE than Chronos-Bolt in both cities on the development period: {ok} -> " + ("December is run once with it" if ok else "December is NOT run under this protocol"))
        print("done")
    elif mode == "final":
        cand = sys.argv[2]; O = grid(DEC[0], DEC[1], False); out = {}
        for T in ("chicago", "nyc"):
            for cond in ("matched", "native"):
                for r in range(1, 11): run_one(T, cond, r, cand, "dec", DEC[0], O)
        print(f"\n===== December 2021, candidate {cand}, 10 runs per condition, every 6 slots (each slot once), MAE / RMSE of the mean forecast =====")
        for T in ("chicago", "nyc"):
            b, flat = baseline_pred(T, O); _, mb, rb = perday(T, O, b, flat); q, qn = (cmp_chronos(T, O, np.mean([np.load(f"{OUT}/pred_dec_{cand}_{T}_{c}_r{r}.npy") for r in range(1, 11)], 0), "dec") for c in ("matched", "native"))
            dm = perday(T, O, np.mean([np.load(f"{OUT}/pred_dec_{cand}_{T}_matched_r{r}.npy") for r in range(1, 11)], 0), flat)[0]; dn = perday(T, O, np.mean([np.load(f"{OUT}/pred_dec_{cand}_{T}_native_r{r}.npy") for r in range(1, 11)], 0), flat)[0]
            bb = boot(dn - dm, np.random.default_rng(1)); mt = np.mean([perday(T, O, np.load(f"{OUT}/pred_dec_{cand}_{T}_matched_r{r}.npy"), flat)[1] for r in range(1, 11)])
            print(f"\n{T}: weekly average (a=0) {mb:.3f} / {rb:.3f} | Chronos-Bolt-small {q['chronos'][0]:.3f} / {q['chronos'][1]:.3f} (median-forecast MAE {q['chronos_median']:.3f})")
            print(f"   ours + 7 days, scale-matched helpers: {q['ours'][0]:.3f} / {q['ours'][1]:.3f} | daily MAE difference to Chronos {q['diff'][0]:+.4f} [{q['diff'][1]:+.4f}, {q['diff'][2]:+.4f}], ours better on {100 * q['better_days']:.0f}% of days | Poisson-median MAE {q['median_ours']:.3f}")
            print(f"   ours + 7 days, native helpers:       {qn['ours'][0]:.3f} / {qn['ours'][1]:.3f} | scale-matched minus native, daily MAE: {-(dn - dm).mean():+.4f} [{-np.percentile(bb, 97.5):+.4f}, {-np.percentile(bb, 2.5):+.4f}]")
            print(f"   mean MAE of the ten single runs (matched): {mt:.3f}")
        print("done")
    else: print(__doc__)
