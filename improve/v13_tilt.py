"""PROTOCOL_V13.md, Experiment 1: ridge-regularised multiplicative tilt of the shares of the final method v1, fitted and selected on the share helpers of the target only.
For a development target T (Chicago or NYC) the two share helpers are the pseudo-targets of calibrate.py (v1 trained on the OTHER helper only, 5 runs per head), so no taxi
datum of T enters any step before scoring. Usage (from improve/): python v13_tilt.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); B77 = f"{HERE}/../bench77/data"; MCD = f"{HERE}/../multicity/data"; AG = f"{HERE}/data/agg"; OUT = f"{HERE}/out"
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE, boot = ns["totals"], ns["YP"], ns["SHARE"], ns["boot"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})
norm = lambda p: p / p.sum(1, keepdims=True)
LAMS = [0.1, 1.0, 10.0, 100.0]; EPS = 1e-6; CLIP = 2.0; NRUN = 5
PSEUDO = {"chicago": [dict(tag="nyc_for_chicago_p1", srcs=lambda r: f"sf_for_chicago_p{r}"), dict(tag="sf_for_chicago_p1", srcs=lambda r: f"nyc_for_chicago_p{r}")],
          "nyc": [dict(tag="chicago", srcs=lambda r: f"sf_for_chicago_p{r}"), dict(tag="sf_for_nyc_p1", srcs=lambda r: "chicago")]}
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}


def static(tag):
    if tag in ("chicago", "nyc"): return np.load(f"{B77}/region_static_{tag}.npz")
    if tag == "sf": return np.load(f"{MCD}/region_static_sf.npz")
    return np.load(f"{AG}/{tag}_static.npz")


def feats(tag):                                                    # log1p jobs, log1p residents, log busyness (map feature 12), log area
    S = static(tag); E = np.load(f"{HERE}/data/extra_{tag}.npz")
    return np.stack([np.log1p(E["jobs"]), np.log1p(E["res"]), S["X"][:, 11], np.log(S["area_km2"])], 1).astype(np.float64)


def airports(tag): return np.load(f"{HERE}/data/enpl_{tag}.npy") > 0


def load_Y(tag):
    return np.load(YP[tag]).astype(np.float64) if tag in ("chicago", "nyc", "sf", "dc") else np.load(f"{AG}/{tag}_Y.npy").astype(np.float64)


def pseudo(T):
    out = []
    for p in PSEUDO[T]:
        Y = load_Y(p["tag"]); tot = Y.sum(2); idx = [(d, s) for d in range(Y.shape[0]) for s in range(48) if tot[d, s] > 0]
        di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yd = Y[di, si]; Tt = tot[di, si]
        ld = lambda h: np.mean([norm(np.load(f"{OUT}/multi_{p['tag']}_from_{p['srcs'](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, NRUN + 1)], 0)
        sh = 0.5 * ld("eh") + 0.5 * ld("eo"); assert sh.shape == Yd.shape
        out.append(dict(tag=p["tag"], Y=Yd, Tt=Tt, share=sh, Z=feats(p["tag"]), air=airports(p["tag"])))
    return out


def annual(sh, Tt): return (sh * Tt[:, None]).sum(0) / Tt.sum()


def fit(cities, lam):                                              # ridge on log(q / p), weights p, non-airport regions, city-balanced; standardisation over the training cities only
    Zall = np.vstack([c["Z"][~c["air"]] for c in cities]); mu, sd = Zall.mean(0), Zall.std(0) + 1e-9; A, b = [], []
    for c in cities:
        na = ~c["air"]; p = annual(c["share"], c["Tt"]); q = c["Y"].sum(0) / c["Y"].sum(); y = np.log((q + EPS) / (p + EPS))[na]
        Zs = (c["Z"][na] - mu) / sd; w = p[na] / p[na].sum(); X = np.hstack([Zs, np.ones((len(Zs), 1))]); A.append((X * w[:, None]).T @ X); b.append((X * w[:, None]).T @ y)
    A = sum(A) / len(cities); b = sum(b) / len(cities); P = lam * np.eye(A.shape[0]); P[-1, -1] = 0.0
    return np.linalg.solve(A + P, b), mu, sd


def tilt(share, Z, air, beta, mu, sd):                              # tilt non-airport regions, keep the mass of the airport regions
    z = np.clip(((Z - mu) / sd) @ beta[:-1], -CLIP, CLIP); out = share.copy(); na = ~air
    t = share[:, na] * np.exp(z[na])[None]; mass = 1.0 - share[:, air].sum(1, keepdims=True) if air.any() else 1.0
    out[:, na] = t / t.sum(1, keepdims=True) * mass; return out


def mae(share, c): return np.abs(share * c["Tt"][:, None] - c["Y"]).mean()


rng = np.random.default_rng(0); summ = {}; killed = {}
for T in ("chicago", "nyc"):
    P = pseudo(T); print(f"\n===== {T}: share helpers {[c['tag'] for c in P]} =====")
    folds = {lam: [] for lam in LAMS}; signs = {}
    for k in (0, 1):
        tr, va = P[k], P[1 - k]
        for lam in LAMS:
            beta, mu, sd = fit([tr], lam); t = tilt(va["share"], va["Z"], va["air"], beta, mu, sd)
            folds[lam].append(100 * (mae(va["share"], va) - mae(t, va)) / mae(va["share"], va)); signs[(k, lam)] = beta[:3]
    for lam in LAMS: print(f"   lambda {lam:6.1f}: gain of the held-out helper MAE (fit on one helper, validate on the other) {folds[lam][0]:+.2f}% / {folds[lam][1]:+.2f}%; mean {np.mean(folds[lam]):+.2f}%")
    best = max(LAMS, key=lambda l: (round(np.mean(folds[l]), 9), l)); print(f"   selected lambda = {best} (the largest of the ties goes first)")
    for k in (0, 1): print(f"   fold {k} coefficients (jobs, residents, busyness) at lambda {best}: {np.round(signs[(k, best)], 3)}")
    rev = any(np.sign(signs[(0, best)][j]) != np.sign(signs[(1, best)][j]) for j in (0, 1)); worse_both = all(folds[best][k] <= 0 for k in (0, 1))
    killed[T] = rev or worse_both
    print(f"   kill check: sign reversal of jobs or residents between folds: {rev}; held-out-helper MAE not better than v1 in both folds: {worse_both} -> {'STOP' if killed[T] else 'go on'}")
    summ[T] = dict(best=best, P=P)
print("\n===== development targets, strict trips (the city totals of the paper); a target whose kill check triggered is not scored =====")
res = {}
for T in ("chicago", "nyc"):
    if killed[T]: print(f"   {T}: kill check triggered, not scored, v1 stays"); continue
    P = summ[T]["P"]; beta, mu, sd = fit(P, summ[T]["best"])
    Y = np.load(YP[T]).astype(np.float64); nd = Y.shape[0]; di, si, Tt, unit, lev = totals(T, T); y = Y[di, si]
    H = {h: np.mean([norm(np.load(f"{OUT}/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0) for h in ("eh", "eo")}; v1 = 0.5 * H["eh"] + 0.5 * H["eo"]
    new = tilt(v1, feats(T), airports(T), beta, mu, sd); o = {}; d = {}
    for k, S in (("v1", v1), ("tilt", new)):
        e = S * Tt[:, None] - y; a = np.abs(e); o[k] = (a.mean(), np.sqrt((e ** 2).mean())); d[k] = np.bincount(di, weights=a.mean(1), minlength=nd) / np.maximum(np.bincount(di, minlength=nd), 1)
    df = d["v1"] - d["tilt"]; bb = boot(df, rng); gm = 100 * (o["v1"][0] - o["tilt"][0]) / o["v1"][0]; gr = 100 * (o["v1"][1] - o["tilt"][1]) / o["v1"][1]; res[T] = (gm, gr)
    print(f"   {T}: v1 {o['v1'][0]:.3f} / {o['v1'][1]:.3f}; tilt {o['tilt'][0]:.3f} / {o['tilt'][1]:.3f}; daily MAE difference v1 - tilt {df.mean():+.4f} [{np.percentile(bb, 2.5):+.4f}, {np.percentile(bb, 97.5):+.4f}]; MAE change {gm:+.2f}%, RMSE change {gr:+.2f}%")
mean = np.mean([res[t][0] for t in res]); ok = len(res) == 2 and mean >= 3 and all(res[t][0] >= -2 for t in res) and all(res[t][1] >= -5 for t in res)
print(f"\n   development rule: mean MAE change {mean:+.2f}% (need >= +3), no city MAE change below -2%, no city RMSE change below -5% -> {'PASSES' if ok else 'does not pass'}")
