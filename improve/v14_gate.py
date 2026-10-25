"""PROTOCOL_V14.md: helper-gated ridge tilt of the final method v1 (frozen rule: lambda = 1, every leave-one-helper-out fold gains at least 2% MAE, the coefficients of jobs and
residents are positive in every fold fit; otherwise v1). The tilt is the one of v13_tilt.py. Targets: the development targets Chicago and NYC (reproduction of the gate with the
frozen rule) and the five held-out targets (second look; scored once, hourly for Washington, DC and its clusters, Austin on its 248 days, as v10_heldout.py).
Usage (from improve/): python v14_gate.py dev | python v14_gate.py heldout"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); B77 = f"{HERE}/../bench77/data"; MCD = f"{HERE}/../multicity/data"; AG = f"{HERE}/data/agg"; OUT = f"{HERE}/out"; O = f"{HERE}/out_cal"
from v12_rules import norm
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE, boot = ns["totals"], ns["YP"], ns["SHARE"], ns["boot"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})
LAM, MINGAIN, EPS, CLIP, NRUN = 1.0, 2.0, 1e-6, 2.0, 5
S = lambda s: (lambda r: s)
PSEUDO = {"chicago": [("nyc_for_chicago_p1", lambda r: f"sf_for_chicago_p{r}"), ("sf_for_chicago_p1", lambda r: f"nyc_for_chicago_p{r}")],
          "nyc": [("chicago", lambda r: f"sf_for_chicago_p{r}"), ("sf_for_nyc_p1", S("chicago"))],
          "sf": [("chicago", S("nyc")), ("nyc", S("chicago"))],
          "sf_zcta": [("chicago", lambda r: f"nyc_for_sf_zcta_p{r}"), ("nyc_for_sf_zcta_p1", S("chicago"))],
          "dc": [("chicago", lambda r: f"nyc,sf_for_dc_p{r}"), ("nyc", lambda r: f"chicago,sf_for_dc_p{r}"), ("sf_for_dc_p1", S("chicago,nyc"))],
          "dc_cluster": [("chicago", lambda r: f"nyc_for_dc_cluster_p{r},sf_for_dc_cluster_p{r}"), ("nyc_for_dc_cluster_p1", lambda r: f"chicago,sf_for_dc_cluster_p{r}"),
                         ("sf_for_dc_cluster_p1", lambda r: f"chicago,nyc_for_dc_cluster_p{r}")],
          "austin": [("chicago", lambda r: f"nyc,sf_for_austin_p{r}"), ("nyc", lambda r: f"chicago,sf_for_austin_p{r}"), ("sf_for_austin_p1", S("chicago,nyc"))]}
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
TG = {"sf": ("sf", YP["sf"], False, lambda r: "chicago+nyc"), "dc": ("dc", YP["dc"], True, lambda r: f"chicago+nyc+sf_for_dc_p{r}"),
      "sf_zcta": ("sf", f"{MCD}/sf_zcta_30min.npy", False, lambda r: f"chicago+nyc_for_sf_zcta_p{r}"),
      "dc_cluster": ("dc", f"{MCD}/dc_cluster_30min.npy", True, lambda r: f"chicago+nyc_for_dc_cluster_p{r}+sf_for_dc_cluster_p{r}"),
      "austin": ("austin", f"{MCD}/austin_2016_cell_30min.npy", False, lambda r: f"chicago+nyc+sf_for_austin_p{r}")}


def static(tag):
    if tag in ("chicago", "nyc"): return np.load(f"{B77}/region_static_{tag}.npz")
    if os.path.exists(f"{MCD}/region_static_{tag}.npz"): return np.load(f"{MCD}/region_static_{tag}.npz")
    return np.load(f"{AG}/{tag}_static.npz")


def feats(tag):
    S_ = static(tag); E = np.load(f"{HERE}/data/extra_{tag}.npz")
    return np.stack([np.log1p(E["jobs"]), np.log1p(E["res"]), S_["X"][:, 11], np.log(S_["area_km2"])], 1).astype(np.float64)


def airports(tag): return np.load(f"{HERE}/data/enpl_{tag}.npy") > 0


def load_Y(tag): return np.load(YP[tag]).astype(np.float64) if tag in ("chicago", "nyc", "sf", "dc") else np.load(f"{AG}/{tag}_Y.npy").astype(np.float64)


def pseudo(T):
    out = []
    for tag, srcs in PSEUDO[T]:
        Y = load_Y(tag); tot = Y.sum(2); idx = [(d, s) for d in range(Y.shape[0]) for s in range(48) if tot[d, s] > 0]
        di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yd = Y[di, si]; Tt = tot[di, si]
        ld = lambda h: np.mean([norm(np.load(f"{OUT}/multi_{tag}_from_{srcs(r)}_v{h}_seed{r - 1}.npy")) for r in range(1, NRUN + 1)], 0)
        sh = 0.5 * ld("eh") + 0.5 * ld("eo"); assert sh.shape == Yd.shape, (tag, sh.shape, Yd.shape)
        out.append(dict(tag=tag, Y=Yd, Tt=Tt, share=sh, Z=feats(tag), air=airports(tag)))
    return out


def annual(sh, Tt): return (sh * Tt[:, None]).sum(0) / Tt.sum()


def fit(cities, lam):
    Zall = np.vstack([c["Z"][~c["air"]] for c in cities]); mu, sd = Zall.mean(0), Zall.std(0) + 1e-9; A, b = [], []
    for c in cities:
        na = ~c["air"]; p = annual(c["share"], c["Tt"]); q = c["Y"].sum(0) / c["Y"].sum(); y = np.log((q + EPS) / (p + EPS))[na]
        Zs = (c["Z"][na] - mu) / sd; w = p[na] / p[na].sum(); X = np.hstack([Zs, np.ones((len(Zs), 1))]); A.append((X * w[:, None]).T @ X); b.append((X * w[:, None]).T @ y)
    A = sum(A) / len(cities); b = sum(b) / len(cities); P = lam * np.eye(A.shape[0]); P[-1, -1] = 0.0
    return np.linalg.solve(A + P, b), mu, sd


def tilt(share, Z, air, beta, mu, sd):
    z = np.clip(((Z - mu) / sd) @ beta[:-1], -CLIP, CLIP); out = share.copy(); na = ~air
    t = share[:, na] * np.exp(z[na])[None]; mass = 1.0 - share[:, air].sum(1, keepdims=True) if air.any() else 1.0
    out[:, na] = t / t.sum(1, keepdims=True) * mass; return out


def mae(share, c): return np.abs(share * c["Tt"][:, None] - c["Y"]).mean()


def gate(T):
    P = pseudo(T); rows = []; ok = True
    print(f"\n===== gate for {T}: share helpers {[c['tag'] for c in P]} (lambda {LAM}, every fold gain >= {MINGAIN}%, jobs and residents coefficients > 0 in every fold) =====")
    for j, va in enumerate(P):
        tr = [c for k, c in enumerate(P) if k != j]; beta, mu, sd = fit(tr, LAM); t = tilt(va["share"], va["Z"], va["air"], beta, mu, sd)
        g = 100 * (mae(va["share"], va) - mae(t, va)) / mae(va["share"], va); pos = beta[0] > 0 and beta[1] > 0; ok &= (g >= MINGAIN and pos)
        print(f"   fold {j}: validate on {va['tag']:24s} gain {g:+7.2f}%  coefficients jobs {beta[0]:+.3f} residents {beta[1]:+.3f} busyness {beta[2]:+.3f} area {beta[3]:+.3f}  -> {'pass' if (g >= MINGAIN and pos) else 'fail'}")
    print(f"   GATE: {'PASSES, the tilt is applied' if ok else 'FAILS, v1 stays'}"); return P, ok


def v1_of(T, r_src, tag_runs, n):
    base = O if T not in ("chicago", "nyc") else OUT
    return {h: np.mean([norm(np.load(f"{base}/multi_{T}_from_{r_src(r)}_v{h}_seed{r - 1}.npy")) for r in range(1, n + 1)], 0) for h in ("eh", "eo")}


rng = np.random.default_rng(0); mode = sys.argv[1]
if mode == "dev":
    res = {}
    for T in ("chicago", "nyc"):
        P, ok = gate(T)
        Y = np.load(YP[T]).astype(np.float64); nd = Y.shape[0]; di, si, Tt, unit, lev = totals(T, T); y = Y[di, si]
        H = v1_of(T, SRC[T], None, 10); v1 = 0.5 * H["eh"] + 0.5 * H["eo"]; beta, mu, sd = fit(P, LAM)
        new = tilt(v1, feats(T), airports(T), beta, mu, sd) if ok else v1; o = {}
        for k, S_ in (("v1", v1), ("gated", new)):
            e = S_ * Tt[:, None] - y; o[k] = (np.abs(e).mean(), np.sqrt((e ** 2).mean()))
        print(f"   {T}: v1 {o['v1'][0]:.3f} / {o['v1'][1]:.3f}; gated tilt {o['gated'][0]:.3f} / {o['gated'][1]:.3f}")
else:
    summary = {}
    for tag, (city, yfile, hourly, mat) in TG.items():
        P, ok = gate(tag)
        di_ = None
        if tag == "austin":
            asrc = open(f"{HERE}/austin_eval.py").read(); ans = {"__file__": f"{HERE}/austin_eval.py"}
            exec(asrc[:asrc.index("Y = np.load(f\"{MC}/austin_2016_cell_30min.npy\")")], ans)
            di, si, Tt, ndays = ans["di"], ans["si"], ans["Tt"], ans["ndays"]
        else:
            di, si, Tt, unit, lev = totals(city, tag); ndays = None
        Y = np.load(yfile).astype(np.float64); nd = Y.shape[0]
        def err(p):
            F = np.full(Y.shape, np.nan); F[di, si] = norm(p) * Tt[:, None]; E = F - Y
            if hourly: E = E.reshape(nd, 24, 2, -1).sum(2)
            E = E.reshape(nd, -1); okk = ~np.isnan(E); d_ = np.nanmean(np.abs(E), 1)
            return np.abs(E[okk]).mean(), np.sqrt((E[okk] ** 2).mean()), (d_[:ndays] if ndays else d_)
        runs = {h: [f"{O}/multi_{tag}_from_{mat(r)}_v{h}_seed{r - 1}.npy" for r in range(1, 11)] for h in ("eh", "eo")}
        v1 = 0.5 * norm(np.mean([np.load(f) for f in runs["eh"]], 0)) + 0.5 * norm(np.mean([np.load(f) for f in runs["eo"]], 0))
        beta, mu, sd = fit(P, LAM); tl = tilt(v1, feats(tag), airports(tag), beta, mu, sd); gated = tl if ok else v1
        res = {"final method v1": err(v1), "tilt without the gate": err(tl), "gated tilt (the rule)": err(gated)}
        print(f"   {tag}: mass moved by the tilt (mean absolute share change summed over regions per slot) {np.abs(tl - v1).sum(1).mean():.4f}; fraction of regions with a higher share {np.mean((tl - v1).mean(0) > 0):.2f}")
        for k, (m, r_, _) in res.items(): print(f"   {k:26s} MAE {m:.3f}  RMSE {r_:.3f}")
        diff = res["final method v1"][2] - res["gated tilt (the rule)"][2]; b = boot(diff, rng)
        gm = 100 * (res["final method v1"][0] - res["gated tilt (the rule)"][0]) / res["final method v1"][0]; gr = 100 * (res["final method v1"][1] - res["gated tilt (the rule)"][1]) / res["final method v1"][1]
        print(f"   v1 - gated: daily MAE difference {diff.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]; MAE change {gm:+.2f}%, RMSE change {gr:+.2f}% (positive = gated better)")
        summary[tag] = (gm, gr, ok)
    print("\n===== summary (positive = gated tilt better than v1; a blocked target counts as no change) =====")
    for t, (gm, gr, ok) in summary.items(): print(f"   {t:11s} gate {'passes' if ok else 'fails':6s} MAE change {gm:+6.2f}%  RMSE change {gr:+6.2f}%")
    m = np.mean([v[0] for v in summary.values()]); r = np.mean([v[1] for v in summary.values()]); cm = sum(v[0] > 0 for v in summary.values()); cr = sum(v[1] > 0 for v in summary.values())
    adopt = len(summary) == 5 and m > 0 and r > 0 and cm >= 3 and cr >= 3
    print(f"   mean MAE change {m:+.2f}%, mean RMSE change {r:+.2f}%, better in MAE in {cm} of 5 and in RMSE in {cr} of 5 -> {'ADOPTED' if adopt else 'NOT adopted, v1 stays'}")
