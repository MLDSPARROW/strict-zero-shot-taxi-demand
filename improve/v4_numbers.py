"""Paper v4 numbers for all four targets (2026-10-03). FULLY STRICT. DC scored hourly (primary) per PROTOCOL_HELDOUT.md.
(1) level-unit selection table (helper-only LOO) for every target; (2) final MAE/RMSE with 95% CIs (7-day block bootstrap);
(3) day-level comparison of FINAL vs base allocator and vs the best simple baseline: mean daily difference, 95% CI,
    share of days better, Cohen's d_z, n days; (4) allocation-only metrics; (5) partition-count sensitivity for DC."""
import numpy as np, importlib.util, sys
from scipy.stats import spearmanr
spec = importlib.util.spec_from_file_location("sa", "score_all.py"); sa = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(sa)
rng = np.random.default_rng(0)
print("(1) LEVEL unit selection: mean |error| of held-out helper (helper-only LOO)")
for T in ["chicago", "nyc", "sf", "dc"]:
    H = [c for c in sa.YP if c != T]
    loo = {u: np.mean([abs(np.exp(np.mean([np.log(sa.N[c] / sa.U[c][u]) for c in H if c != h])) * sa.U[h][u] / sa.N[h] - 1) for h in H]) for u in sa.U[T]}
    print(f"   {T:8s} " + " | ".join(f"{u} {100*v:.1f}%" for u, v in loo.items()))
for T in ["chicago", "nyc", "sf", "dc"]:
    Y = sa.Y[T]; tot = Y.sum(2); idx = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Y[di, si]; u, lev, Tt = sa.strict_total(T, di, si); R = Yc.shape[1]; n = len(di)
    H = sa.MODEL_H[T]
    pf = 0.5 * sa.ens([f"out/multi_{T}_from_{H(r)}_veh_seed{r-1}.npy" for r in range(1, 11)]) + 0.5 * sa.ens([f"out/multi_{T}_from_{H(r)}_veo_seed{r-1}.npy" for r in range(1, 11)])
    pb = sa.ens([f"../multicity/multi_{T}_from_{'+'.join(sa.SHARE[T])}_seed{s}.npy" for s in range(10)])
    E_ = np.load(f"data/extra_{T}.npz"); pj = np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0); jr = E_["jobs"] + E_["res"]; pjr = np.repeat((jr / jr.sum())[None], n, 0)
    hourly = T == "dc"
    def daily(p):
        P = np.zeros_like(Y); P[di, si] = p * Tt[:, None]; E = P - Y
        if hourly: E = E.reshape(E.shape[0], 24, 2, -1).sum(2)
        E = E.reshape(E.shape[0], -1); return np.abs(E).mean(1), (E ** 2).mean(1)
    fa, fs = daily(pf); nb = len(fa) // 7
    bm = [fa[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(fa) - 7, nb)])].mean() for _ in range(2000)]
    br = [np.sqrt(fs[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(fs) - 7, nb)])].mean()) for _ in range(2000)]
    print(f"\n===== {T.upper()} ({'hourly' if hourly else '30-min'}) =====\n(2) FINAL MAE {fa.mean():.2f} [95% CI {np.percentile(bm,2.5):.2f}, {np.percentile(bm,97.5):.2f}] | RMSE {np.sqrt(fs.mean()):.2f} [{np.percentile(br,2.5):.2f}, {np.percentile(br,97.5):.2f}] (n = {len(fa)} days)")
    for nm, p in [("base joint allocator", pb), ("jobs-proportional", pj), ("jobs + residents", pjr)]:
        a, s = daily(p); out = []
        for x, y, mn in [(fa, a, "MAE"), (fs, s, "MSE")]:
            d = x - y; bs = [d[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(d) - 7, nb)])].mean() for _ in range(2000)]
            out.append(f"{mn}: mean daily diff {d.mean():+.3f} [95% CI {np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}], final better on {100*(d<0).mean():.0f}% of days, d_z {d.mean()/d.std(ddof=1):+.2f}")
        print(f"(3) FINAL vs {nm:20s} " + " | ".join(out))
    Sh = Yc / tot[di, si][:, None]
    def alloc(p):
        q = np.clip(p, 1e-12, 1); s_ = np.clip(Sh, 1e-12, 1); mM = 0.5 * (q + s_)
        js = 0.5 * (np.sum(q * np.log(q / mM), 1) + np.sum(np.where(Sh > 0, s_ * np.log(s_ / mM), 0), 1))
        pt, tt = (p * tot[di, si][:, None]).sum(0), Yc.sum(0); top = set(np.argsort(-tt)[:10])
        return f"{np.abs(p - Sh).mean():.5f} | {js.mean():.4f} | {spearmanr(pt, tt)[0]:.3f} | {len(top & set(np.argsort(-pt)[:10]))/10:.1f}"
    print("(4) allocation-only (share MAE | mean JS divergence | Spearman annual totals | top-10 recall)")
    for nm, p in [("jobs-proportional", pj), ("jobs + residents", pjr), ("base joint allocator", pb), ("final", pf)]: print(f"    {nm:22s} {alloc(p)}")
    if T == "dc":
        print("(5) DC partitions k (hourly MAE / RMSE)")
        for k in [1, 2, 5, 10]:
            p = 0.5 * sa.ens([f"out/multi_dc_from_{H(r)}_veh_seed{r-1}.npy" for r in range(1, k + 1)]) + 0.5 * sa.ens([f"out/multi_dc_from_{H(r)}_veo_seed{r-1}.npy" for r in range(1, k + 1)])
            a, s = daily(p); print(f"    k={k:2d}: {a.mean():.2f} / {np.sqrt(s.mean()):.2f}")
