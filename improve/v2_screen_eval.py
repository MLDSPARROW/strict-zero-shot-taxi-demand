"""PROTOCOL_V2.md screen evaluation (development targets only). For each variant, the ensemble of runs r = 1..n (helper
partition r, seed r - 1) is scored on the target's observed slots: MAE_share (the pre-declared screen metric), allocation-only
trip MAE / RMSE (true slot totals, as printed by run_x.py), peak ratio and top-10 recall of the yearly shares.
Rule: a variant passes if its MAE_share is >= 3% better than its baseline in one development target and not > 3% worse in
the other. Usage (from improve/): python v2_screen_eval.py [n_runs, default 3]"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); B = f"{HERE}/../bench77/data"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
YF = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy"}
BASE = {"ep": "eo", "epa": "eo", "eha": "eh", "eo": None, "eh": None}
res = {}
for T in ("chicago", "nyc"):
    Y = np.load(YF[T]).astype(np.float64); tot = Y.sum(2)
    idx = [(d, s) for d in range(Y.shape[0]) for s in range(48) if tot[d, s] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx])
    Sh = Y[di, si] / tot[di, si][:, None]; Tt = tot[di, si]; true_year = Y.sum((0, 1)) / Y.sum()
    print(f"\n===== {T}: {Y.shape[2]} regions, {len(idx)} observed slots, ensemble of runs 1..{N} =====")
    for v in BASE:
        fs = [f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{v}_seed{r - 1}.npy" for r in range(1, N + 1)]
        if not all(os.path.exists(f) for f in fs): print(f"   {v:4s} missing runs"); continue
        P = np.mean([p / p.sum(1, keepdims=True) for p in map(np.load, fs)], 0)
        e = P * Tt[:, None] - Y[di, si]; ms = np.abs(P - Sh).mean()
        single = [np.abs(np.load(f) - Sh).mean() for f in fs]
        yr = (P * Tt[:, None]).sum(0); yr /= yr.sum(); top = len(set(np.argsort(-yr)[:10]) & set(np.argsort(-true_year)[:10])) / 10
        res[(T, v)] = ms
        print(f"   {v:4s} MAE_share {ms:.5f} (single runs {np.mean(single):.5f} +- {np.std(single, ddof=1) if N > 1 else 0:.5f}) | "
              f"allocation-only MAE {np.abs(e).mean():.3f} RMSE {np.sqrt((e ** 2).mean()):.3f} | peak ratio {P.max(1).mean() / Sh.max(1).mean():.2f} | top-10 recall {top:.1f}")
print("\n===== screen rule (>= 3% better in one development target, not > 3% worse in the other) =====")
for v, b in BASE.items():
    if b is None or any((T, x) not in res for T in ("chicago", "nyc") for x in (v, b)): continue
    g = {T: 100 * (res[(T, b)] - res[(T, v)]) / res[(T, b)] for T in ("chicago", "nyc")}
    ok = max(g.values()) >= 3 and min(g.values()) >= -3
    print(f"   {v} vs {b}: Chicago {g['chicago']:+.1f}%, NYC {g['nyc']:+.1f}% -> {'PASS' if ok else 'fail'}")
