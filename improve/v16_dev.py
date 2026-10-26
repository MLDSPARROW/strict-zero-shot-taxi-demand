"""PROTOCOL_V16.md, development (October and November 2021, Chicago and NYC): fixed averages of the forecasts of Chronos-Bolt-small and of the cross-city forecasters of PROTOCOL_V8 (C1 to C4, two
scale-matched partitions each, saved in out_xc/), on the same origins and slots as xc7d.py dev. No new training. Usage (from improve/): python v16_dev.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import xc7d as X
O = X.grid(X.DEV[0], X.DEV[1], True); rng = np.random.default_rng(0); RULES = {"S1": "(Chronos + C2) / 2", "S2": "(2 Chronos + C2) / 3", "S3": "(Chronos + C1 + C2 + C3 + C4) / 5"}; res = {}
for T in ("chicago", "nyc"):
    Y3 = np.load(X.YP[T]); flat = Y3.reshape(-1, Y3.shape[2]).astype(np.float32); PM, PD, cov = X.chronos(T, "dev"); t = O[:, None] + np.arange(X.HOR)[None]; assert cov[t].all()
    cm = np.stack([PM[t[:, j]] for j in range(X.HOR)], 1); C = {c: np.mean([np.load(f"{X.OUT}/pred_dev_{c}_{T}_matched_r{r}.npy") for r in (1, 2)], 0) for c in ("C1", "C2", "C3", "C4")}
    P = {"Chronos-Bolt": cm, "C2 alone": C["C2"], "S1": (cm + C["C2"]) / 2, "S2": (2 * cm + C["C2"]) / 3, "S3": (cm + C["C1"] + C["C2"] + C["C3"] + C["C4"]) / 5}
    out = {k: X.perday(T, O, p, flat) for k, p in P.items()}; dc = out["Chronos-Bolt"][0]; res[T] = {}
    print(f"\n===== {T}: {len(O)} origins, 6 slots each =====")
    for k, (d, m, r_) in out.items():
        line = f"   {k:13s} MAE {m:.3f} RMSE {r_:.3f}"
        if k in RULES:
            df = d - dc; b = X.boot(df, rng); lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5); res[T][k] = (m, r_, out["Chronos-Bolt"][1], out["Chronos-Bolt"][2], hi)
            line += f"  ({RULES[k]}); daily MAE difference to Chronos {df.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], better on {100 * (df < 0).mean():.0f}% of days"
        print(line)
print("\n===== go rule of PROTOCOL_V16: MAE and RMSE below Chronos-Bolt in both cities and the upper end of the 95% interval of the daily MAE difference below zero in both =====")
passing = []
for k in RULES:
    ok = all(res[T][k][0] < res[T][k][2] and res[T][k][1] < res[T][k][3] and res[T][k][4] < 0 for T in res); crit = np.mean([res[T][k][0] / res[T][k][2] for T in res])
    print(f"   {k} {RULES[k]:36s} mean MAE ratio to Chronos {crit:.3f} -> {'PASSES' if ok else 'does not pass'}"); 
    if ok: passing.append((crit, k))
print("   chosen for December: " + (min(passing)[1] if passing else "none, December is NOT run"))
