"""PROTOCOL_V16.md, test (December 2021, once): S2 = (2 Chronos-Bolt-small + C2) / 3 with C2 trained for December as in PROTOCOL_V8 (ten scale-matched partitions, helpers before 1 December 2021, seeds r - 1, origins every
6 slots; each slot of December once), against the December forecasts of Chronos-Bolt-small saved by PROTOCOL_V6. Reports Chronos-Bolt, C2 alone, the weekly average and S2, with the 7-day block bootstrap of the daily
MAE difference S2 - Chronos-Bolt, and H1 of the protocol. Restartable (run_one reuses saved files). Usage (from improve/): python v16_dec.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import xc7d as X
O = X.grid(X.DEC[0], X.DEC[1], False); rng = np.random.default_rng(0); h1 = {}
for T in ("chicago", "nyc"):
    for r in range(1, 11): X.run_one(T, "matched", r, "C2", "dec", X.DEC[0], O)
for T in ("chicago", "nyc"):
    Y3 = np.load(X.YP[T]); flat = Y3.reshape(-1, Y3.shape[2]).astype(np.float32); PM, PD, cov = X.chronos(T, "dec"); t = O[:, None] + np.arange(X.HOR)[None]; assert cov[t].all()
    cm = np.stack([PM[t[:, j]] for j in range(X.HOR)], 1); c2 = np.mean([np.load(f"{X.OUT}/pred_dec_C2_{T}_matched_r{r}.npy") for r in range(1, 11)], 0); s2 = (2 * cm + c2) / 3
    b, fl = X.baseline_pred(T, O); P = {"weekly average": b, "Chronos-Bolt": cm, "C2 alone": c2, "S2": s2}; out = {k: X.perday(T, O, p, flat) for k, p in P.items()}
    print(f"\n===== {T}, December 2021, {len(O)} origins every 6 slots =====")
    for k, (d, m, r_) in out.items(): print(f"   {k:15s} MAE {m:.3f} RMSE {r_:.3f}")
    df = out["S2"][0] - out["Chronos-Bolt"][0]; bb = X.boot(df, rng); lo, hi = np.percentile(bb, 2.5), np.percentile(bb, 97.5)
    print(f"   S2 - Chronos-Bolt: daily MAE difference {df.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], S2 better on {100 * (df < 0).mean():.0f}% of days; MAE change {100 * (out['Chronos-Bolt'][1] - out['S2'][1]) / out['Chronos-Bolt'][1]:+.2f}%, RMSE change {100 * (out['Chronos-Bolt'][2] - out['S2'][2]) / out['Chronos-Bolt'][2]:+.2f}% (positive = S2 better)")
    h1[T] = out["S2"][1] < out["Chronos-Bolt"][1] and out["S2"][2] < out["Chronos-Bolt"][2] and hi < 0
print(f"\nH1 (MAE and RMSE below Chronos-Bolt in both cities, upper end of the interval below zero in both): {all(h1.values())} {h1}")
