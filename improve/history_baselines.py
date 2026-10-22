"""PROTOCOL_V5.md, comparison 2: target-history baselines (NOT strict: they use the observed demand of the previous weeks of the target;
context only). Last week = observed count of day d - 7; historical average = mean of days d - 7, d - 14, d - 21, d - 28 (same weekday, slot,
region). Scored from day 28 on, on the calendar slots of the final method, together with the final method (v1: runs 1 to 10, condition M,
equal average of eh and eo, strict city totals) and the zero forecast on exactly the same days and slots. Hourly for Washington, DC and its
clusters. Usage (from improve/): python history_baselines.py"""
import os, sys, io, contextlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/v1_i1_eval.py").read(); ns = {"__file__": f"{HERE}/v1_i1_eval.py"}
exec(src[:src.index("\n\nfor t in (sys.argv[1:]")], ns)
TG, target, norm, boot, O = ns["TG"], ns["target"], ns["norm"], ns["boot"], ns["O"]
rng = np.random.default_rng(0); D0 = 28
for tag in ("chicago", "nyc", "sf", "dc", "sf_zcta", "dc_cluster", "austin"):
    city, yf, mat, nat, H = TG[tag]; Y, di, si, Tt, ndays, hourly = target(tag); nd = Y.shape[0]
    fin = lambda h: np.mean([norm(np.load(f"{O}/multi_{tag}_from_{mat(r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0)
    P = 0.5 * fin("eh") + 0.5 * fin("eo")
    F = np.full(Y.shape, np.nan); F[di, si] = P * Tt[:, None]
    L7 = np.full(Y.shape, np.nan); L7[7:] = Y[:-7]
    HA = np.full(Y.shape, np.nan); HA[28:] = np.mean([Y[28 - 7 * k: nd - 7 * k] for k in (1, 2, 3, 4)], 0)
    ok = np.zeros(Y.shape, bool); ok[di, si] = True; ok[:D0] = False
    if ndays: ok[ndays:] = False
    agg = (lambda Z: Z.reshape(nd, 24, 2, -1).sum(2)) if hourly else (lambda Z: Z)
    okh = agg(ok.astype(float)) == (2 if hourly else 1)           # an hour counts only if both half-hours are scored
    Yv = agg(np.where(ok, Y, 0.0)); M_ = {"Zero forecast": np.zeros_like(Yv), "Last week (uses target history)": agg(np.where(ok, L7, 0.0)),
                                         "Historical average of 4 weeks (uses target history)": agg(np.where(ok, HA, 0.0)),
                                         "Final method, strict (no target data)": agg(np.where(ok, F, 0.0))}
    n_d = int(ok.any((1, 2)).sum())
    print(f"\n===== {tag} ({'hourly' if hourly else '30-min'}), days {D0 + 1} to {int(np.where(ok.any((1, 2)))[0].max()) + 1}: {n_d} days, {int(okh.sum())} scored region-slots =====")
    dm = {}
    for k, p in M_.items():
        e = (p - Yv)[okh]; dm[k] = np.where(okh, np.abs(p - Yv), np.nan).reshape(nd, -1)
        print(f"   {k:55s} MAE {np.abs(e).mean():.3f}  RMSE {np.sqrt((e ** 2).mean()):.3f}")
    for k in ("Last week (uses target history)", "Historical average of 4 weeks (uses target history)"):
        d = np.nanmean(dm["Final method, strict (no target data)"], 1) - np.nanmean(dm[k], 1); d = d[~np.isnan(d)]; b = boot(d, rng)
        print(f"   final minus {k.split(' (')[0].lower()}: daily MAE {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}], final better on {100 * np.mean(d < 0):.0f}% of days")
print("\ndone")
