"""PROTOCOL_V17.md step 1: read-only diagnostics of the saved development forecasts of C2 (mean of partitions 1, 2) against Chronos-Bolt-small. Run with the .venv310 interpreter from improve/."""
import numpy as np, sys
sys.argv = ["x"]; import xc7d as X
O = X.grid(*X.DEV, True); H = X.HOR
for T in ("chicago", "nyc"):
    Y3 = np.load(X.YP[T]); flat = Y3.reshape(-1, Y3.shape[2]).astype(np.float32); R = flat.shape[1]
    p = np.mean([np.load(f"{X.OUT}/pred_dev_C2_{T}_matched_r{r}.npy") for r in (1, 2)], 0)          # (no, H, R)
    PM, PD, cov = X.chronos(T, "dev"); t = O[:, None] + np.arange(H)[None]
    c = np.stack([PM[t[:, j]] for j in range(H)], 1); y = np.stack([flat[t[:, j]] for j in range(H)], 1)
    ea, ec = np.abs(p - y), np.abs(c - y); sa, sc_ = (p - y) ** 2, (c - y) ** 2
    print(f"\n===== {T}: C2 {ea.mean():.3f} / {np.sqrt(sa.mean()):.3f}, Chronos {ec.mean():.3f} / {np.sqrt(sc_.mean()):.3f} =====")
    print("A. by step j: MAE C2, Chronos, gap %; RMSE C2, Chronos")
    for j in range(H): print(f"   j={j + 1}: {ea[:, j].mean():.3f} {ec[:, j].mean():.3f} {100 * (ea[:, j].mean() / ec[:, j].mean() - 1):+.1f}% | {np.sqrt(sa[:, j].mean()):.3f} {np.sqrt(sc_[:, j].mean()):.3f}")
    ctx = np.stack([flat[o - 336:o] for o in O])                                                     # (no, 336, R)
    mean_ctx = ctx.mean(1)                                                                           # (no, R)
    qs = np.quantile(mean_ctx.mean(0), [.2, .4, .6, .8]); rb = np.digitize(mean_ctx.mean(0), qs)     # region bin by mean count
    print("B. by region size bin (mean count in 336 slots): n regions, mean count, MAE C2, Chronos, gap %; RMSE")
    for b in range(5):
        m = rb == b; print(f"   bin {b}: {m.sum()} regions, mean {mean_ctx[:, m].mean():.2f}: {ea[:, :, m].mean():.3f} {ec[:, :, m].mean():.3f} {100 * (ea[:, :, m].mean() / ec[:, :, m].mean() - 1):+.1f}% | {np.sqrt(sa[:, :, m].mean()):.3f} {np.sqrt(sc_[:, :, m].mean()):.3f}")
    hour = (t % 48) // 2
    d = t // 48; wk = np.array([(X.date(2021, 1, 1).toordinal() + int(x) - 1) % 7 for x in d.reshape(-1)]).reshape(d.shape) >= 5
    print("C. by hour block of the forecast slot and weekday / weekend: MAE C2, Chronos, gap %")
    for nm, lo, hi in (("night 0-5", 0, 6), ("morning 6-9", 6, 10), ("midday 10-15", 10, 16), ("evening 16-20", 16, 21), ("late 21-23", 21, 24)):
        m = (hour >= lo) & (hour < hi); print(f"   {nm:14s}: {ea[m].mean():.3f} {ec[m].mean():.3f} {100 * (ea[m].mean() / ec[m].mean() - 1):+.1f}%")
    for nm, m in (("weekday", ~wk), ("weekend", wk)): print(f"   {nm:14s}: {ea[m].mean():.3f} {ec[m].mean():.3f} {100 * (ea[m].mean() / ec[m].mean() - 1):+.1f}%")
    ls = np.abs(ctx[:, -12:].mean(1) - ctx[:, -60:-12].mean(1)) / (ctx.std(1) + 0.1)                 # level-shift statistic (no, R)
    dd = (ea - ec).mean(1)                                                                          # (no, R)
    qd = np.quantile(ls, np.linspace(0, 1, 6)); print("D. level-shift quintiles: mean (|C2 error| - |Chronos error|), share of rows where C2 is better")
    for k in range(5):
        m = (ls >= qd[k]) & (ls <= qd[k + 1]); print(f"   quintile {k + 1} (stat {qd[k]:.2f}-{qd[k + 1]:.2f}): {dd[m].mean():+.4f}, C2 better {100 * (dd[m] < 0).mean():.0f}%")
