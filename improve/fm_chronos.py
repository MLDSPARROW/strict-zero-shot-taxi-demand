"""PROTOCOL_V6.md: Chronos-Bolt (small) as version D of the cost-of-strictness table. Run with the separate environment:
  ../.venv_fm/Scripts/python.exe fm_chronos.py          (from improve/)
Context = previous 336 half-hour counts of each region, forecast = next 6 slots, origins every 6 slots through December 2021, so each
December slot is forecast once. Point forecast = mean of the nine quantiles (clipped at 0) for MAE and RMSE; median-forecast MAE also
reported. Version A (strict final method) is recomputed for the same slots as a consistency check. Restartable (chunks are saved)."""
import os, sys, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1"); torch.set_num_threads(2)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE = ns["totals"], ns["YP"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})                  # as predict_final.py
O = f"{HERE}/out_cal"; CH = f"{O}/fm_chunks"; os.makedirs(CH, exist_ok=True)
MAT = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
PAPER_A = {"chicago": (2.18, 6.31), "nyc": (9.44, 21.02)}                            # table of the paper, version A
CTX, HOR, D0, D1, BATCH = 336, 6, 334, 364, 256                                      # December 2021 = days 334 to 364 (0 = 1 January)
norm = lambda p: p / p.sum(1, keepdims=True)
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float32); nd, _, R = Y.shape; flat = Y.reshape(nd * 48, R)
    orig = np.array([d * 48 + s for d in range(D0, D1 + 1) for s in range(0, 48, HOR)])            # forecast origins (flat slot index)
    jobs = [(k, r) for k in orig for r in range(R)]; n = len(jobs); chunk = 2048
    print(f"\n===== {T}: {R} regions, {len(orig)} origins, {n} series to forecast =====", flush=True); t0 = time.time()
    for c0 in range(0, n, chunk):
        f = f"{CH}/{T}_{c0}.npy"
        if os.path.exists(f): continue
        part = jobs[c0:c0 + chunk]; ctx = torch.tensor(np.stack([flat[k - CTX:k, r] for k, r in part]))
        q = torch.cat([pipe.predict(ctx[i:i + BATCH], prediction_length=HOR) for i in range(0, len(part), BATCH)]).numpy()   # chunk x 9 x 6
        np.save(f, q.astype(np.float32)); print(f"   {min(c0 + chunk, n)}/{n} series [{time.time() - t0:.0f}s]", flush=True)
    Q = np.concatenate([np.load(f"{CH}/{T}_{c0}.npy") for c0 in range(0, n, chunk)])
    bad = int((~np.isfinite(Q)).sum()); Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    qm = np.clip(Q.mean(1), 0, None); md = np.clip(Q[:, 4, :], 0, None)                             # quantile mean, median: chunk-series x 6
    K = np.array([k for k, _ in jobs]); Rr = np.array([r for _, r in jobs])
    PM = np.zeros((nd * 48, R), np.float32); PD = np.zeros_like(PM)
    for j in range(HOR): PM[K + j, Rr] = qm[:, j]; PD[K + j, Rr] = md[:, j]
    sl = slice(D0 * 48, (D1 + 1) * 48); y = flat[sl]; e = PM[sl] - y; e_md = PD[sl] - y
    hz = [np.abs(PM[K + j, Rr] - flat[K + j, Rr]).mean() for j in range(HOR)]
    print(f"   D, Chronos-Bolt-small, previous 7 days, 1 to 6 slots ahead: MAE {np.abs(e).mean():.3f}  RMSE {np.sqrt((e ** 2).mean()):.3f}"
          f"   (median forecast MAE {np.abs(e_md).mean():.3f}; non-finite values replaced: {bad})")
    print("   MAE by horizon step 1..6:", " ".join(f"{h:.3f}" for h in hz))
    di, si, Tt, unit, lev = totals(T, T)
    fin = lambda h: np.mean([norm(np.load(f"{O}/multi_{T}_from_{MAT[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0)
    A = np.full(Y.shape, np.nan); A[di, si] = (0.5 * fin("eh") + 0.5 * fin("eo")) * Tt[:, None]
    ea = (A.reshape(nd * 48, R)[sl] - y); ok = ~np.isnan(ea); am, ar = np.abs(ea[ok]).mean(), np.sqrt((ea[ok] ** 2).mean())
    pm, pr = PAPER_A[T]
    print(f"   A, strictly zero-shot (recomputed): MAE {am:.3f}  RMSE {ar:.3f}   table: {pm} / {pr} -> "
          f"{'OK (within 0.02)' if abs(am - pm) <= 0.02 and abs(ar - pr) <= 0.02 else 'REPORT: differs from the table'}")
    dm, dr = np.abs(e).mean(), np.sqrt((e ** 2).mean())
    print(f"   cost of strictness against D: MAE {am - dm:+.2f}, {100 * (am - dm) / dm:+.0f}% | RMSE {ar - dr:+.2f}, {100 * (ar - dr) / dr:+.0f}%")
print("\ndone")
