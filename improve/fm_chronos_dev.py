"""PROTOCOL_V8.md: Chronos-Bolt-small on the development period (1 October to 30 November 2021). The same model, context (336 half-hour counts of each
region), horizon (6 slots) and point forecast (mean of the nine quantiles, clipped at 0) as fm_chronos.py (PROTOCOL_V6.md), on origins every 12 slots with
alternating offsets (even days 0, 12, 24, 36; odd days 6, 18, 30, 42), so that every time of the day is covered and the cost is half of the 6-slot grid.
Run from improve/ with the separate environment:  ../.venv_fm/Scripts/python.exe fm_chronos_dev.py   (restartable; chunks are saved)"""
import os, sys, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
try:                                                     # below-normal priority: the laptop is shared with other runs
    import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception: pass
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1"); torch.set_num_threads(2)
B = f"{HERE}/../bench77/data"; YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy"}
OUT = f"{HERE}/out_xc"; CH = f"{OUT}/fm_chunks_dev"; os.makedirs(CH, exist_ok=True)
CTX, HOR, BATCH, D0, D1 = 336, 6, 256, 273, 333


def grid(d0, d1):
    out = []
    for d in range(d0, d1 + 1): out += [d * 48 + s for s in ((0, 12, 24, 36) if d % 2 == 0 else (6, 18, 30, 42))]
    return np.array(out)


from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float32); nd, _, R = Y.shape; flat = Y.reshape(nd * 48, R); orig = grid(D0, D1)
    jobs = [(k, r) for k in orig for r in range(R)]; n = len(jobs); chunk = 2048
    print(f"\n===== {T}: {R} regions, {len(orig)} origins, {n} series to forecast =====", flush=True); t0 = time.time()
    for c0 in range(0, n, chunk):
        f = f"{CH}/{T}_{c0}.npy"
        if os.path.exists(f): continue
        part = jobs[c0:c0 + chunk]; ctx = torch.tensor(np.stack([flat[k - CTX:k, r] for k, r in part]))
        q = torch.cat([pipe.predict(ctx[i:i + BATCH], prediction_length=HOR) for i in range(0, len(part), BATCH)]).numpy()
        np.save(f, q.astype(np.float32)); print(f"   {min(c0 + chunk, n)}/{n} series [{time.time() - t0:.0f}s]", flush=True)
    Q = np.concatenate([np.load(f"{CH}/{T}_{c0}.npy") for c0 in range(0, n, chunk)])
    bad = int((~np.isfinite(Q)).sum()); Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    qm = np.clip(Q.mean(1), 0, None); md = np.clip(Q[:, 4, :], 0, None)
    K = np.array([k for k, _ in jobs]); Rr = np.array([r for _, r in jobs])
    PM = np.zeros((nd * 48, R), np.float32); PD = np.zeros_like(PM); cov = np.zeros(nd * 48, bool)
    for j in range(HOR): PM[K + j, Rr] = qm[:, j]; PD[K + j, Rr] = md[:, j]; cov[K + j] = True
    np.savez(f"{OUT}/fm_dev_{T}.npz", PM=PM, PD=PD, cov=cov)
    e = (PM - flat)[cov]; print(f"   Chronos-Bolt-small, development period: MAE {np.abs(e).mean():.3f} RMSE {np.sqrt((e ** 2).mean()):.3f} (non-finite values replaced: {bad})", flush=True)
print("\ndone")
