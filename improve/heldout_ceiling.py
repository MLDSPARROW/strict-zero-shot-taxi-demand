"""Diagnosis for the held-out targets (uses target data; not a method): headroom of the allocation.
(a) PERFECT allocation (true weekday/weekend x slot shares of the target) with the STRICT city total;
(b) final method's shares with the TRUE city total; (c) Poisson noise floor (perfect expected counts)."""
import numpy as np, importlib.util, sys
spec = importlib.util.spec_from_file_location("sa", "score_all.py"); sa = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(sa)
rng = np.random.default_rng(0)
for T in ["chicago", "nyc", "sf", "dc"]:
    Y = sa.Y[T]; tot = Y.sum(2); idx = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Y[di, si]; To = tot[di, si]
    u, lev, Ts = sa.strict_total(T, di, si); we = sa.DOW[T] >= 5
    tru = np.zeros((2, 48, Y.shape[2]))
    for k in (0, 1):
        for s in range(48): v = Y[we == bool(k), s].sum(0); tru[k, s] = v / max(v.sum(), 1e-9)
    pt = tru[we[di].astype(int), si]
    H = sa.MODEL_H[T]
    pf = 0.5 * sa.ens([f"out/multi_{T}_from_{H(r)}_veh_seed{r-1}.npy" for r in range(1, 11)]) + 0.5 * sa.ens([f"out/multi_{T}_from_{H(r)}_veo_seed{r-1}.npy" for r in range(1, 11)])
    def sc(p, TT): E = p * TT[:, None] - Yc; return f"{np.abs(E).mean():.2f} / {np.sqrt((E**2).mean()):.2f}"
    lam = pt * To[:, None]; sim = rng.poisson(lam); fl = f"{np.abs(lam - sim).mean():.2f} / {np.sqrt(((lam - sim)**2).mean()):.2f}"
    print(f"{T:8s} final strict {sc(pf, Ts)} | final + TRUE total {sc(pf, To)} | PERFECT allocation + strict total {sc(pt, Ts)} | "
          f"PERFECT allocation + TRUE total {sc(pt, To)} | Poisson floor {fl}")
