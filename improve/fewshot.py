"""PROTOCOL_V7.md: the strict forecast plus the previous seven days of the target (Gamma-Poisson shrinkage; no training).
Step 1: choose a on October and November 2021 of Chicago and New York City. Step 2: evaluate on December 2021 with the frozen a.
Usage (from improve/): python fewshot.py"""
import os, sys, numpy as np
from datetime import timedelta
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE, START = ns["totals"], ns["YP"], ns["SHARE"], ns["START"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})                  # as predict_final.py
O = f"{HERE}/out_cal"; HOR = 6; GRID = [0, 0.5, 1, 2, 4, 8, 16, 32, 64]; CITIES = ("chicago", "nyc")
MAT = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
norm = lambda p: p / p.sum(1, keepdims=True)


def load(T):
    Y = np.load(YP[T]).astype(np.float64); nd, _, R = Y.shape; flat = Y.reshape(nd * 48, R)
    di, si, Tt, unit, lev = totals(T, T)
    fin = lambda h: np.mean([norm(np.load(f"{O}/multi_{T}_from_{MAT[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0)
    F = np.full((nd * 48, R), np.nan); F[di * 48 + si] = (0.5 * fin("eh") + 0.5 * fin("eo")) * Tt[:, None]     # strict forecast
    That = np.full(nd * 48, np.nan); That[di * 48 + si] = Tt
    wk = np.array([(START[T] + timedelta(days=d)).weekday() >= 5 for d in range(nd)])
    return flat, F, That, flat.sum(1), wk


def forecasts(D, origins, grid):
    flat, F, That, Tobs, wk = D; res = {a: [] for a in grid}; truth, strict, hz = [], [], []
    same = lambda t: [t - 48 * i for i in range(1, 8) if wk[(t - 48 * i) // 48] == wk[t // 48]]      # days of the same type in the last 7
    for t0 in origins:
        W = slice(t0 - 336, t0); m = ~np.isnan(That[W]); rw = Tobs[W][m].sum() / That[W][m].sum()
        us = np.arange(t0 - HOR, t0); rho = float(np.clip((Tobs[us].sum() + 1) / (sum(Tobs[same(u)].mean() for u in us) + 1), 0.5, 2.0))
        for j in range(HOR):
            t = t0 + j
            if np.isnan(F[t, 0]): continue                                                          # no strict forecast for this slot
            idx = same(t); S = flat[idx].sum(0); n = len(idx); lam0 = np.clip(rw * F[t], 1e-6, None)
            for a in grid: res[a].append((a + rho * S) / (a / lam0 + n))
            truth.append(flat[t]); strict.append(F[t]); hz.append(j)
    return {a: np.array(v) for a, v in res.items()}, np.array(truth), np.array(strict), np.array(hz)


def score(p, y): e = p - y; return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


D = {T: load(T) for T in CITIES}
tune = [d * 48 + s for d in range(273, 334) for s in range(0, 48, HOR)]                  # October and November 2021
dec = [d * 48 + s for d in range(334, 365) for s in range(0, 48, HOR)]                   # December 2021
tab = {}
print("===== step 1: choice of a on October and November 2021 (MAE / RMSE of the mean forecast) =====")
for T in CITIES:
    res, y, st, hz = forecasts(D[T], tune, GRID); tab[T] = {a: score(res[a], y) for a in GRID}
    print(f"   {T}: strict forecast {score(st, y)[0]:.3f} / {score(st, y)[1]:.3f} | " + " | ".join(f"a={a:g}: {tab[T][a][0]:.3f} / {tab[T][a][1]:.3f}" for a in GRID))
crit = {a: np.mean([tab[T][a][0] / tab[T][0][0] for T in CITIES]) for a in GRID}
A_ = min(GRID, key=lambda a: (crit[a], a)); print("   criterion mean MAE(a)/MAE(0):", " ".join(f"{a:g}:{crit[a]:.3f}" for a in GRID), f"-> a = {A_:g} (frozen)")
print(f"\n===== step 2: December 2021 with a = {A_:g} =====")
R_ = {}
for T in CITIES:
    res, y, st, hz = forecasts(D[T], dec, sorted({0, A_})); R_[T] = {"strict": score(st, y), "data": score(res[0], y), "fs": score(res[A_], y)}
    h = [score(res[A_][hz == j], y[hz == j])[0] for j in range(HOR)]
    print(f"   {T}: strict forecast (A) {R_[T]['strict'][0]:.3f} / {R_[T]['strict'][1]:.3f} | week of data only (a=0) {R_[T]['data'][0]:.3f} / {R_[T]['data'][1]:.3f} | "
          f"strict + week (a={A_:g}) {R_[T]['fs'][0]:.3f} / {R_[T]['fs'][1]:.3f} | MAE by horizon step 1..6: " + " ".join(f"{x:.3f}" for x in h))
print("\nH1 (strict + week better than strict, MAE and RMSE, both cities):", all(R_[T]["fs"][0] < R_[T]["strict"][0] and R_[T]["fs"][1] < R_[T]["strict"][1] for T in CITIES))
print("H2 (strict + week better than the week alone, MAE, both cities):", all(R_[T]["fs"][0] < R_[T]["data"][0] for T in CITIES))
print("H3 (against Chronos-Bolt-small of PROTOCOL_V6): compare with results/fm_chronos.txt when that run has finished")
print("done")
