"""EVALUATION of the aggregation control (PROTOCOL_REVISION.md, part B). Trip errors only.
Conditions: N native helpers (seeds 0-9), M proposed scale matching, R random grouping with the same K, H half and D twice
the target median area. M, R, H and D were all run on the same Kaggle CPU workers (out_kaggle/); the laptop runs of M
(out/) are shown as a check. Final = equal average of the 10-run ensembles of heads eh and eo.
Reports MAE and RMSE, the mean daily MAE difference to M with a 7-day block-bootstrap 95% interval (2,000 draws), and the
mean and standard deviation over the 10 single runs. Usage: python control_eval.py"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); B = f"{HERE}/../bench77/data"; KG = f"{HERE}/out_kaggle"
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy"}
NAT = {"chicago": "nyc+sf", "nyc": "chicago+sf"}
HEL = {"chicago": lambda c, r: f"nyc_for_chicago_{c}{r}+sf_for_chicago_{c}{r}", "nyc": lambda c, r: f"chicago+sf_for_nyc_{c}{r}"}


def boot(x, rng):
    nb = len(x) // 7
    return [x[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(x) - 7, nb)])].mean() for _ in range(2000)]


for T in ("chicago", "nyc"):
    rng = np.random.default_rng(0); P = np.load(f"{HERE}/out_cal/pred_{T}.npz"); di, si, Tt = P["di"], P["si"], P["total"]
    Yc = np.load(YP[T]).astype(np.float64)[di, si]
    norm = lambda p: p / p.sum(1, keepdims=True)
    def files(cond, h):
        if cond == "N": return [f"{HERE}/out/multi_{T}_from_{NAT[T]}_v{h}_seed{s}.npy" for s in range(10)]
        if cond == "M laptop": return [f"{HERE}/out/multi_{T}_from_{HEL[T]('p', r)}_v{h}_seed{r-1}.npy" for r in range(1, 11)]
        c = {"M": "p"}.get(cond, cond); return [f"{KG}/multi_{T}_from_{HEL[T](c, r)}_v{h}_seed{r-1}.npy" for r in range(1, 11)]
    def err(p):
        E = norm(p) * Tt[:, None] - Yc; A = np.abs(E)
        return A.mean(), np.sqrt((E ** 2).mean()), np.bincount(di, A.mean(1)) / np.bincount(di)
    print(f"\n===== {T.upper()} (final = average of heads eh and eo; MAE / RMSE) =====")
    res = {}
    for cond in ("N", "M laptop", "M", "R", "H", "D"):
        F = {h: [np.load(f) for f in files(cond, h)] for h in ("eh", "eo")}
        fin = 0.5 * norm(np.mean(F["eh"], 0)) + 0.5 * norm(np.mean(F["eo"], 0))
        single = np.array([err(0.5 * norm(a) + 0.5 * norm(b))[:2] for a, b in zip(F["eh"], F["eo"])])
        res[cond] = err(fin)
        print(f"   {cond:9s} final {res[cond][0]:.2f} / {res[cond][1]:.2f}  | heads eh {err(np.mean(F['eh'],0))[0]:.2f} / {err(np.mean(F['eh'],0))[1]:.2f}, "
              f"eo {err(np.mean(F['eo'],0))[0]:.2f} / {err(np.mean(F['eo'],0))[1]:.2f} | single runs MAE {single[:,0].mean():.2f} +- {single[:,0].std(ddof=1):.2f}, "
              f"RMSE {single[:,1].mean():.2f} +- {single[:,1].std(ddof=1):.2f}")
    for cond in ("N", "R", "H", "D", "M laptop"):
        d = res[cond][2] - res["M"][2]; b = boot(d, rng)
        print(f"   {cond:9s} minus M: mean daily MAE difference {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}], "
              f"M better on {100*(d>0).mean():.0f}% of days")
