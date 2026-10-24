"""DESCRIPTIVE (development targets only): how the scale-matched XGBoost baseline (out_xgb/, saved by xgb_save.py) allocates trips compared
with the final method and its heads: strict trip MAE / RMSE, yearly total-variation distance, airport share, top-10 recall, the jobs law beta
(v11_law2.py, extensive, airports left out) and the regions with the largest yearly over- and under-prediction.
Usage (from improve/): python v11_xgb_compare.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE = ns["totals"], ns["YP"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})
s2 = open(f"{HERE}/v11_law2.py").read(); n2 = {"__file__": f"{HERE}/v11_law2.py"}; exec(s2[:s2.index("SRC = {")], n2); cov, law = n2["cov"], n2["law"]
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
norm = lambda q: q / q.sum(1, keepdims=True); B = f"{HERE}/../bench77/data"
for T in [t for t in ("chicago", "nyc") if os.path.exists(f"{HERE}/out_xgb/xgb_sm_{t}.npy")]:
    Y = np.load(YP[T]).astype(np.float64); di, si, Tt, unit, lev = totals(T, T); y = Y[di, si]; ty = Y.sum((0, 1)) / Y.sum()
    c, area, air = cov(T); rid = np.load(f"{B}/region_static_{T}.npz")["rid"]
    H = {h: np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0) for h in ("eh", "eo", "ep")}
    H["v1"] = 0.5 * H["eh"] + 0.5 * H["eo"]; H["xgb"] = norm(np.load(f"{HERE}/out_xgb/xgb_sm_{T}.npy").astype(np.float64))
    print(f"\n===== {T}: true airport share {100 * ty[air].sum():.1f}%, true jobs beta {law(ty, c['jobs'], area, air, False):.2f} =====")
    yr = {}
    for h in ("v1", "eh", "eo", "ep", "xgb"):
        e = H[h] * Tt[:, None] - y; q = (H[h] * Tt[:, None]).sum(0); q /= q.sum(); yr[h] = q
        print(f"   {h:4s} strict MAE {np.abs(e).mean():.3f} RMSE {np.sqrt((e ** 2).mean()):7.3f} | yearly TV {0.5 * np.abs(q - ty).sum():.3f} | airports {100 * q[air].sum():5.1f}% | "
              f"top-10 recall {len(set(np.argsort(-q)[:10]) & set(np.argsort(-ty)[:10])) / 10:.1f} | jobs beta {law(q, c['jobs'], area, air, False):.2f} | "
              f"peak ratio {(H[h].max(1) / (y / y.sum(1, keepdims=True)).max(1)).mean():.2f}")
    print("   largest yearly differences XGBoost minus v1 (rid: true %, v1 %, xgb %):")
    d = yr["xgb"] - yr["v1"]
    for i in list(np.argsort(d)[:5]) + list(np.argsort(d)[::-1][:5]): print(f"      {int(rid[i]):4d}: {100 * ty[i]:5.2f} {100 * yr['v1'][i]:5.2f} {100 * yr['xgb'][i]:5.2f}")
