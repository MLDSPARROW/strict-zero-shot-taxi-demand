"""PROTOCOL_V12.md, development targets Chicago and NYC: the fixed rules R1 to R3 against the final method v1 (strict trips, the city totals of the paper,
mean daily MAE difference v1 - rule with the 7-day block bootstrap, 2,000 draws). Same inputs as v10_dev.py. Usage (from improve/): python v12_dev.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v12_rules import rules, norm
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE, boot = ns["totals"], ns["YP"], ns["SHARE"], ns["boot"]
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
rng = np.random.default_rng(0); summ = {}
for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float64); nd = Y.shape[0]; di, si, Tt, unit, lev = totals(T, T); y = Y[di, si]
    H = {h: np.mean([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0) for h in ("eh", "eo")}
    v1 = 0.5 * H["eh"] + 0.5 * H["eo"]; g = norm(np.load(f"{HERE}/out_xgb/xgb_sm_{T}.npy").astype(np.float64))
    E_ = np.load(f"{HERE}/data/extra_{T}.npz"); jr = (E_["jobs"] + E_["res"]) / (E_["jobs"] + E_["res"]).sum()
    rows = {"final method v1": v1, "XGBoost scale-matched": g, "Jobs + residents": norm(np.repeat(jr[None], len(v1), 0)), **rules(v1, g, jr)}
    out, daily = {}, {}
    for k, P in rows.items():
        e = P * Tt[:, None] - y; a = np.abs(e); out[k] = (a.mean(), np.sqrt((e ** 2).mean()))
        daily[k] = np.bincount(di, weights=a.mean(1), minlength=nd) / np.maximum(np.bincount(di, minlength=nd), 1)
    print(f"\n===== {T}: {len(jr)} regions, {len(di)} slots =====")
    m0, r0 = out["final method v1"]
    for k, (m, r_) in out.items():
        if k.startswith("R"):
            d = daily["final method v1"] - daily[k]; b = boot(d, rng)
            print(f"   {k:46s} MAE {m:.3f} RMSE {r_:.3f} | v1 - rule {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}] | MAE change {100 * (m0 - m) / m0:+.2f}% RMSE change {100 * (r0 - r_) / r0:+.2f}%")
            summ.setdefault(k, {})[T] = (100 * (m0 - m) / m0, 100 * (r0 - r_) / r0)
        else: print(f"   {k:46s} MAE {m:.3f} RMSE {r_:.3f}")
print("\n===== development rule of PROTOCOL_V12 (comparator v1; mean MAE change >= +3%, no city MAE change below -2%, no city RMSE change below -5%) =====")
for k, v in summ.items():
    mean = np.mean([v[t][0] for t in v]); ok = mean >= 3 and all(v[t][0] >= -2 for t in v) and all(v[t][1] >= -5 for t in v)
    print(f"   {k:46s} mean MAE change {mean:+.2f}% | Chicago {v['chicago'][0]:+.2f}% / {v['chicago'][1]:+.2f}%, NYC {v['nyc'][0]:+.2f}% / {v['nyc'][1]:+.2f}% | {'PASSES' if ok else 'does not pass'}")
