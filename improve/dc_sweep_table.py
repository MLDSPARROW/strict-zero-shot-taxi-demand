"""Table of the block-size sweep for Washington, DC (PROTOCOL_V3.md): final method, five runs per head, hourly, fully strict, in the format of the
sweep table of the paper (ensemble MAE / RMSE, single runs mean +- SD, mean daily MAE difference to q = 1 with the 7-day block bootstrap; here a
POSITIVE difference means a HIGHER error than q = 1, as in the paper's table). Single run r = average of run r of the two heads.
Usage (from improve/): python dc_sweep_table.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, boot = ns["totals"], ns["YP"], ns["boot"]
O = f"{HERE}/out_cal"; norm = lambda p: p / p.sum(1, keepdims=True); rng = np.random.default_rng(0)
di, si, Tt, unit, lev = totals("dc", "dc"); Y = np.load(YP["dc"]).astype(np.float64); nd = Y.shape[0]
def err(p):
    F = np.full(Y.shape, np.nan); F[di, si] = norm(p) * Tt[:, None]; E = (F - Y).reshape(nd, 24, 2, -1).sum(2).reshape(nd, -1); A = np.abs(E)
    return A.mean(), np.sqrt((E ** 2).mean()), A.mean(1)
R = range(1, 6)
def files(code, h): return [f"{O}/multi_dc_from_chicago+nyc+sf_for_dc_{code}{r}_v{h}_seed{r - 1}.npy" for r in R]
nat = lambda h: [f"{O}/multi_dc_from_chicago+nyc+sf_v{h}_seed{s}.npy" for s in range(5)]
COND = [("Native helpers (q <= 0.5)", nat("eh"), nat("eo")), ("0.75", files("x075p", "eh"), files("x075p", "eo")), ("1, proposed", files("p", "eh"), files("p", "eo")),
        ("1.5", files("x150p", "eh"), files("x150p", "eo")), ("2", files("x200p", "eh"), files("x200p", "eo")), ("4", files("x400p", "eh"), files("x400p", "eo"))]
res = {}
for name, fe, fo in COND:
    E_, O_ = [norm(np.load(f)) for f in fe], [norm(np.load(f)) for f in fo]
    ens = err(0.5 * np.mean(E_, 0) + 0.5 * np.mean(O_, 0)); single = np.array([err(0.5 * a + 0.5 * b)[:2] for a, b in zip(E_, O_)]); res[name] = (ens, single)
ref = res["1, proposed"][0][2]
print("Washington, DC (hourly), final method, five runs per head: MAE / RMSE | single runs MAE +- SD, RMSE +- SD | difference to q = 1 [95% interval], positive = higher error")
for name, ((mae, rmse, daily), sg) in res.items():
    if name == "1, proposed": print(f"   {name:26s} {mae:.3f} / {rmse:.2f} | {sg[:, 0].mean():.3f} +- {sg[:, 0].std(ddof=1):.3f}, {sg[:, 1].mean():.2f} +- {sg[:, 1].std(ddof=1):.2f} |"); continue
    d = daily - ref; b = boot(d, rng)
    print(f"   {name:26s} {mae:.3f} / {rmse:.2f} | {sg[:, 0].mean():.3f} +- {sg[:, 0].std(ddof=1):.3f}, {sg[:, 1].mean():.2f} +- {sg[:, 1].std(ddof=1):.2f} | {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
