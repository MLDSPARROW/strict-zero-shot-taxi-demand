"""EVALUATION of PROTOCOL_V3.md: the block-size sweep for Washington, DC (hourly) and the mean rule against the median rule at DC tracts and
at the San Francisco ZIP-code areas. Fully strict trip errors on the calendar slots; city totals exactly as coarse_eval.py. Final = equal
average of the heads eh and eo, each the mean of the normalised shares of the runs. Differences are mean daily MAE differences to the median
rule (q = 1) with a 7-day block bootstrap, positive when the condition is better. Conditions whose runs are not finished are skipped.
Usage (from improve/): python scale_rule_eval.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, MC, boot = ns["totals"], ns["YP"], ns["MC"], ns["boot"]
O = f"{HERE}/out_cal"; norm = lambda p: p / p.sum(1, keepdims=True); rng = np.random.default_rng(0)


def ensemble(fe, fo):
    if not all(os.path.exists(f) for f in fe + fo): return None
    return 0.5 * np.mean([norm(np.load(f)) for f in fe], 0) + 0.5 * np.mean([norm(np.load(f)) for f in fo], 0)


def run(title, tag, city, yfile, hourly, conds):
    di, si, Tt, unit, lev = totals(city, tag); Y = np.load(yfile).astype(np.float64); nd = Y.shape[0]
    def err(p):
        F = np.full(Y.shape, np.nan); F[di, si] = norm(p) * Tt[:, None]; E = F - Y
        if hourly: E = E.reshape(nd, 24, 2, -1).sum(2)
        E = E.reshape(nd, -1); A = np.abs(E); return np.nanmean(A), np.sqrt(np.nanmean(E ** 2)), np.nanmean(A, 1)
    res = {}
    for name, (fe, fo) in conds.items():
        p = ensemble(fe, fo)
        if p is None: print(f"   {name:24s} runs not finished, skipped"); continue
        res[name] = err(p)
    ref = res["q=1, median rule"]; print(f"\n===== {title} ({'hourly' if hourly else '30-min'}); final method, MAE / RMSE, difference to the median rule [95% interval], positive = better =====")
    out = {}
    for name, (mae, rmse, daily) in res.items():
        if name == "q=1, median rule": print(f"   {name:24s} {mae:.4f} / {rmse:.4f}"); continue
        d = ref[2] - daily; b = boot(d, rng); lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5); out[name] = (d.mean(), lo, hi)
        print(f"   {name:24s} {mae:.4f} / {rmse:.4f} | {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], better on {100 * np.mean(d > 0):.0f}% of days ({100 * d.mean() / ref[0]:+.1f}% of the median-rule MAE)")
    return res, out


R = range(1, 6)
def dcc(code):                                    # run r uses partition r and seed r - 1
    f = lambda h: [f"{O}/multi_dc_from_chicago+nyc+sf_for_dc_{code}{r}_v{h}_seed{r - 1}.npy" for r in R]
    return f("eh"), f("eo")
nat_eh, nat_eo = [[f"{O}/multi_dc_from_chicago+nyc+sf_v{h}_seed{s}.npy" for s in range(5)] for h in ("eh", "eo")]
dc_conds = {"native helpers": (nat_eh, nat_eo), "q=0.75": dcc("x075p"), "q=1, median rule": dcc("p"), "q=1.5": dcc("x150p"),
            "mean rule (q=1.656)": dcc("xmeanp"), "q=2": dcc("x200p"), "q=4": dcc("x400p")}
res_dc, out_dc = run("Washington, DC tracts, 206 regions", "dc", "dc", YP["dc"], True, dc_conds)

R10 = range(1, 11)
zc = lambda f: ([f(r, "eh") for r in R10], [f(r, "eo") for r in R10])
zc_med = zc(lambda r, h: f"{O}/multi_sf_zcta_from_chicago+nyc_for_sf_zcta_p{r}_v{h}_seed{r - 1}.npy")
zc_mean = zc(lambda r, h: f"{O}/multi_sf_zcta_from_chicago_for_sf_zcta_mp{r}+nyc_for_sf_zcta_mp{r}_v{h}_seed{r - 1}.npy")
res_zc, out_zc = run("San Francisco ZIP-code areas, 26 regions", "sf_zcta", "sf", f"{MC}/sf_zcta_30min.npy", False,
                     {"q=1, median rule": zc_med, "mean rule (q=2.882)": zc_mean})

print("\n===== PRE-REGISTERED READING (PROTOCOL_V3.md) =====")
h1 = out_dc.get("mean rule (q=1.656)"); h2 = out_zc.get("mean rule (q=2.882)")
ok = lambda x: x is not None and x[1] > 0; bad = lambda x: x is not None and x[2] < 0
if h1: print(f"   H1 (DC, mean rule better than median rule, interval excludes zero): {ok(h1)} ({h1[0]:+.4f} [{h1[1]:+.4f}, {h1[2]:+.4f}])")
if h2: print(f"   H2 (SF ZIP, mean rule better than median rule, interval excludes zero): {ok(h2)} ({h2[0]:+.4f} [{h2[1]:+.4f}, {h2[2]:+.4f}])")
if h1 and h2:
    print(f"   mean rule: {'SUPPORTED' if ok(h1) and ok(h2) else 'NOT SUPPORTED' if bad(h1) or bad(h2) else 'INCONCLUSIVE'}")
sweep = {k: v[0] for k, v in res_dc.items() if k.startswith("q=") or k.startswith("native")}
if len(res_dc) >= 6:
    q_of = {"q=0.75": 0.75, "q=1, median rule": 1.0, "q=1.5": 1.5, "q=2": 2.0, "q=4": 4.0}
    best = min(q_of, key=lambda k: res_dc[k][0]) if all(k in res_dc for k in q_of) else None
    if best: print(f"   H3 (DC sweep, lowest MAE at q between 0.5 and 2 and q=4 worse than q=1): best q = {q_of[best]} -> {0.5 <= q_of[best] <= 2 and res_dc['q=4'][0] > res_dc['q=1, median rule'][0]}")
print("done")
