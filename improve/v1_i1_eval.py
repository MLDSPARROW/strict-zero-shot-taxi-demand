"""PROTOCOL_V2.md addendum 3: the final (v1) method scored with the I1 point forecasts on every target. City totals, slots and
mean forecasts are those of the paper (coarse_eval.py / austin_eval.py); first, the rebuilt mean-forecast MAE and RMSE are compared
with the archived values (tolerance 0.02 on MAE, as declared). Then, per target: mean-forecast MAE / RMSE, MAE of the median of
a negative binomial predictive distribution (dispersion from the target's share helpers, condition M partition 1), coverage of
the central 80% interval, the zero forecast and all baselines, and paired 7-day block bootstraps of daily differences
(median-forecast MAE against the zero forecast and the best baseline; MSE against the best-RMSE baseline).
Target demand is read for scoring only. Usage (from improve/): python v1_i1_eval.py [targets, default all]"""
import os, sys, io, contextlib, numpy as np
from datetime import date
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from point_functional import dispersion, Quantiles
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, MC, YP, boot, SHARE = ns["totals"], ns["MC"], ns["YP"], ns["boot"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})                  # as predict_final.py (checked in chi_eval.py)
B, AG, O = f"{HERE}/../bench77/data", f"{HERE}/data/agg", f"{HERE}/out_cal"
norm = lambda p: p / p.sum(1, keepdims=True)
TG = {"chicago": ("chicago", YP["chicago"], lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", None, ["nyc_for_chicago_p1", "sf_for_chicago_p1"]),
      "nyc": ("nyc", YP["nyc"], lambda r: f"chicago+sf_for_nyc_p{r}", None, ["chicago", "sf_for_nyc_p1"]),
      "sf": ("sf", YP["sf"], lambda r: "chicago+nyc", None, ["chicago", "nyc"]),
      "dc": ("dc", YP["dc"], lambda r: f"chicago+nyc+sf_for_dc_p{r}", "chicago+nyc+sf", ["chicago", "nyc", "sf_for_dc_p1"]),
      "sf_zcta": ("sf", f"{MC}/sf_zcta_30min.npy", lambda r: f"chicago+nyc_for_sf_zcta_p{r}", "chicago+nyc", ["chicago", "nyc_for_sf_zcta_p1"]),
      "dc_cluster": ("dc", f"{MC}/dc_cluster_30min.npy", lambda r: f"chicago+nyc_for_dc_cluster_p{r}+sf_for_dc_cluster_p{r}", "chicago+nyc+sf",
                     ["chicago", "nyc_for_dc_cluster_p1", "sf_for_dc_cluster_p1"]),
      "austin": ("austin", None, lambda r: f"chicago+nyc+sf_for_austin_p{r}", "chicago+nyc+sf", ["chicago", "nyc", "sf_for_austin_p1"])}
PAPER = {"chicago": (1.83, 5.21), "nyc": (7.45, 17.59), "sf": (0.48, 1.08), "dc": (0.86, 2.66), "sf_zcta": (3.443, 6.262),
         "dc_cluster": (3.181, 7.998), "austin": (0.121, 1.096)}          # archived values of the final method (paper tables, results/*.txt)
GOAL = {"nyc": "MAE < 6.63 (zero forecast) and < 6.83 (UrbanGPT); RMSE < 17.51 (jobs-proportional)", "sf": "MAE <= 0.40, RMSE < 0.98 (Chi et al.)",
        "dc": "MAE < 0.66, RMSE < 2.40 (jobs + residents), hourly", "chicago": "not more than 3% worse than 1.83 / 5.21"}


def helper(tag):
    if tag in ("chicago", "nyc"): return np.load(f"{B}/{tag}_2021_{'ca' if tag == 'chicago' else 'zone'}_30min.npy"), date(2021, 1, 1)
    if tag == "sf": return np.load(YP["sf"]), date(2023, 1, 1)
    return np.load(f"{AG}/{tag}_Y.npy"), (date(2023, 1, 1) if tag.startswith("sf_") else date(2021, 1, 1))


def target(tag):
    city, yf, mat, nat, H = TG[tag]
    if tag == "austin":
        with contextlib.redirect_stdout(io.StringIO()):
            a = {"__file__": f"{HERE}/austin_eval.py"}; exec(open(f"{HERE}/austin_eval.py").read(), a)
        return a["Y"], a["di"], a["si"], a["Tt"], a["ndays"], False
    di, si, Tt, unit, lev = totals(city, tag); return np.load(yf).astype(np.float64), di, si, Tt, None, city == "dc"


def evaluate(tag):
    city, yf, mat, nat, H = TG[tag]; Y, di, si, Tt, ndays, hourly = target(tag); n = len(di)
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); A = np.load(f"{HERE}/data/acs_{tag}.npz"); R = len(E_["jobs"])
    rep = lambda v: np.repeat((v / v.sum())[None], n, 0)
    ld = lambda f: np.mean([norm(np.load(f(r))) for r in range(1, 11)], 0)
    fin = lambda src_, h: (lambda r: f"{O}/multi_{tag}_from_{src_(r)}_v{h}_seed{r - 1}.npy")
    M = {"Zero forecast": None, "Uniform": np.full((n, R), 1 / R), "Jobs-proportional": rep(E_["jobs"]), "Jobs + residents": rep(E_["jobs"] + E_["res"]),
         "Chi et al. allocation": np.mean([norm(np.load(f"{O}/chi_{tag}_seed{s}.npy")) for s in range(10)], 0)}
    if nat: M["Ours v1, native helpers (N)"] = 0.5 * ld(fin(lambda r: nat, "eh")) + 0.5 * ld(fin(lambda r: nat, "eo"))
    M["Ours v1, final (M)"] = 0.5 * ld(fin(mat, "eh")) + 0.5 * ld(fin(mat, "eo"))
    k_of, tab = dispersion([helper(h) for h in H], hourly=hourly); Q = Quantiles(k_of, 2 * float(Tt.max()) + 1)
    Yv = Y.reshape(Y.shape[0], 24, 2, -1).sum(2) if hourly else Y
    print(f"\n===== {tag} ({'hourly' if hourly else '30-min'}, {R} regions, {n} calendar slots); dispersion from {', '.join(H)} =====")
    print(f"   {'method':32s} mean-forecast MAE / RMSE | median-forecast MAE | 80% coverage")
    D = {}
    for name, p in M.items():
        L = np.full(Y.shape, np.nan); L[di, si] = 0.0 if p is None else norm(p) * Tt[:, None]
        if hourly: L = L.reshape(Y.shape[0], 24, 2, -1).sum(2)
        ok = ~np.isnan(L); assert np.nanmax(L) <= Q.g[-1]; lam, y = L[ok], Yv[ok]; med = Q(lam, 0.5)
        Med = np.full(Yv.shape, np.nan); Med[ok] = med; Em = L - Yv
        f = lambda Z: (Z.reshape(Yv.shape[0], -1))[:ndays] if ndays else Z.reshape(Yv.shape[0], -1)
        D[name] = {"mse": np.nanmean(f(Em) ** 2, 1), "med": np.nanmean(np.abs(f(Med - Yv)), 1)}
        mae, rmse = np.abs(lam - y).mean(), np.sqrt(((lam - y) ** 2).mean()); cov = np.mean((y >= Q(lam, 0.1)) & (y <= Q(lam, 0.9)))
        print(f"   {name:32s} {mae:.3f} / {rmse:.3f}               | {np.abs(med - y).mean():.3f}                | {cov:.3f}")
        if name == "Ours v1, final (M)":
            pm, pr = PAPER[tag]; d = abs(mae - pm)
            print(f"   reproduction check vs archived final method {pm} / {pr}: MAE difference {mae - pm:+.3f}, RMSE difference {rmse - pr:+.3f} -> "
                  f"{'OK (within 0.02)' if d <= 0.02 else 'REPORT: outside the declared 0.02 tolerance'}")
    rng = np.random.default_rng(0); ours = "Ours v1, final (M)"; base = [k for k in M if not k.startswith(("Ours", "Zero"))]
    bm, bs = min(base, key=lambda k: np.nanmean(D[k]["med"])), min(base, key=lambda k: np.nanmean(D[k]["mse"]))
    for ref in ("Zero forecast", bm):
        d = D[ref]["med"] - D[ours]["med"]; b = boot(d, rng)
        print(f"   {ref} minus ours, daily median-forecast MAE: {np.nanmean(d):+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], ours better on {100 * np.mean(d > 0):.0f}% of days")
    d = D[bs]["mse"] - D[ours]["mse"]; b = boot(d, rng)
    print(f"   {bs} (best RMSE baseline) minus ours, daily MSE: {np.nanmean(d):+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], ours better on {100 * np.mean(d > 0):.0f}% of days")
    if tag in GOAL: print(f"   success criterion fixed in PROTOCOL_V2.md: {GOAL[tag]}")


for t in (sys.argv[1:] or list(TG)): evaluate(t)
print("\ndone")
