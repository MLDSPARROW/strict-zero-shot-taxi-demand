"""PROTOCOL_V5.md, comparison 1: the allocation of Chi et al. on native helpers (Chi-N, the ten seeds of PROTOCOL_CHI_BASELINE.md) against
the same allocation on the scale-matched helpers (Chi-M, ten runs of runner_chi_m.py), fully strict trip MAE and RMSE with the city totals,
slots and hourly scoring of the final method (v1_i1_eval.py), next to the final method (v1: condition M, runs 1 to 10, eh and eo). Paired
7-day block bootstrap (2,000 draws) of the mean daily MAE difference Chi-N minus Chi-M (positive = Chi-M better). Reading rule fixed in the
protocol: GENERALISES if Chi-M is better with an interval excluding zero in at least 4 of the 6 targets and not significantly worse in any;
NOT GENERALISING if significantly worse in at least 3; otherwise MIXED. Usage (from improve/): python chi_m_eval.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/v1_i1_eval.py").read(); ns = {"__file__": f"{HERE}/v1_i1_eval.py"}
exec(src[:src.index("\n\nfor t in (sys.argv[1:]")], ns)
TG, target, norm, boot, O = ns["TG"], ns["target"], ns["norm"], ns["boot"], ns["O"]
rng = np.random.default_rng(0); better, worse, rows = [], [], {}
for tag in ("chicago", "nyc", "dc", "sf_zcta", "dc_cluster", "austin"):
    city, yf, mat, nat, H = TG[tag]; Y, di, si, Tt, ndays, hourly = target(tag); nd = Y.shape[0]
    Yv = Y.reshape(nd, 24, 2, -1).sum(2) if hourly else Y
    f = lambda Z: (Z.reshape(nd, -1))[:ndays] if ndays else Z.reshape(nd, -1)
    def score(p):
        L = np.full(Y.shape, np.nan); L[di, si] = norm(p) * Tt[:, None]
        if hourly: L = L.reshape(nd, 24, 2, -1).sum(2)
        E = f(L - Yv); ok = ~np.isnan(E); return np.abs(E[ok]).mean(), np.sqrt((E[ok] ** 2).mean()), np.nanmean(np.abs(E), 1)
    chi = lambda suf: [np.load(f"{O}/chi_{tag}{suf}_seed{s}.npy") for s in range(10)]
    N, Mc = chi(""), chi("_M"); ens = lambda r: np.mean([norm(x) for x in r], 0)
    ours = 0.5 * np.mean([norm(np.load(f"{O}/multi_{tag}_from_{mat(r)}_veh_seed{r - 1}.npy")) for r in range(1, 11)], 0) \
        + 0.5 * np.mean([norm(np.load(f"{O}/multi_{tag}_from_{mat(r)}_veo_seed{r - 1}.npy")) for r in range(1, 11)], 0)
    sN, sM, sO = score(ens(N)), score(ens(Mc)), score(ours)
    d = sN[2] - sM[2]; d = d[~np.isnan(d)]; b = boot(d, rng); lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
    singN = np.array([score(x)[:2] for x in N]); singM = np.array([score(x)[:2] for x in Mc])
    print(f"\n===== {tag} ({'hourly' if hourly else '30-min'}) =====")
    print(f"   Chi-N, native helpers      MAE {sN[0]:.3f}  RMSE {sN[1]:.3f}   (single runs {singN[:, 0].mean():.3f} +- {singN[:, 0].std(ddof=1):.3f})")
    print(f"   Chi-M, scale-matched       MAE {sM[0]:.3f}  RMSE {sM[1]:.3f}   (single runs {singM[:, 0].mean():.3f} +- {singM[:, 0].std(ddof=1):.3f})")
    print(f"   final method (v1)          MAE {sO[0]:.3f}  RMSE {sO[1]:.3f}")
    print(f"   Chi-N minus Chi-M, daily MAE: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], Chi-M better on {100 * np.mean(d > 0):.0f}% of days; "
          f"MAE change {100 * (sN[0] - sM[0]) / sN[0]:+.1f}%, RMSE change {100 * (sN[1] - sM[1]) / sN[1]:+.1f}% (positive = Chi-M better)")
    if lo > 0: better.append(tag)
    if hi < 0: worse.append(tag)
print(f"\nChi-M significantly better in {len(better)} of 6 targets {better}; significantly worse in {len(worse)} {worse}")
print("READING:", "GENERALISES" if len(better) >= 4 and not worse else "NOT GENERALISING" if len(worse) >= 3 else "MIXED")
print("done")
