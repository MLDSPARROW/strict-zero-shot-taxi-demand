"""EXPLORATION on the development targets Chicago and NYC only (PROTOCOL_V2.md addendum 6; no held-out city is read).
For every method (v1 final, Chi et al., jobs, jobs + residents, uniform) the median-forecast MAE under a negative binomial
predictive distribution with a CONSTANT dispersion k, over a grid of k, plus the best k on this grid. Question: is the loss of
the learned allocator against the zero forecast (NYC) due to over-confident predictive distributions, and does the ranking of
the methods change when every method is given its own best constant dispersion? (Oracle on the development target: an
upper bound for the cross-city-calibrated version, not a result.)  Usage (from improve/): python explore_dev2.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/v1_i1_eval.py").read(); ns = {"__file__": f"{HERE}/v1_i1_eval.py"}
exec(src[:src.index("for t in (sys.argv")], ns)
from point_functional import Quantiles
target, TG, O, norm = ns["target"], ns["TG"], ns["O"], ns["norm"]
KS = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015)
for tag in ("chicago", "nyc"):
    city, yf, mat, nat, H = TG[tag]; Y, di, si, Tt, ndays, hourly = target(tag); n = len(di)
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); R = len(E_["jobs"]); Yd = Y[di, si]
    ld = lambda h: np.mean([norm(np.load(f"{O}/multi_{tag}_from_{mat(r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0)
    rep = lambda v: np.repeat((v / v.sum())[None], n, 0)
    M = {"Ours v1": 0.5 * ld("eh") + 0.5 * ld("eo"), "Chi et al.": np.mean([norm(np.load(f"{O}/chi_{tag}_seed{s}.npy")) for s in range(10)], 0),
         "Jobs-proportional": rep(E_["jobs"]), "Jobs + residents": rep(E_["jobs"] + E_["res"]), "Uniform": np.full((n, R), 1.0 / R)}
    print(f"\n===== {tag}: zero forecast MAE {Yd.mean():.3f}; median-forecast MAE by constant NB dispersion k (small k = cautious) =====")
    print(f"   {'method':20s} " + " ".join(f"k={k:<6}" for k in KS) + " | best k, MAE")
    Qs = {k: Quantiles(lambda x, k=k: np.full_like(x, k), 2 * float(Tt.max()) + 1, qs=(0.5,)) for k in KS}
    for name, P in M.items():
        lam = norm(P) * Tt[:, None]; r = [np.abs(Qs[k](lam, 0.5) - Yd).mean() for k in KS]; b = int(np.argmin(r))
        print(f"   {name:20s} " + " ".join(f"{x:8.3f}" for x in r) + f" | k={KS[b]}, {r[b]:.3f}")
print("\ndone")
