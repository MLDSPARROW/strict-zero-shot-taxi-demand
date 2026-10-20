"""EXPLORATION on the development targets Chicago and NYC only (PROTOCOL_V2.md addendum 6; no held-out city is read).
Existing fully strict v1 predictions (restored calendar runs, 10 runs per head). Two questions:
 (a) MAE: does a more dispersed predictive distribution (k scaled by s < 1, i.e. allowance for allocation error that the helper
     in-sample dispersion does not contain) lower the median-forecast MAE?
 (b) RMSE/MAE: does blending the v1 shares with a smoother public prior (uniform, jobs + residents, people prior) lower the strict
     trip-level mean-forecast RMSE and MAE?
Usage (from improve/): python explore_dev.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/v1_i1_eval.py").read(); ns = {"__file__": f"{HERE}/v1_i1_eval.py"}
exec(src[:src.index("for t in (sys.argv")], ns)
from point_functional import Quantiles
target, TG, O, norm, helper, dispersion = ns["target"], ns["TG"], ns["O"], ns["norm"], ns["helper"], ns["dispersion"]

for tag in ("chicago", "nyc"):
    city, yf, mat, nat, H = TG[tag]; Y, di, si, Tt, ndays, hourly = target(tag); n = len(di)
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); A = np.load(f"{HERE}/data/acs_{tag}.npz"); R = len(E_["jobs"])
    ld = lambda h: np.mean([norm(np.load(f"{O}/multi_{tag}_from_{mat(r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0)
    V1 = 0.5 * ld("eh") + 0.5 * ld("eo")
    prior = {"uniform": np.full(R, 1.0 / R), "jobs+res": (E_["jobs"] + E_["res"]) / (E_["jobs"] + E_["res"]).sum(),
             "people": (A["jobs_n"] / A["jobs_n"].sum() + A["pop"] / A["pop"].sum()) / 2}
    k_of, tab = dispersion([helper(h) for h in H]); Q = Quantiles(k_of, 2 * float(Tt.max()) + 1)
    Yd = Y[di, si]                                                           # observed slots x regions

    def trip(P):
        lam = norm(P) * Tt[:, None]; e = lam - Yd; return np.abs(e).mean(), np.sqrt((e ** 2).mean()), lam

    print(f"\n===== {tag}: {R} regions, {n} slots; zero forecast MAE {Yd.mean():.3f} =====")
    print("   (b) blend of v1 shares with a smoother prior, strict mean-forecast MAE / RMSE")
    for name, pr in [("v1 alone", None)] + list(prior.items()):
        for w in ([0.0] if pr is None else [0.1, 0.2, 0.3, 0.5]):
            P = V1 if pr is None else (1 - w) * V1 + w * pr[None]
            mae, rmse, _ = trip(P)
            print(f"      {name:10s} w={w:.1f}: MAE {mae:.3f}  RMSE {rmse:.3f}")
    print("   (a) median-forecast MAE of v1 when k is scaled by s (s = 1 is the helper in-sample dispersion used so far)")
    _, _, lam = trip(V1)
    for s in (1.0, 0.5, 0.25, 0.1, 0.05):
        Qs = Quantiles(lambda x, s=s: k_of(x) * s, 2 * float(Tt.max()) + 1); med = Qs(lam, 0.5)
        print(f"      s={s:<5}: median-forecast MAE {np.abs(med - Yd).mean():.3f} (share of cells forecast as zero {np.mean(med == 0):.3f}, observed zero share {np.mean(Yd == 0):.3f})")
print("\ndone")
