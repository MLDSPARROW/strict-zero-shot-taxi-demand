"""EVALUATION step (PROTOCOL_REVISION.md, part A). The only script that opens the demand of the target city. It scores the
saved predictions of predict_final.py on every calendar slot, including slots without any recorded trip.
Reports MAE and RMSE (Washington, DC hourly), the 95% interval of the final method from a 7-day block bootstrap over days
(2,000 draws), paired daily differences to each baseline with their block-bootstrap intervals, and the mean and standard
deviation over the 10 single runs. Usage: python evaluate_strict.py chicago nyc sf dc"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); B, M = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data"
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy",
      "sf": f"{M}/sf_2023_tract_30min.npy", "dc": f"{M}/dc_2021_tract_30min.npy"}
NAMES = {"uniform": "Uniform", "jobs": "Jobs-proportional", "jobs_res": "Jobs + residents", "base": "Base allocator",
         "eh": "Head eh", "eo": "Head eo", "final": "Final method"}


def boot(x, rng, f=np.mean):
    nb = len(x) // 7
    return [f(x[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(x) - 7, nb)])]) for _ in range(2000)]


def evaluate(T):
    rng = np.random.default_rng(0); P = np.load(f"{HERE}/out_cal/pred_{T}.npz"); di, si = P["di"], P["si"]
    Y = np.load(YP[T]).astype(np.float64); nd = Y.shape[0]; hourly = T == "dc"
    empty = int((Y[di, si].sum(1) == 0).sum())
    def errors(pred):                       # per-day absolute and squared errors over the existing slots of each day
        F = np.full(Y.shape, np.nan); F[di, si] = pred; E = F - Y
        if hourly: E = E.reshape(nd, 24, 2, -1).sum(2)
        E = E.reshape(nd, -1); ok = ~np.isnan(E)
        return np.nanmean(np.abs(E), 1), np.nanmean(E ** 2, 1), np.abs(E[ok]).mean(), np.sqrt((E[ok] ** 2).mean())
    print(f"\n===== {T.upper()} ({'hourly' if hourly else '30-min'}): {len(di)} calendar slots, {empty} of them without any recorded trip =====")
    res = {m: errors(P[m]) for m in NAMES}
    for m, nm in NAMES.items(): print(f"   {nm:20s} MAE {res[m][2]:.2f}  RMSE {res[m][3]:.2f}")
    fa, fs = res["final"][0], res["final"][1]
    print(f"   Final: MAE 95% interval [{np.percentile(boot(fa, rng), 2.5):.2f}, {np.percentile(boot(fa, rng), 97.5):.2f}], "
          f"RMSE 95% interval [{np.percentile(boot(fs, rng, lambda z: np.sqrt(z.mean())), 2.5):.2f}, {np.percentile(boot(fs, rng, lambda z: np.sqrt(z.mean())), 97.5):.2f}]")
    for m in ("base", "jobs", "jobs_res"):
        out = []
        for k, lab in ((0, "MAE"), (1, "MSE")):
            d = res["final"][k] - res[m][k]; b = boot(d, rng)
            out.append(f"{lab} diff {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}], better on {100*(d<0).mean():.0f}% of days, d_z {d.mean()/d.std(ddof=1):+.2f}")
        print(f"   Final vs {NAMES[m]:18s} " + " | ".join(out))
    r = np.array([errors(P["runs"][k])[2:] for k in range(P["runs"].shape[0])])
    print(f"   10 single runs: MAE {r[:,0].mean():.2f} +- {r[:,0].std(ddof=1):.2f} (range {r[:,0].min():.2f} to {r[:,0].max():.2f}), "
          f"RMSE {r[:,1].mean():.2f} +- {r[:,1].std(ddof=1):.2f} (range {r[:,1].min():.2f} to {r[:,1].max():.2f})")


if __name__ == "__main__":
    for T in sys.argv[1:]: evaluate(T)
