"""PREDICTION step (PROTOCOL_REVISION.md, part A). Builds the strictly zero-shot trip predictions of every method for a
target city on its calendar slots. The demand file of the target city is never opened here: the level, the rhythm and
the shares use helper cities, public covariates and the calendar only.
Output: out_cal/pred_<T>.npz with slots (d, s), the predicted city total per slot and one trip matrix per method.
Usage: python predict_final.py chicago nyc sf dc"""
import os, sys, json, numpy as np
from datetime import date, timedelta
from slots import calendar_slots
HERE = os.path.dirname(os.path.abspath(__file__)); B, M = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data"
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy",
      "sf": f"{M}/sf_2023_tract_30min.npy", "dc": f"{M}/dc_2021_tract_30min.npy"}
START = {"chicago": date(2021, 1, 1), "nyc": date(2021, 1, 1), "sf": date(2023, 1, 1), "dc": date(2021, 1, 1)}
SHARE = {"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"], "sf": ["chicago", "nyc"], "dc": ["chicago", "nyc", "sf"]}
MODEL_H = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}",
           "sf": lambda r: "chicago+nyc", "dc": lambda r: f"chicago+nyc+sf_for_dc_p{r}"}
G = {"chicago": "16000US1714000", "nyc": "16000US3651000", "sf": "16000US0667000", "dc": "16000US1150000"}
acs = json.load(open(f"{HERE}/raw/acs_city.json"))["data"]
U = {}
for c, g in G.items():
    a = acs[g]; e = np.load(f"{HERE}/data/extra_{c}.npz")
    U[c] = {"jobs": e["jobs"].sum(), "residents": e["res"].sum(), "population": a["B01003"]["estimate"]["B01003001"],
            "taxi commuters": a["B08301"]["estimate"]["B08301016"], "transit commuters": a["B08301"]["estimate"]["B08301010"],
            "car-free households": a["B25044"]["estimate"]["B25044003"] + a["B25044"]["estimate"]["B25044010"]}


def days(c):
    n = (date(START[c].year + 1, 1, 1) - START[c]).days; return [START[c] + timedelta(days=d) for d in range(n)]


def predict(T):
    helpers = [c for c in YP if c != T]
    Y = {c: np.load(YP[c]).astype(np.float64) for c in helpers}            # helper demand only
    N = {c: Y[c].sum() for c in helpers}
    DOW = {c: np.array([d.weekday() for d in days(c)]) for c in YP}; MON = {c: np.array([d.month for d in days(c)]) for c in YP}
    # LEVEL: unit chosen by leave-one-helper-out over the helpers, then the log-pooled ratio estimator
    loo = {u: np.mean([abs(np.exp(np.mean([np.log(N[c] / U[c][u]) for c in helpers if c != h])) * U[h][u] / N[h] - 1) for h in helpers]) for u in U[T]}
    unit = min(loo, key=loo.get); lev = U[T][unit] * np.exp(np.mean([np.log(N[c] / U[c][unit]) for c in helpers]))
    # RHYTHM: mean helper weekday x slot profile and a month factor from same-year share helpers
    sl = calendar_slots(T, START[T]); di, si = np.array([a for a, _ in sl]), np.array([b for _, b in sl])
    def profile(c):
        tot = Y[c].sum(2); r = np.array([tot[DOW[c] == k].sum(0) for k in range(7)]); return r / r.sum()
    S = SHARE[T]; rh = np.mean([profile(c) for c in S], 0); nd = np.bincount(DOW[T], minlength=7); w = DOW[T][di]
    Tt = lev * rh[w, si] / nd[w]
    same = [c for c in S if START[c].year == START[T].year]
    if same:
        mf = np.mean([(lambda tc: np.array([tc[MON[c] == k].mean() for k in range(1, 13)]))(Y[c].sum((1, 2))) for c in same], 0); mf /= mf.mean()
        Tt = Tt * mf[MON[T][di] - 1] / mf[MON[T] - 1].mean()
    Tt = Tt / Tt.sum() * lev
    # SHARES: saved allocator predictions on the same calendar slots
    def ens(fs):
        P = [np.load(f) for f in fs]
        assert all(p.shape[0] == len(sl) for p in P), f"prediction rows differ from the calendar slots: {fs[0]}"
        p = np.mean(P, 0); return p / p.sum(1, keepdims=True)
    E_ = np.load(f"{HERE}/data/extra_{T}.npz"); R = len(E_["jobs"]); n = len(sl); jr = E_["jobs"] + E_["res"]
    sh = {"uniform": np.full((n, R), 1 / R), "jobs": np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0),
          "jobs_res": np.repeat((jr / jr.sum())[None], n, 0)}
    if T in ("sf", "dc"):
        base = [f"{HERE}/../multicity/out_cal/multi_{T}_from_{'+'.join(S)}_seed{s}.npy" for s in range(10)]
        heads = {h: [f"{HERE}/out_cal/multi_{T}_from_{MODEL_H[T](r)}_v{h}_seed{r-1}.npy" for r in range(1, 11)] for h in ("eh", "eo")}
    else:
        # Chicago and NYC: the earlier runs already use exactly the calendar slot set (checked by the row count in ens)
        base = [f"{HERE}/../multicity/multi_{T}_from_{'+'.join(S)}_seed{s}.npy" for s in range(10)]
        heads = {h: [f"{HERE}/out/multi_{T}_from_{MODEL_H[T](r)}_v{h}_seed{r-1}.npy" for r in range(1, 11)] for h in ("eh", "eo")}
    sh["base"] = ens(base); sh["eh"] = ens(heads["eh"]); sh["eo"] = ens(heads["eo"]); sh["final"] = 0.5 * sh["eh"] + 0.5 * sh["eo"]
    runs = np.stack([0.5 * ens([a]) + 0.5 * ens([b]) for a, b in zip(heads["eh"], heads["eo"])])    # single run r = eh run r + eo run r
    np.savez_compressed(f"{HERE}/out_cal/pred_{T}.npz", di=di, si=si, total=Tt, unit=unit, level=lev,
                        **{m: (p * Tt[:, None]).astype(np.float32) for m, p in sh.items()}, runs=(runs * Tt[None, :, None]).astype(np.float32))
    print(f"{T}: {n} calendar slots, {R} regions, level unit '{unit}', predicted yearly total {lev:,.0f}; methods {list(sh)} + 10 single runs", flush=True)


if __name__ == "__main__":
    os.makedirs(f"{HERE}/out_cal", exist_ok=True)
    for T in sys.argv[1:]: predict(T)
