"""CROSS-CITY CALIBRATION LAYER (PROTOCOL_V2.md addendum 6, written before the pseudo-target runs). For a target T and every
method (v1 final, Chi et al., jobs, jobs + residents, uniform) the layer is fitted ONLY on pseudo-targets: the share helpers c of
T, predicted by the method trained on the other share helpers of T (never on c's own or T's taxi data), scale-matched, and scored
with the true slot totals of c (allocation error only). Step 1: shrinkage of the shares towards a smoother prior (uniform or
jobs + residents, weight w in {0,.1,.2,.3,.5}) chosen by the pseudo-target RMSE. Step 2: the constant negative binomial dispersion
k (grid 0.02 x 2^(j/2), j = 0..19) chosen by the pseudo-target NB log-likelihood. Forecasts for T: mean = total x shrunk share (RMSE);
median of NB(mean, k) (MAE). Usage (from improve/): python calibrate.py dev   (T = chicago, nyc)"""
import os, sys, numpy as np
from scipy.special import gammaln
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/v1_i1_eval.py").read(); ns = {"__file__": f"{HERE}/v1_i1_eval.py"}
exec(src[:src.index("for t in (sys.argv")], ns)
from point_functional import Quantiles
target, TG, O, MC, norm, helper, dispersion, boot, YP = (ns[k] for k in ("target", "TG", "O", "MC", "norm", "helper", "dispersion", "boot", "YP"))
from slots import calendar_slots
from datetime import date
START = {"chicago": date(2021, 1, 1), "nyc": date(2021, 1, 1), "sf": date(2023, 1, 1)}
OUT, AG, B = f"{HERE}/out", f"{HERE}/data/agg", f"{HERE}/../bench77/data"
KGRID = [0.02 * 2 ** (j / 2) for j in range(20)]; WGRID = [0.0, 0.1, 0.2, 0.3, 0.5]
NRUN = 5
# pseudo-targets per target T: tag of the pseudo-target regions (partition 1), helper sources for run r (partition r), native city for Chi et al.
PSEUDO = {"chicago": [dict(c="nyc", tag="nyc_for_chicago_p1", srcs=lambda r: f"sf_for_chicago_p{r}", chi_help="sf"),
                      dict(c="sf", tag="sf_for_chicago_p1", srcs=lambda r: f"nyc_for_chicago_p{r}", chi_help="nyc")],
          "nyc": [dict(c="chicago", tag="chicago", srcs=lambda r: f"sf_for_chicago_p{r}", chi_help="sf"),
                  dict(c="sf", tag="sf_for_nyc_p1", srcs=lambda r: "chicago", chi_help="chicago")]}
METHODS = ["Ours v1", "Chi et al.", "Jobs-proportional", "Jobs + residents", "Uniform"]


def load_Y(tag):
    if tag in ("chicago", "nyc", "sf", "dc"): return np.load(YP[tag]).astype(np.float64)
    return np.load(f"{AG}/{tag}_Y.npy").astype(np.float64)


def pseudo_samples(T, method):
    out = []
    for p in PSEUDO[T]:
        if method == "Chi et al.":                                         # native scale, as in the baseline's own protocol
            tag = p["c"]; Y = load_Y(tag)
        else:
            tag = p["tag"]; Y = load_Y(tag)
        tot = Y.sum(2); idx = [(d, s) for d in range(Y.shape[0]) for s in range(48) if tot[d, s] > 0]
        di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yd = Y[di, si]; Tt = tot[di, si]; R = Yd.shape[1]
        E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); jr = E_["jobs"] + E_["res"]
        if method == "Ours v1":
            ld = lambda h: np.mean([norm(np.load(f"{OUT}/multi_{p['tag']}_from_{p['srcs'](r)}_v{h}_seed{r - 1}.npy")) for r in range(1, NRUN + 1)], 0)
            share = 0.5 * ld("eh") + 0.5 * ld("eo")
        elif method == "Chi et al.":                                       # saved on calendar slots; keep the observed ones (SF has 2 empty slots)
            cal = calendar_slots(p["c"], START[p["c"]]); pos = {x: i for i, x in enumerate(cal)}; rows = [pos[x] for x in idx]
            share = np.mean([norm(np.load(f"{O}/chi_{p['c']}_pseudoT_{T}_seed{s}.npy"))[rows] for s in range(NRUN)], 0)
        elif method == "Jobs-proportional": share = np.repeat((E_["jobs"] / E_["jobs"].sum())[None], len(di), 0)
        elif method == "Jobs + residents": share = np.repeat((jr / jr.sum())[None], len(di), 0)
        else: share = np.full((len(di), R), 1.0 / R)
        assert share.shape == Yd.shape, (T, method, tag, share.shape, Yd.shape)
        out.append(dict(Y=Yd, Tt=Tt, share=share, uni=np.full(R, 1.0 / R), jr=jr / jr.sum(), name=tag))
    return out


def nb_ll(y, lam, k): return np.mean(gammaln(y + k) - gammaln(k) + k * np.log(k / (k + lam)) + y * np.log(lam / (k + lam)))


def fit(samples):
    """Step 1 (w, prior) by pooled pseudo-target RMSE, step 2 (k) by NB log-likelihood on the shrunk means."""
    res = {}
    for pn in ("uniform", "jobs+res"):
        for w in WGRID:
            if w == 0 and pn == "jobs+res": continue
            rm = [np.sqrt(np.mean((s["Tt"][:, None] * ((1 - w) * s["share"] + w * (s["uni"] if pn == "uniform" else s["jr"])[None]) - s["Y"]) ** 2)) for s in samples]
            res[(pn, w)] = float(np.mean(rm))
    pn, w = min(res, key=lambda kk: (res[kk], kk[1]))
    def best_k(pn_, w_):
        ll = [np.mean([nb_ll(s["Y"], np.maximum(s["Tt"][:, None] * ((1 - w_) * s["share"] + w_ * (s["uni"] if pn_ == "uniform" else s["jr"])[None]), 1e-9), k) for s in samples]) for k in KGRID]
        return KGRID[int(np.argmax(ll))]
    return dict(pn=pn, w=w, k_full=best_k(pn, w), k_disp=best_k("uniform", 0.0), rmse_pseudo=res)


def evaluate_dev(T):
    city, yf, mat, nat, H = TG[T]; Y, di, si, Tt, ndays, hourly = target(T); n = len(di); Yd = Y[di, si]
    E_ = np.load(f"{HERE}/data/extra_{T}.npz"); R = len(E_["jobs"]); jr = E_["jobs"] + E_["res"]
    ld = lambda h: np.mean([norm(np.load(f"{O}/multi_{T}_from_{mat(r)}_v{h}_seed{r - 1}.npy")) for r in range(1, 11)], 0)
    shares = {"Ours v1": 0.5 * ld("eh") + 0.5 * ld("eo"), "Chi et al.": np.mean([norm(np.load(f"{O}/chi_{T}_seed{s}.npy")) for s in range(10)], 0),
              "Jobs-proportional": np.repeat((E_["jobs"] / E_["jobs"].sum())[None], n, 0), "Jobs + residents": np.repeat((jr / jr.sum())[None], n, 0),
              "Uniform": np.full((n, R), 1.0 / R)}
    k_of, tab = dispersion([helper(h) for h in H]); Q1 = Quantiles(k_of, 2 * float(Tt.max()) + 1, qs=(0.5,))
    QK = {k: Quantiles(lambda x, k=k: np.full_like(x, k), 2 * float(Tt.max()) + 1, qs=(0.5,)) for k in KGRID}
    print(f"\n===== {T}: {R} regions, {n} calendar slots; zero forecast MAE {Yd.mean():.3f} =====")
    print(f"   {'method':18s} | uncalibrated: mean MAE / RMSE, I1 median MAE | calibration (prior, w, k_full, k_disp) | full: RMSE, mean MAE, median MAE | disp-only (w = 0): RMSE, median MAE")
    daily = {}; out = {}
    for m in METHODS:
        s = shares[m]; cal = fit(pseudo_samples(T, m)); lam0 = norm(s) * Tt[:, None]
        mae0, rmse0 = np.abs(lam0 - Yd).mean(), np.sqrt(((lam0 - Yd) ** 2).mean()); i1 = np.abs(Q1(lam0, 0.5) - Yd).mean()
        pi = np.full(R, 1.0 / R) if cal["pn"] == "uniform" else jr / jr.sum()
        lamf = Tt[:, None] * ((1 - cal["w"]) * norm(s) + cal["w"] * pi[None]); medf = QK[cal["k_full"]](lamf, 0.5); med_d = QK[cal["k_disp"]](lam0, 0.5)
        out[m] = dict(rmse0=rmse0, i1=i1, rmse_full=np.sqrt(((lamf - Yd) ** 2).mean()), mae_full=np.abs(lamf - Yd).mean(), med_full=np.abs(medf - Yd).mean(),
                      rmse_disp=rmse0, med_disp=np.abs(med_d - Yd).mean(), cal=cal)
        dd = lambda E: np.bincount(di, weights=E.mean(1), minlength=Y.shape[0]) / np.maximum(np.bincount(di, minlength=Y.shape[0]), 1)
        daily[m] = dict(med_full=dd(np.abs(medf - Yd)), med_disp=dd(np.abs(med_d - Yd)), mse_full=dd((lamf - Yd) ** 2), mse0=dd((lam0 - Yd) ** 2))
        print(f"   {m:18s} | {mae0:7.3f} / {rmse0:7.3f}, {i1:7.3f}             | {cal['pn']:8s} w={cal['w']:.1f} k_full={cal['k_full']:.3f} k_disp={cal['k_disp']:.3f} "
              f"| {out[m]['rmse_full']:7.3f}, {out[m]['mae_full']:7.3f}, {out[m]['med_full']:7.3f} | {rmse0:7.3f}, {out[m]['med_disp']:7.3f}")
    daily["zero"] = dict(med_full=np.bincount(di, weights=Yd.mean(1), minlength=Y.shape[0]) / np.maximum(np.bincount(di, minlength=Y.shape[0]), 1))
    return out, daily


if __name__ == "__main__":
    which = {"dev": ["chicago", "nyc"]}[sys.argv[1]]
    R = {T: evaluate_dev(T) for T in which}
    print("\n===== PRE-REGISTERED ADOPTION RULES (addendum 6) =====")
    rr = {T: R[T][0]["Ours v1"] for T in which}
    r_ok = all(rr[T]["rmse_full"] <= rr[T]["rmse0"] + 1e-9 for T in which) and any(rr[T]["rmse_full"] <= 0.97 * rr[T]["rmse0"] for T in which)
    print(f"   (R) shrinkage adopted: {r_ok}  (ours RMSE full vs v1: " + ", ".join(f"{T} {rr[T]['rmse_full']:.3f} vs {rr[T]['rmse0']:.3f}" for T in which) + ")")
    key = "med_full" if r_ok else "med_disp"
    m_ok = all(rr[T][key] <= 0.97 * rr[T]["i1"] for T in which)
    print(f"   (M) calibrated median-forecast MAE <= 0.97 x I1 in both cities ({'full' if r_ok else 'dispersion only'}): {m_ok}  ("
          + ", ".join(f"{T} {rr[T][key]:.3f} vs {rr[T]['i1']:.3f}" for T in which) + ")")
    rng = np.random.default_rng(0)
    for T in which:
        out, daily = R[T]; ours = daily["Ours v1"][key]; base = [m for m in METHODS if m != "Ours v1"]; bm = min(base, key=lambda m: np.nanmean(daily[m][key]))
        for ref, v in (("zero forecast", daily["zero"]["med_full"]), (bm, daily[bm][key])):
            ok = ~np.isnan(v - ours); d = (v - ours)[ok]; b = boot(d, rng)
            print(f"   {T}: {ref} minus ours (calibrated median-forecast daily MAE): {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], ours better on {100 * np.mean(d > 0):.0f}% of days")
    print(f"   DECISION: MAE component {'ADOPTED' if m_ok else 'not adopted'}; RMSE shrinkage {'ADOPTED' if r_ok else 'not adopted'}")
