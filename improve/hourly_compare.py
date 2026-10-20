"""PROTOCOL_V2.md addendum 5: the final method (v1) and the baselines scored at hourly resolution, and for Austin on K = 10 and
K = 7 zones, for comparison with published studies. No model change; predictions, city totals and slots as in v1_i1_eval.py.
Target demand is read for scoring only. Usage (from improve/): python hourly_compare.py"""
import os, sys, numpy as np
from sklearn.cluster import KMeans
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/v1_i1_eval.py").read(); ns = {"__file__": f"{HERE}/v1_i1_eval.py"}
exec(src[:src.index("for t in (sys.argv")], ns)
target, TG, O, MC, norm = ns["target"], ns["TG"], ns["O"], ns["MC"], ns["norm"]


def predictions(tag):
    city, yf, mat, nat, H = TG[tag]; Y, di, si, Tt, ndays, hourly = target(tag); n = len(di)
    E_ = np.load(f"{HERE}/data/extra_{tag}.npz"); R = len(E_["jobs"])
    rep = lambda v: np.repeat((v / v.sum())[None], n, 0)
    ld = lambda f: np.mean([norm(np.load(f(r))) for r in range(1, 11)], 0)
    fin = lambda h: (lambda r: f"{O}/multi_{tag}_from_{mat(r)}_v{h}_seed{r - 1}.npy")
    M = {"Zero forecast": None, "Jobs-proportional": rep(E_["jobs"]), "Jobs + residents": rep(E_["jobs"] + E_["res"]),
         "Chi et al. allocation": np.mean([norm(np.load(f"{O}/chi_{tag}_seed{s}.npy")) for s in range(10)], 0),
         "Ours v1, final": 0.5 * ld(fin("eh")) + 0.5 * ld(fin("eo"))}
    return Y, di, si, Tt, M


def score(Y, di, si, Tt, p, Z=None, true_totals=False):
    L = np.full(Y.shape, np.nan); tot = Y.sum(2)[di, si] if true_totals else Tt
    L[di, si] = 0.0 if p is None else norm(p) * tot[:, None]
    T = Y
    if Z is not None: L, T = L @ Z.T, Y @ Z.T                       # rows are all-NaN or all-valid, so NaN never leaks across slots
    nd = Y.shape[0]; L, T = L.reshape(nd, 24, 2, -1).sum(2), T.reshape(nd, 24, 2, -1).sum(2)
    ok = ~np.isnan(L); e = L[ok] - T[ok]; return np.abs(e).mean(), np.sqrt((e ** 2).mean()), T[ok].mean()


def table(title, tag, Z=None, true_totals=False):
    Y, di, si, Tt, M = predictions(tag); R = Y.shape[2] if Z is None else Z.shape[0]
    print(f"\n===== {title}: {R} {'zones' if Z is not None else 'regions'}, hourly, {'TRUE slot totals (diagnostic)' if true_totals else 'fully strict'} =====")
    for name, p in M.items():
        if true_totals and name == "Zero forecast": continue
        mae, rmse, mean = score(Y, di, si, Tt, p, Z, true_totals)
        print(f"   {name:24s} MAE {mae:7.3f}  RMSE {rmse:7.3f}   (mean hourly count {mean:.3f})")


for tag in ("chicago", "nyc", "sf"): table(tag, tag)
cent = np.load(f"{MC}/region_static_austin.npz")["centroid_km"]; R = len(cent)
for K in (10, 7):
    lab = KMeans(K, n_init=10, random_state=0).fit(cent).labels_; Z = np.zeros((K, R)); Z[lab, np.arange(R)] = 1
    print(f"\nAustin K = {K}: cells per zone {sorted(Z.sum(1).astype(int).tolist(), reverse=True)}")
    table(f"austin, K = {K}", "austin", Z); table(f"austin, K = {K}", "austin", Z, true_totals=True)
print("\ndone")
