"""FULLY STRICT v3 (declared 2026-10-01): the LEVEL unit is chosen per target WITHOUT the target: among the helper taxi cities
(the 4 taxi cities minus the target), run leave-one-city-out and pick the unit with the lowest mean |error|. Then
LEVEL = target units x geometric-mean helper trips-per-unit (helpers = all taxi cities except the target, incl. DC for LEVEL only).
RHYTHM v2 (same-year month factor), SHARE = 10-seed ensembles."""
import json, numpy as np, importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location("v2", "strict_v2.py"); v2 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(v2)
st = v2.st
spec9 = importlib.util.spec_from_file_location("t9", "tests_round9.py"); src = open("tests_round9.py").read()
ns = {}; exec(src[src.index('acs = json.load'):src.index('for cities, label in')], {"json": json, "np": np, "B": "../bench77/data", "M": "../multicity/data"}, ns)
U, trips = ns["U"], ns["trips"]
TAXI = ["chicago", "nyc", "sf", "dc"]
for tgt, (srcs, S) in st.CFG.items():
    H = [c for c in TAXI if c != tgt]
    loo = {u: np.mean([abs(np.exp(np.mean([np.log(trips[c] / U[c][u]) for c in H if c != h])) * U[h][u] / trips[h] - 1) for h in H]) for u in U[tgt]}
    unit = min(loo, key=loo.get)
    Yt = st.Y[tgt]; tot = Yt.sum(2)
    idx = [(a, b) for a in range(Yt.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Yt[di, si]
    T2 = v2.strict_T(tgt, S, di, si, 2)                                   # car-free level (v2)
    lev3 = U[tgt][unit] * np.exp(np.mean([np.log(trips[c] / U[c][unit]) for c in H])); T3 = T2 / T2.sum() * lev3
    print(f"=== {tgt.upper()}: helper-only leave-one-out picks '{unit}' (" + ", ".join(f"{u} {100*v:.0f}%" for u, v in sorted(loo.items(), key=lambda x: x[1])[:3])
          + f") -> predicted yearly total {100*(lev3/Yt.sum()-1):+.0f}% (car-free v2: {100*(T2.sum()/Yt.sum()-1):+.0f}%)")
    for n in ["eh", "eho", "eo"]:
        p = np.mean([np.load(f"out/multi_{tgt}_from_{srcs}_v{n}_seed{s}.npy") for s in range(10)], 0); p /= p.sum(1, keepdims=True)
        r = []
        for T in (T2, T3):
            E = p * T[:, None] - Yc; A = np.abs(E); m = Yc >= 5; r.append(f"{A.mean():.2f} / {np.sqrt((E**2).mean()):.2f} / {100*np.mean(A[m]/Yc[m]):.1f}%")
        print(f"   {n:4s} v2 (car-free): {r[0]:22s} | v3 ({unit}): {r[1]}")
