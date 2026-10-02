"""FULLY STRICT (v3) trip errors per seed for any variant/helper set: python strict_seeds.py <helpers-suffix> <variant> <seeds>"""
import sys, numpy as np, io, contextlib
src = open("strict_v3.py").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(src[:src.index("for tgt, (srcs, S) in st.CFG.items():")], ns)
st, v2, U, trips, TAXI = ns["st"], ns["v2"], ns["U"], ns["trips"], ns["TAXI"]
def strict_scores(tgt, files):
    srcs, S = st.CFG[tgt]; H = [c for c in TAXI if c != tgt]
    loo = {u: np.mean([abs(np.exp(np.mean([np.log(trips[c] / U[c][u]) for c in H if c != h])) * U[h][u] / trips[h] - 1) for h in H]) for u in U[tgt]}
    unit = min(loo, key=loo.get)
    Yt = st.Y[tgt]; tot = Yt.sum(2); idx = [(a, b) for a in range(Yt.shape[0]) for b in range(48) if tot[a, b] > 0]
    di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx]); Yc = Yt[di, si]
    T = v2.strict_T(tgt, S, di, si, 2); T = T / T.sum() * U[tgt][unit] * np.exp(np.mean([np.log(trips[c] / U[c][unit]) for c in H]))
    out = []
    for f in files:
        p = np.load(f) if isinstance(f, str) else f; p = p / p.sum(1, keepdims=True)
        E = p * T[:, None] - Yc; A = np.abs(E); m = Yc >= 5; out.append((A.mean(), np.sqrt((E ** 2).mean()), 100 * np.mean(A[m] / Yc[m])))
    return out
if __name__ == "__main__":
    for tgt, base in [("chicago", "nyc+sf"), ("nyc", "chicago+sf")]:
        for label, helpers, fl in [("without Austin (eh)", base, "eh"), ("with Austin (eh)", base + "+austin", "eh")]:
            fs = []
            for s in range(3):
                f = f"out/multi_{tgt}_from_{helpers}_v{fl}_seed{s}.npy"
                try: np.load(f); fs.append(f)
                except FileNotFoundError: pass
            sc = strict_scores(tgt, fs)
            print(f"{tgt:8s} {label:22s} " + " | ".join(f"run {k+1}: {a:.2f} / {b:.2f} / {c:.1f}%" for k, (a, b, c) in enumerate(sc)))
