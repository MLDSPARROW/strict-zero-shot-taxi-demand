"""PROTOCOL_V17.md step 2: candidate C5 = C2 of xc7d.py plus a richer description of the 336-slot context (all from counts before the origin).
Usage (from improve/, .venv310 interpreter): python xc17.py leak | python xc17.py dev"""
import os, sys, numpy as np
mode = sys.argv[1] if len(sys.argv) > 1 else ""; sys.argv = ["x"]
import xc7d as X
_prep, _feat = X.prep, X.features
CHUNK = 8


def prep(Y, st):
    P = _prep(Y, st); z = np.zeros((1, Y.shape[1]))
    P["C2"] = np.concatenate([z, np.cumsum(Y.astype(np.float64) ** 2, 0)]); P["CZ"] = np.concatenate([z, np.cumsum(Y == 0, 0, dtype=np.float64)]); return P


def extra(P, O):
    """(n_o, R, k) extra features of the origins O, then repeated per step j by the caller"""
    Y, C, C2, CZ = P["Y"], P["C"], P["C2"], P["CZ"]; O = np.asarray(O)
    wm = (C[O] - C[O - X.CTX]) / X.CTX; var = np.maximum((C2[O] - C2[O - X.CTX]) / X.CTX - wm ** 2, 0); sd = np.sqrt(var); zf = (CZ[O] - CZ[O - X.CTX]) / X.CTX
    m = lambda k: (C[O] - C[O - k]) / k
    ch = [np.log((C[O - X.CTX + (i + 1) * CHUNK] - C[O - X.CTX + i * CHUNK]) / CHUNK + 0.1) - np.log(wm + 0.1) for i in range(X.CTX // CHUNK)]
    ls = np.log((m(12) + 0.1) / ((C[O - 12] - C[O - 60]) / 48 + 0.1))
    return np.stack(ch + [sd, zf, np.log1p(m(24)), np.log1p(m(48)), np.log1p(m(12)), ls], -1)             # (n_o, R, 42 + 6)


def features(P, Xs, O, static=True):
    F, M, y, W = _feat(P, Xs, O, static); O = np.asarray(O); no, R = len(O), P["Y"].shape[1]; E = extra(P, O); Fs = []
    for j in range(X.HOR):
        t = O + j; daily = np.stack([np.log1p(P["Y"][t - 48 * i]) for i in range(1, 8)], -1)            # the same slot on each of the previous 7 days (t - 48 i < O)
        lsj = E[:, :, -1:] * float(j)
        Fs.append(np.concatenate([E, daily, lsj], -1).reshape(no * R, -1).astype(np.float32))
    return np.concatenate([F, np.concatenate(Fs)], 1), M, y, W


X.prep, X.features = prep, features
X.CAND["C5"] = dict(depth=6, static=False)
if __name__ == "__main__" and mode == "leak":
    from datetime import date
    rng = np.random.default_rng(1); st = date(2021, 1, 1); Y = rng.poisson(2.0, (60 * 48, 7)).astype(np.float32); P = prep(Y, st); O = np.array([20 * 48 + 12, 30 * 48 + 6, 45 * 48 + 24]); Xs = np.zeros((7, 1)); F0 = features(P, Xs, O, False)[0]; worst = 0.0
    for k, o in enumerate(O):
        Y2 = Y.copy(); Y2[o:] = rng.poisson(50, Y2[o:].shape); F1 = features(prep(Y2, st), Xs, O[k:k + 1], False)[0]
        rows = np.concatenate([np.arange(j * len(O) * 7 + k * 7, j * len(O) * 7 + (k + 1) * 7) for j in range(X.HOR)]); worst = max(worst, float(np.abs(F0[rows] - F1).max()))
    print(f"leak test C5: largest change of any feature when all counts from the origin on are replaced: {worst:.2e}; {F0.shape[1]} features")
elif __name__ == "__main__" and mode == "dev":
    O = X.grid(*X.DEV, True)
    for T in ("chicago", "nyc"):
        for r in (1, 2): X.run_one(T, "matched", r, "C5", "dev", X.DEV[0], O)
    print("\n===== C5 on October and November 2021 (MAE / RMSE of the mean forecast) =====")
    rat, rr = [], []
    for T in ("chicago", "nyc"):
        q = X.cmp_chronos(T, O, np.mean([np.load(f"{X.OUT}/pred_dev_C5_{T}_matched_r{r}.npy") for r in (1, 2)], 0), "dev")
        c2 = X.cmp_chronos(T, O, np.mean([np.load(f"{X.OUT}/pred_dev_C2_{T}_matched_r{r}.npy") for r in (1, 2)], 0), "dev")
        print(f"{T}: C5 {q['ours'][0]:.3f} / {q['ours'][1]:.3f} | C2 {c2['ours'][0]:.3f} / {c2['ours'][1]:.3f} | Chronos-Bolt {q['chronos'][0]:.3f} / {q['chronos'][1]:.3f}"
              f" | C5/C2 MAE {q['ours'][0] / c2['ours'][0]:.3f}, RMSE {q['ours'][1] / c2['ours'][1]:.3f} | C5/Chronos MAE {q['ours'][0] / q['chronos'][0]:.3f}, RMSE {q['ours'][1] / q['chronos'][1]:.3f}"
              f" | daily diff to Chronos {q['diff'][0]:+.4f} [{q['diff'][1]:+.4f}, {q['diff'][2]:+.4f}]")
        rat.append((q['ours'][0] / c2['ours'][0], q['ours'][1] / c2['ours'][1]))
    scr = np.mean([a for a, b in rat]) <= 0.98 and max(a for a, b in rat) <= 1.01 and max(b for a, b in rat) <= 1.02
    print(f"screen of PROTOCOL_V17 (mean MAE ratio to C2 <= 0.98, no city above 1.01, no RMSE ratio above 1.02): {'PASSED' if scr else 'NOT passed'}")
