"""PROTOCOL_V19.md: scale-aware neural forecaster N2 (scale max(1, mean), multiplicative augmentation, normalised loss). Usage (from improve/, .venv310): python nn19.py dev"""
import os, sys, time, numpy as np, torch, torch.nn as nn
mode = sys.argv[1] if len(sys.argv) > 1 else ""
import nn18 as N
X, H, CTX = N.X, N.H, N.CTX


class Data2(N.Data):
    def batch(self, O, Rr, aug=None):
        idx = torch.from_numpy(O[:, None] - CTX + np.arange(CTX)[None]); r_ = torch.from_numpy(Rr)[:, None]; ctx = self.Y[idx, r_]
        t = torch.from_numpy(O[:, None] + np.arange(H)[None]); base = torch.stack([self.Y[t - 48 * i, r_] for i in range(1, 8)], 0).mean(0)
        y = self.Y[t, r_] if t.max() < self.Y.shape[0] else torch.zeros(len(O), H)
        if aug is not None: a = torch.from_numpy(aug.astype(np.float32))[:, None]; ctx, base, y = ctx * a, base * a, y * a
        s = ctx.mean(1).clamp(min=1.); l6, p6 = ctx[:, -6:].mean(1), ctx[:, -12:-6].mean(1)
        cf = torch.from_numpy(N.cal_feats(self.P, O))
        x = torch.cat([(ctx / s[:, None]).clamp(max=30.), torch.log(s)[:, None], (l6 / s)[:, None], ((l6 - p6) / s)[:, None], cf], 1); return x, base, s, y


def sample(hs, pl, n, rng, aug=False):
    w = np.array([len(O) * hs[k].R for k, O in pl], float); w /= w.sum(); hk = rng.choice(len(pl), n, p=w); xs, bs, ss, ys = [], [], [], []
    for i, (k, O) in enumerate(pl):
        m = int((hk == i).sum())
        if m == 0: continue
        a = np.exp(rng.uniform(np.log(.5), np.log(2.), m)) if aug else None
        x, b, s, y = hs[k].batch(rng.choice(O, m), rng.integers(0, hs[k].R, m), a); xs.append(x); bs.append(b); ss.append(s); ys.append(y)
    p = torch.randperm(sum(len(a) for a in xs)); return [torch.cat(a)[p] for a in (xs, bs, ss, ys)]


def helpers2(names, cutoff):
    hs = []
    for nm in names:
        Y, Xs, st = X.load_series(nm); last = Y.shape[0] // 48 if nm.startswith("sf") else cutoff; hs.append(Data2(Y[:last * 48], st)); hs[-1].last = last
    return hs


def train(T, r, cutoff):
    torch.manual_seed(r - 1); rng = np.random.default_rng(r - 1); hs = helpers2(X.HEL[T]["matched"](r), cutoff); tr, va = N.pool(hs, False), N.pool(hs, True)
    xv, bv, sv, yv = sample(hs, va, 100000, np.random.default_rng(99)); net = N.Net(xv.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    mae_base = (bv - yv).abs().mean().item(); best, bad, bst = 1e9, 0, None; t0 = time.time()
    for ep in range(12):
        net.train(); x, b, s, y = sample(hs, tr, 1000000, rng, True)
        for i in range(0, len(x), 2048):
            sl = slice(i, i + 2048); opt.zero_grad(); z = (net(x[sl], b[sl], s[sl]) - y[sl]) / s[sl][:, None]
            l = nn.functional.huber_loss(z, torch.zeros_like(z), delta=1.0) + 0.05 * z.abs().mean(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = (net(xv, bv, sv) - yv).abs().mean().item()
        print(f"   {T} r={r} epoch {ep + 1}: helper validation MAE {v:.4f} (weekly average {mae_base:.4f}) {time.time() - t0:.0f}s", flush=True)
        if ep == 0 and v >= mae_base: print("   KILL CHECK FAILED"); sys.exit(1)
        if v < best - 1e-5: best, bad, bst = v, 0, {k: a.clone() for k, a in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    net.load_state_dict(bst); net.eval(); return net


def run(T, r, O):
    f = f"{X.OUT}/pred_dev_N2_{T}_matched_r{r}.npy"
    if os.path.exists(f): return np.load(f)
    net = train(T, r, X.DEV[0]); Y, Xs, st = X.load_series(T); d = Data2(Y, st); R = Y.shape[1]; Oo = np.repeat(O, R); Rr = np.tile(np.arange(R), len(O)); out = []
    with torch.no_grad():
        for i in range(0, len(Oo), 20000):
            x, b, s, _ = d.batch(Oo[i:i + 20000], Rr[i:i + 20000]); out.append(net(x, b, s).numpy())
    p = np.concatenate(out).reshape(len(O), R, H).transpose(0, 2, 1).astype(np.float32); np.save(f, p); return p


if mode == "dev":
    O = X.grid(*X.DEV, True)
    for T in ("chicago", "nyc"):
        for r in (1, 2): run(T, r, O)
    print("\n===== N2, B2 (N2 + C5), B3 (N1 + N2 + C5) on October and November 2021, MAE / RMSE =====")
    L = lambda c, T: np.mean([np.load(f"{X.OUT}/pred_dev_{c}_{T}_matched_r{r}.npy") for r in (1, 2)], 0); rat = {k: [] for k in ("N2", "B2", "B3")}
    for T in ("chicago", "nyc"):
        n1, n2, c5 = L("N1", T), L("N2", T), L("C5", T); c2 = X.cmp_chronos(T, O, L("C2", T), "dev"); q1 = X.cmp_chronos(T, O, n1, "dev")
        for nm, p in (("N2", n2), ("B2", .5 * (n2 + c5)), ("B3", (n1 + n2 + c5) / 3)):
            q = X.cmp_chronos(T, O, p, "dev"); rat[nm].append((q["ours"][0] / c2["ours"][0], q["ours"][1] / c2["ours"][1]))
            print(f"{T} {nm}: {q['ours'][0]:.3f} / {q['ours'][1]:.3f} | N1 {q1['ours'][0]:.3f} / {q1['ours'][1]:.3f} | C2 {c2['ours'][0]:.3f} / {c2['ours'][1]:.3f} | Chronos {q['chronos'][0]:.3f} / {q['chronos'][1]:.3f} | ratio to C2 MAE {rat[nm][-1][0]:.3f} RMSE {rat[nm][-1][1]:.3f}"
                  f" | ratio to Chronos MAE {q['ours'][0] / q['chronos'][0]:.3f} RMSE {q['ours'][1] / q['chronos'][1]:.3f} | daily diff {q['diff'][0]:+.4f} [{q['diff'][1]:+.4f}, {q['diff'][2]:+.4f}]")
    for nm, v in rat.items():
        ok = np.mean([a for a, b in v]) <= 0.97 and max(a for a, b in v) <= 1.01 and max(b for a, b in v) <= 1.02; print(f"screen of PROTOCOL_V19 for {nm}: {'PASSED' if ok else 'NOT passed'}")
    print("screen done")
