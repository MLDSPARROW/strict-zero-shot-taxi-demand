"""PROTOCOL_V18.md: neural forecaster N1 on the 336-slot context, trained on helper cities only; B1 = equal average of N1 and C5.
Usage (from improve/, .venv310 interpreter): python nn18.py dev"""
import os, sys, time, numpy as np, torch, torch.nn as nn
mode = sys.argv[1] if len(sys.argv) > 1 else ""; sys.argv = ["x"]
import xc7d as X
torch.set_num_threads(2); CTX, H = X.CTX, X.HOR


def cal_feats(P, O):
    d, s = O // 48, O % 48
    return np.stack([np.sin(2 * np.pi * s / 48), np.cos(2 * np.pi * s / 48), np.sin(2 * np.pi * P["wk"][d] / 7), np.cos(2 * np.pi * P["wk"][d] / 7), P["weekend"][d], P["h"][d], P["hp"][d], P["hn"][d]], -1).astype(np.float32)


class Data:
    def __init__(self, Y, st): self.P = X.prep(Y, st); self.Y = torch.from_numpy(Y.astype(np.float32)); self.R = Y.shape[1]
    def batch(self, O, Rr):
        """O, Rr: arrays (n,) of origin slots and regions -> inputs (n, 336 + 9), baseline (n, 6), scale (n,), label (n, 6)"""
        idx = torch.from_numpy(O[:, None] - CTX + np.arange(CTX)[None]); ctx = self.Y[idx, torch.from_numpy(Rr)[:, None]]
        s = ctx.mean(1) + 0.1; t = torch.from_numpy(O[:, None] + np.arange(H)[None]); r_ = torch.from_numpy(Rr)[:, None]
        base = torch.stack([self.Y[t - 48 * i, r_] for i in range(1, 8)], 0).mean(0)
        y = self.Y[t, r_] if (t.max() < self.Y.shape[0]) else torch.zeros(len(O), H)
        cf = torch.from_numpy(cal_feats(self.P, O)); x = torch.cat([(ctx / s[:, None]).clamp(max=30.), torch.log(s)[:, None], cf], 1)
        return x, base, s, y


class Net(nn.Module):
    def __init__(self, nin):
        super().__init__(); self.f = nn.Sequential(nn.Linear(nin, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, H))
    def forward(self, x, base, s): return torch.clamp(base + s[:, None] * self.f(x), min=0.)


def loss_fn(p, y): return 0.5 * nn.functional.huber_loss(p, y, delta=2.0) + 0.5 * (p - y).abs().mean()


def helpers(names, cutoff):
    hs = []
    for nm in names:
        Y, Xs, st = X.load_series(nm); last = Y.shape[0] // 48 if nm.startswith("sf") else cutoff; hs.append(Data(Y[:last * 48], st)); hs[-1].last = last
    return hs


def pool(hs, va):
    out = []
    for k, h in enumerate(hs):
        O = np.array([o for o in X.grid(8, h.last - 1, True) if o + H + 1 <= h.last * 48 and ((o // 48) % 12 == 5) == va]); out.append((k, O))
    return out


def sample(hs, pl, n, rng):
    ks = np.concatenate([[k] * len(O) for k, O in pl]); w = np.array([len(O) * hs[k].R for k, O in pl], float); w /= w.sum()
    hk = rng.choice(len(pl), n, p=w); xs, bs, ss, ys = [], [], [], []
    for i, (k, O) in enumerate(pl):
        m = int((hk == i).sum())
        if m == 0: continue
        x, b, s, y = hs[k].batch(rng.choice(O, m), rng.integers(0, hs[k].R, m)); xs.append(x); bs.append(b); ss.append(s); ys.append(y)
    p = torch.randperm(sum(len(a) for a in xs)); return [torch.cat(a)[p] for a in (xs, bs, ss, ys)]


def train(T, r, cutoff):
    torch.manual_seed(r - 1); rng = np.random.default_rng(r - 1); hs = helpers(X.HEL[T]["matched"](r), cutoff); tr, va = pool(hs, False), pool(hs, True)
    xv, bv, sv, yv = sample(hs, va, 100000, np.random.default_rng(99)); net = Net(xv.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    mae_base = (bv - yv).abs().mean().item(); best, bad, bst = 1e9, 0, None; t0 = time.time()
    for ep in range(15):
        net.train(); x, b, s, y = sample(hs, tr, 300000, rng)
        for i in range(0, len(x), 512):
            sl = slice(i, i + 512); opt.zero_grad(); l = loss_fn(net(x[sl], b[sl], s[sl]), y[sl]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = (net(xv, bv, sv) - yv).abs().mean().item()
        print(f"   {T} r={r} epoch {ep + 1}: helper validation MAE {v:.4f} (weekly average {mae_base:.4f}) {time.time() - t0:.0f}s", flush=True)
        if ep == 0 and v >= mae_base: print("   KILL CHECK FAILED: not below the weekly average after epoch 1"); sys.exit(1)
        if v < best - 1e-5: best, bad, bst = v, 0, {k: a.clone() for k, a in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    net.load_state_dict(bst); net.eval(); return net


def run(T, r, O):
    f = f"{X.OUT}/pred_dev_N1_{T}_matched_r{r}.npy"
    if os.path.exists(f): return np.load(f)
    net = train(T, r, X.DEV[0]); Y, Xs, st = X.load_series(T); d = Data(Y, st); R = Y.shape[1]; Oo = np.repeat(O, R); Rr = np.tile(np.arange(R), len(O)); out = []
    with torch.no_grad():
        for i in range(0, len(Oo), 20000):
            x, b, s, _ = d.batch(Oo[i:i + 20000], Rr[i:i + 20000]); out.append(net(x, b, s).numpy())
    p = np.concatenate(out).reshape(len(O), R, H).transpose(0, 2, 1).astype(np.float32); np.save(f, p); return p


if mode == "dev":
    O = X.grid(*X.DEV, True)
    for T in ("chicago", "nyc"):
        for r in (1, 2): run(T, r, O)
    print("\n===== N1 and B1 (N1 + C5 equal average) on October and November 2021, MAE / RMSE =====")
    rat = {"N1": [], "B1": []}
    for T in ("chicago", "nyc"):
        n1 = np.mean([np.load(f"{X.OUT}/pred_dev_N1_{T}_matched_r{r}.npy") for r in (1, 2)], 0); c5 = np.mean([np.load(f"{X.OUT}/pred_dev_C5_{T}_matched_r{r}.npy") for r in (1, 2)], 0)
        c2 = X.cmp_chronos(T, O, np.mean([np.load(f"{X.OUT}/pred_dev_C2_{T}_matched_r{r}.npy") for r in (1, 2)], 0), "dev")
        for nm, p in (("N1", n1), ("B1", 0.5 * (n1 + c5))):
            q = X.cmp_chronos(T, O, p, "dev"); rat[nm].append((q["ours"][0] / c2["ours"][0], q["ours"][1] / c2["ours"][1]))
            print(f"{T} {nm}: {q['ours'][0]:.3f} / {q['ours'][1]:.3f} | C2 {c2['ours'][0]:.3f} / {c2['ours'][1]:.3f} | Chronos {q['chronos'][0]:.3f} / {q['chronos'][1]:.3f} | ratio to C2 MAE {rat[nm][-1][0]:.3f} RMSE {rat[nm][-1][1]:.3f}"
                  f" | ratio to Chronos MAE {q['ours'][0] / q['chronos'][0]:.3f} RMSE {q['ours'][1] / q['chronos'][1]:.3f} | daily diff {q['diff'][0]:+.4f} [{q['diff'][1]:+.4f}, {q['diff'][2]:+.4f}]")
    for nm, v in rat.items():
        ok = np.mean([a for a, b in v]) <= 0.97 and max(a for a, b in v) <= 1.01 and max(b for a, b in v) <= 1.02
        print(f"screen of PROTOCOL_V18 for {nm}: {'PASSED' if ok else 'NOT passed'}")
    print("screen done")
