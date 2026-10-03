"""Multi-source joint allocator (multi-city step 1).
Train on one or more SOURCE cities with city-balanced sampling (each step: pick a source city uniformly, then
a batch of its training slots), feature scaling = median/IQR pooled over all source regions, early stopping on
the mean over source cities of held-out-day MAE_share. Predict the TARGET city from its static map only.
Pre-declared comparison (2026-09-29): single source vs single source + SF, 5 seeds + 5-seed average, both
directions; success = lower MAE_share of the 5-seed average in BOTH directions (count metrics reported too).
Usage: python run_multi.py <target> <src1,src2,...> <seeds> <tag>
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datetime import date, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "improve")); from slots import calendar_slots

HERE = os.path.dirname(os.path.abspath(__file__))
B77 = os.path.join(HERE, "..", "bench77", "data")
MC = os.path.join(HERE, "data")
torch.set_num_threads(os.cpu_count())
CAL = "--calendar" in sys.argv                  # PROTOCOL_REVISION.md: predict on calendar slots, never read target demand
sys.argv = [a for a in sys.argv if a != "--calendar"]
TGT, SRCS = sys.argv[1], sys.argv[2].split(",")
SEEDS = [int(s) for s in sys.argv[3].split(",")]
TAG = sys.argv[4]
assert TGT not in SRCS
CITY = {"nyc": (f"{B77}/nyc_2021_zone_30min.npy", f"{B77}/region_static_nyc.npz", date(2021, 1, 1), "US"),
        "chicago": (f"{B77}/chicago_2021_ca_30min.npy", f"{B77}/region_static_chicago.npz", date(2021, 1, 1), "US"),
        "sf": (f"{MC}/sf_2023_tract_30min.npy", f"{MC}/region_static_sf.npz", date(2023, 1, 1), "US"),
        "porto": (f"{MC}/porto_region_30min.npy", f"{MC}/region_static_porto.npz", date(2013, 7, 1), "PT"),
        "dc": (f"{MC}/dc_2021_tract_30min.npy", f"{MC}/region_static_dc.npz", date(2021, 1, 1), "US"),
        "porto30": (f"{MC}/porto30_region_30min.npy", f"{MC}/region_static_porto.npz", date(2013, 7, 1), "PT"),
        # controlled degradations of SF (same regions/features, degraded demand labels) -- diagnostic only
        "sf_time": (f"{MC}/sf_time_2023_tract_30min.npy", f"{MC}/region_static_sf.npz", date(2023, 1, 1), "US"),
        "sf_space": (f"{MC}/sf_space_2023_tract_30min.npy", f"{MC}/region_static_sf.npz", date(2023, 1, 1), "US"),
        "sf_both": (f"{MC}/sf_both_2023_tract_30min.npy", f"{MC}/region_static_sf.npz", date(2023, 1, 1), "US")}
FEAT = list(range(18))


def cal_fn(start, country):
    if country == "US":
        hol = set(USFederalHolidayCalendar().holidays(start=str(start), end=str(start + timedelta(days=400))).date)
    else:
        import holidays as _h
        hol = set(_h.Portugal(years=range(start.year, start.year + 2)).keys())
    def cal(day, slot):
        d = start + timedelta(days=int(day)); dow = d.weekday()
        return np.array([np.sin(2*np.pi*slot/48), np.cos(2*np.pi*slot/48), np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
                         float(dow >= 5), float(d in hol), np.sin(2*np.pi*d.month/12), np.cos(2*np.pi*d.month/12),
                         float(slot // 2 in (7, 8, 9, 16, 17, 18))], dtype=np.float32)
    return cal


def load(city):
    yp, sp, start, country = CITY[city]
    if CAL and city == TGT:                      # target: calendar slots only, its demand file is not opened
        Y, S, cal = None, np.load(sp), cal_fn(start, country); idx = calendar_slots(city, start)
    else:
        Y = np.load(yp).astype(np.float64); S = np.load(sp)
        tot = Y.sum(axis=2); cal = cal_fn(start, country)
        idx = [(d, s) for d in range(Y.shape[0]) for s in range(48) if tot[d, s] > 0]
    return {"Y": Y, "raw": S["X"][:, FEAT].astype(np.float64), "area": S["area_km2"], "idx": idx,
            "C": np.stack([cal(d, s) for d, s in idx]),
            **({} if Y is None else {"Sh": np.stack([Y[d, s] / tot[d, s] for d, s in idx]).astype(np.float32),
                                     "T": np.array([tot[d, s] for d, s in idx])})}


data = {c: load(c) for c in SRCS + [TGT]}
pool = np.concatenate([data[c]["raw"] for c in SRCS])
q25, med, q75 = np.percentile(pool, [25, 50, 75], axis=0)
scale = np.where(q75 - q25 < 1e-5, 1.0, q75 - q25)
for c in data:
    data[c]["X"] = np.clip((data[c]["raw"] - med) / scale, -8, 8).astype(np.float32)
for c in SRCS:
    rng = np.random.default_rng(0); nd = data[c]["Y"].shape[0]
    vd = set(rng.choice(nd, size=int(nd * 0.15), replace=False).tolist())
    m = np.array([d in vd for d, _ in data[c]["idx"]])
    data[c]["tr"], data[c]["va"] = np.where(~m)[0], np.where(m)[0]
print(f"target {TGT} <- sources {SRCS}: " + ", ".join(f"{c} {len(data[c]['X'])} regions {len(data[c]['idx'])} slots" for c in data), flush=True)


class Allocator(nn.Module):                      # same joint allocator as bench77 (no graph)
    def __init__(self, fin, hid=64, layers=2, drop=0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(fin + 9, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, hid))
        self.self_w = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])
        self.ctx_w = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, 1))
    def forward(self, X, C):
        B, R = C.shape[0], X.shape[0]
        h = self.enc(torch.cat([X[None].expand(B, R, -1), C[:, None, :].expand(B, R, -1)], dim=2))
        for i in range(len(self.self_w)):
            h = self.norm[i](h + F.gelu(self.self_w[i](h) + self.ctx_w[i](h.mean(dim=1, keepdim=True))))
        return F.log_softmax(self.out(h).squeeze(-1), dim=1)


T_ = {c: {k: torch.tensor(data[c][k]) for k in ("X", "C", "Sh") if k in data[c]} for c in data}


def predict(mdl, c, sel):
    with torch.no_grad():
        return torch.cat([mdl(T_[c]["X"], T_[c]["C"][sel[i:i + 512]]).exp() for i in range(0, len(sel), 512)])


def train(seed, epochs=25, bs=64, steps=233):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mdl = Allocator(len(FEAT)); opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        mdl.train()
        for _ in range(steps):
            c = SRCS[rng.integers(len(SRCS))]
            b = rng.choice(data[c]["tr"], size=bs, replace=False)
            loss = -(T_[c]["Sh"][b] * mdl(T_[c]["X"], T_[c]["C"][b])).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        mdl.eval()
        v = float(np.mean([(predict(mdl, c, data[c]["va"]) - T_[c]["Sh"][data[c]["va"]]).abs().mean().item() for c in SRCS]))
        if v < best - 1e-7: best, state, bad = v, {k: t.clone() for k, t in mdl.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4: break
    mdl.load_state_dict(state); mdl.eval()
    return predict(mdl, TGT, np.arange(len(data[TGT]["idx"]))).numpy(), best, ep + 1


def evaluate(p):
    d = data[TGT]; Y = d["Y"]; P = np.zeros_like(Y)
    for k, (dd, s) in enumerate(d["idx"]): P[dd, s] = p[k] * d["T"][k]
    e = P - Y; m5, m1 = Y >= 5, Y >= 1
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAPE_ge5": float(100 * np.mean(np.abs(e[m5]) / Y[m5])), "MAPE_ge1": float(100 * np.mean(np.abs(e[m1]) / Y[m1])),
            "MAE_share": float(np.abs(p - d["Sh"]).mean()),
            "peak_ratio": float(p.max(axis=1).mean() / d["Sh"].max(axis=1).mean())}


results, preds = {}, []
if CAL: os.makedirs(f"{HERE}/out_cal", exist_ok=True)
out = f"{HERE}/{'out_cal/' if CAL else ''}multi_{TGT}_from_{'+'.join(SRCS)}{TAG}"
for seed in SEEDS:
    t0 = time.time()
    if os.path.exists(f"{out}_seed{seed}.npy"):          # resume after an interruption: reuse finished seeds
        p, v, ep = np.load(f"{out}_seed{seed}.npy"), float("nan"), -1
    else:
        p, v, ep = train(seed)
    preds.append(p)
    r = {} if CAL else evaluate(p); r.update({"src_val": v, "epochs": ep, "seconds": round(time.time() - t0)}); results[f"seed{seed}"] = r
    if CAL: print(f"  seed {seed}: prediction saved, {p.shape[0]} calendar slots ({r['seconds']}s, {ep} ep)", flush=True)
    else: print(f"  seed {seed}: MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f} MAPE>=5={r['MAPE_ge5']:.1f}% MAE_share={r['MAE_share']:.5f} "
          f"peak={r['peak_ratio']:.2f} ({r['seconds']}s, {ep} ep)", flush=True)
    np.save(f"{out}_seed{seed}.npy", p)
ens = np.mean(preds, axis=0); ens /= ens.sum(axis=1, keepdims=True)
if not CAL:
    results["ensemble"] = evaluate(ens); r = results["ensemble"]
    print(f"  ENSEMBLE of {len(preds)}: MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f} MAPE>=5={r['MAPE_ge5']:.1f}% "
      f"MAE_share={r['MAE_share']:.5f} peak={r['peak_ratio']:.2f}", flush=True)
json.dump(results, open(f"{out}.json", "w"), indent=1)
print("done", flush=True)
