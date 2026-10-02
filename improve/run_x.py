"""round 6 variants of the multi-source joint allocator (copy of multicity/run_multi.py + switches).
  x = extra static covariates (Rank 1): log jobs/km2, log residents/km2, log rail stations/km2, airport flag
      (improve/build_extra.py; LODES 2021, OSM, OurAirports -- no taxi demand)
  o = fixed area offset (Rank 2): logits = f(x) + log(area_km2); the learned log-relative-area feature is removed
  r = pairwise ranking auxiliary loss (Rank 5): logistic pair loss on logits, pairs sampled per slot among regions
      that are not the slot's top-1% by true share, weight 0.1
ROUND 7 switches (round 7; same pre-declared screen + 10-seed rule, declared 2026-10-01 before running):
  w = within-city feature scaling: each city's features scaled by ITS OWN median/IQR (static map data only)
  e = add log1p(FAA CY2021 airport enplanements, millions) of the airport inside the region (0 elsewhere)
  h = hub mixture: p = (1-a) softmax(z) + a softmax(b * log1p(enpl)) over airport regions; a, b global learned
      scalars (a = 0 for a city with no airport region). Also: source held-out peak ratio is printed (flatness check).
COST-OF-HISTORY (declared 2026-10-02): "<city>_hist" = the target's OWN taxi data Jan-Nov 2021 used as the only helper (NOT strict;
      used only to measure what our method gains from target history). Tested on December.
ROUND 11 (declared 2026-10-02, before running): RULE R1 = use EVERY taxi helper city scale-matched to the target; a helper whose
      median region area is already larger than the target's cannot be matched and is left out. Chicago <- nyc_for_chicago +
      sf_for_chicago + dc_for_chicago; NYC <- sf_for_nyc + dc_for_nyc. Variants eh, eo; 3 seeds; compared FULLY STRICT with the
      current scale-matched models (Chicago <- nyc,sf matched; NYC <- chicago + sf matched).
ROUND 10 (declared 2026-10-01, before running): SCALE-MATCHED HELPERS (aggregate.py): helper cities re-aggregated so their
      median block area matches the target's median region area (public polygons only). Chicago target: nyc_for_chicago +
      sf_for_chicago; NYC target: chicago (already coarser) + sf_for_nyc. Variants eh and eo, 3 seeds, compared FULLY STRICT
      (trip MAE / RMSE / MAPE) with the same variants trained on native-scale helpers, and as the average(eh, eo).
ROUND 9 (declared 2026-10-01, before running): j = add log1p(LODES jobs per km2) of the area as a direct input
      (downtown signal; Test 2 showed top-10% job areas are under-predicted in both targets). Screened as ehj and eoj vs eh / eo.
ROUND 8 (declared 2026-10-01, before running): b = area weight that depends on the AREA's own features:
      logits = f(x) + beta(x) * log(area_km2), beta(x) = sigmoid(linear(h)) in [0,1], h = the area's encoded features
      (same rule in every city; no city-specific parameter). Screened as "ehb" vs eh/eho with the usual 3-seed rule.
ROUND 7b (declared 2026-10-01, before running): combinations eo, eho, xoe screened with the same 3-seed rule vs the
      3-seed current best; passing ones go to the 10-seed paired test.
PRE-DECLARED (2026-09-30, before any run): 3 seeds (0,1,2) per variant, compared with the 3-seed ensemble of the
current best (seeds 0-2 of run_multi.py, same sources). A variant PASSES the screen if its 3-seed ensemble MAE_share
is >= 3% better in at least one direction and not > 3% worse in the other. Passing variants go to 10 seeds and a
paired t-test (p < 0.05, both directions reported). Target demand is used for scoring only.
Usage: python run_x.py <target> <srcs> <seeds> <flags: any of x,o,r>
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datetime import date, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar

HERE = os.path.dirname(os.path.abspath(__file__))
B77 = os.path.join(HERE, "..", "bench77", "data")
MC = os.path.join(HERE, "..", "multicity", "data")
torch.set_num_threads(max(1, os.cpu_count() // 2))
TGT, SRCS = sys.argv[1], sys.argv[2].split(",")
SEEDS = [int(s) for s in sys.argv[3].split(",")]
FL = sys.argv[4]; TAG = "_v" + FL
EXTRA, OFFSET, RANK = "x" in FL, "o" in FL, "r" in FL
WITHIN, ENPLF, HUB = "w" in FL, "e" in FL, "h" in FL
BETA = "b" in FL
JOBS = "j" in FL
assert TGT not in SRCS
CITY = {"nyc": (f"{B77}/nyc_2021_zone_30min.npy", f"{B77}/region_static_nyc.npz", date(2021, 1, 1), "US"),
        "chicago": (f"{B77}/chicago_2021_ca_30min.npy", f"{B77}/region_static_chicago.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago": (f"{HERE}/data/agg/nyc_for_chicago_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_static.npz", date(2021, 1, 1), "US"),
        "sf_for_chicago": (f"{HERE}/data/agg/sf_for_chicago_Y.npy", f"{HERE}/data/agg/sf_for_chicago_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc": (f"{HERE}/data/agg/sf_for_nyc_Y.npy", f"{HERE}/data/agg/sf_for_nyc_static.npz", date(2023, 1, 1), "US"),
        "nyc_for_chicago_p1": (f"{HERE}/data/agg/nyc_for_chicago_p1_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p1_static.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago_p2": (f"{HERE}/data/agg/nyc_for_chicago_p2_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p2_static.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago_p3": (f"{HERE}/data/agg/nyc_for_chicago_p3_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p3_static.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago_p4": (f"{HERE}/data/agg/nyc_for_chicago_p4_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p4_static.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago_p5": (f"{HERE}/data/agg/nyc_for_chicago_p5_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p5_static.npz", date(2021, 1, 1), "US"),
        "sf_for_chicago_p1": (f"{HERE}/data/agg/sf_for_chicago_p1_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p1_static.npz", date(2023, 1, 1), "US"),
        "sf_for_chicago_p2": (f"{HERE}/data/agg/sf_for_chicago_p2_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p2_static.npz", date(2023, 1, 1), "US"),
        "sf_for_chicago_p3": (f"{HERE}/data/agg/sf_for_chicago_p3_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p3_static.npz", date(2023, 1, 1), "US"),
        "sf_for_chicago_p4": (f"{HERE}/data/agg/sf_for_chicago_p4_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p4_static.npz", date(2023, 1, 1), "US"),
        "sf_for_chicago_p5": (f"{HERE}/data/agg/sf_for_chicago_p5_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p5_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p1": (f"{HERE}/data/agg/sf_for_nyc_p1_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p1_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p2": (f"{HERE}/data/agg/sf_for_nyc_p2_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p2_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p3": (f"{HERE}/data/agg/sf_for_nyc_p3_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p3_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p4": (f"{HERE}/data/agg/sf_for_nyc_p4_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p4_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p5": (f"{HERE}/data/agg/sf_for_nyc_p5_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p5_static.npz", date(2023, 1, 1), "US"),
        "dc_for_chicago": (f"{HERE}/data/agg/dc_for_chicago_Y.npy", f"{HERE}/data/agg/dc_for_chicago_static.npz", date(2021, 1, 1), "US"),
        "dc_for_nyc": (f"{HERE}/data/agg/dc_for_nyc_Y.npy", f"{HERE}/data/agg/dc_for_nyc_static.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago_p6": (f"{HERE}/data/agg/nyc_for_chicago_p6_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p6_static.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago_p7": (f"{HERE}/data/agg/nyc_for_chicago_p7_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p7_static.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago_p8": (f"{HERE}/data/agg/nyc_for_chicago_p8_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p8_static.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago_p9": (f"{HERE}/data/agg/nyc_for_chicago_p9_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p9_static.npz", date(2021, 1, 1), "US"),
        "nyc_for_chicago_p10": (f"{HERE}/data/agg/nyc_for_chicago_p10_Y.npy", f"{HERE}/data/agg/nyc_for_chicago_p10_static.npz", date(2021, 1, 1), "US"),
        "sf_for_chicago_p6": (f"{HERE}/data/agg/sf_for_chicago_p6_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p6_static.npz", date(2023, 1, 1), "US"),
        "sf_for_chicago_p7": (f"{HERE}/data/agg/sf_for_chicago_p7_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p7_static.npz", date(2023, 1, 1), "US"),
        "sf_for_chicago_p8": (f"{HERE}/data/agg/sf_for_chicago_p8_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p8_static.npz", date(2023, 1, 1), "US"),
        "sf_for_chicago_p9": (f"{HERE}/data/agg/sf_for_chicago_p9_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p9_static.npz", date(2023, 1, 1), "US"),
        "sf_for_chicago_p10": (f"{HERE}/data/agg/sf_for_chicago_p10_Y.npy", f"{HERE}/data/agg/sf_for_chicago_p10_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p6": (f"{HERE}/data/agg/sf_for_nyc_p6_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p6_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p7": (f"{HERE}/data/agg/sf_for_nyc_p7_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p7_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p8": (f"{HERE}/data/agg/sf_for_nyc_p8_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p8_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p9": (f"{HERE}/data/agg/sf_for_nyc_p9_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p9_static.npz", date(2023, 1, 1), "US"),
        "sf_for_nyc_p10": (f"{HERE}/data/agg/sf_for_nyc_p10_Y.npy", f"{HERE}/data/agg/sf_for_nyc_p10_static.npz", date(2023, 1, 1), "US"),
        "chicago_hist": (f"{HERE}/data/agg/chicago_hist_Y.npy", f"{HERE}/data/agg/chicago_hist_static.npz", date(2021, 1, 1), "US"),
        "nyc_hist": (f"{HERE}/data/agg/nyc_hist_Y.npy", f"{HERE}/data/agg/nyc_hist_static.npz", date(2021, 1, 1), "US"),
        "austin": (f"{MC}/austin_2016_cell_30min.npy", f"{MC}/region_static_austin.npz", date(2016, 6, 4), "US"),
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
    Y = np.load(yp).astype(np.float64); S = np.load(sp)
    tot = Y.sum(axis=2); cal = cal_fn(start, country)
    idx = [(d, s) for d in range(Y.shape[0]) for s in range(48) if tot[d, s] > 0]
    raw = S["X"][:, [f for f in FEAT if not (OFFSET and f == 16)]].astype(np.float64)
    if EXTRA: raw = np.concatenate([raw, np.load(f"{HERE}/data/extra_{city}.npz")["E"].astype(np.float64)], 1)
    enpl = np.load(f"{HERE}/data/enpl_{city}.npy").astype(np.float64)
    if JOBS: raw = np.concatenate([raw, np.load(f"{HERE}/data/extra_{city}.npz")["E"][:, :1].astype(np.float64)], 1)
    if ENPLF: raw = np.concatenate([raw, np.log1p(enpl)[:, None]], 1)
    return {"Y": Y, "raw": raw, "hub": np.log1p(enpl).astype(np.float32), "la": np.log(S["area_km2"]).astype(np.float32), "area": S["area_km2"], "idx": idx,
            "C": np.stack([cal(d, s) for d, s in idx]), "Sh": np.stack([Y[d, s] / tot[d, s] for d, s in idx]).astype(np.float32),
            "T": np.array([tot[d, s] for d, s in idx])}


data = {c: load(c) for c in SRCS + [TGT]}
pool = np.concatenate([data[c]["raw"] for c in SRCS])
q25, med, q75 = np.percentile(pool, [25, 50, 75], axis=0)
scale = np.where(q75 - q25 < 1e-5, 1.0, q75 - q25)
for c in data:
    if WITHIN:
        q25, med, q75 = np.percentile(data[c]["raw"], [25, 50, 75], axis=0); scale = np.where(q75 - q25 < 1e-5, 1.0, q75 - q25)
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
        if BETA: self.beta = nn.Linear(hid, 1)          # created only when used, so other variants keep identical random init
        self.ha, self.hb = nn.Parameter(torch.tensor(-2.0)), nn.Parameter(torch.tensor(1.0))
        self.out = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, 1))
    def forward(self, X, C):
        B, R = C.shape[0], X.shape[0]
        h = self.enc(torch.cat([X[None].expand(B, R, -1), C[:, None, :].expand(B, R, -1)], dim=2))
        for i in range(len(self.self_w)):
            h = self.norm[i](h + F.gelu(self.self_w[i](h) + self.ctx_w[i](h.mean(dim=1, keepdim=True))))
        z = self.out(h).squeeze(-1)
        if OFFSET: z = z + self.la
        if BETA: z = z + torch.sigmoid(self.beta(h)).squeeze(-1) * self.la
        lp = F.log_softmax(z, dim=1)
        if HUB and bool((self.hub > 0).any()):
            hl = torch.where(self.hub > 0, self.hb * self.hub, torch.full_like(self.hub, -1e9))
            a = torch.sigmoid(self.ha)
            lp = torch.logaddexp(torch.log1p(-a) + lp, torch.log(a) + F.log_softmax(hl, dim=0)[None])
        return lp


T_ = {c: {k: torch.tensor(data[c][k]) for k in ("X", "C", "Sh", "la", "hub")} for c in data}
NF = data[TGT]["X"].shape[1]


def run(mdl, c, C):
    mdl.la, mdl.hub = T_[c]["la"], T_[c]["hub"]
    return mdl(T_[c]["X"], C)


def rank_loss(lp, sh, npair=256):
    B, R = sh.shape
    thr = torch.quantile(sh, 0.99, dim=1, keepdim=True)
    i = torch.randint(0, R, (B, npair)); j = torch.randint(0, R, (B, npair))
    si, sj = sh.gather(1, i), sh.gather(1, j)
    ok = (si < thr) & (sj < thr) & (si != sj)
    d = (lp.gather(1, i) - lp.gather(1, j)) * torch.sign(si - sj)
    return (F.softplus(-d) * ok).sum() / ok.sum().clamp(min=1)


def predict(mdl, c, sel):
    with torch.no_grad():
        return torch.cat([run(mdl, c, T_[c]["C"][sel[i:i + 512]]).exp() for i in range(0, len(sel), 512)])


def train(seed, epochs=25, bs=64, steps=233):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mdl = Allocator(NF); opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        mdl.train()
        for _ in range(steps):
            c = SRCS[rng.integers(len(SRCS))]
            b = rng.choice(data[c]["tr"], size=bs, replace=False)
            lp = run(mdl, c, T_[c]["C"][b])
            loss = -(T_[c]["Sh"][b] * lp).sum(dim=1).mean()
            if RANK: loss = loss + 0.1 * rank_loss(lp, T_[c]["Sh"][b])
            opt.zero_grad(); loss.backward(); opt.step()
        mdl.eval()
        v = float(np.mean([(predict(mdl, c, data[c]["va"]) - T_[c]["Sh"][data[c]["va"]]).abs().mean().item() for c in SRCS]))
        if v < best - 1e-7: best, state, bad = v, {k: t.clone() for k, t in mdl.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4: break
    mdl.load_state_dict(state); mdl.eval()
    for c in SRCS:                       # flatness check on the SOURCE cities' own held-out days
        pv, tv = predict(mdl, c, data[c]["va"]), T_[c]["Sh"][data[c]["va"]]
        print(f"    source {c} held-out: MAE_share {(pv-tv).abs().mean().item():.5f} peak ratio {(pv.max(1).values.mean()/tv.max(1).values.mean()).item():.2f}"
              + (f" hub a={torch.sigmoid(mdl.ha).item():.3f} b={mdl.hb.item():.2f}" if HUB else ""), flush=True)
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
out = f"{HERE}/out/multi_{TGT}_from_{'+'.join(SRCS)}{TAG}"
for seed in SEEDS:
    t0 = time.time()
    if os.path.exists(f"{out}_seed{seed}.npy"):          # resume after an interruption: reuse finished seeds
        p, v, ep = np.load(f"{out}_seed{seed}.npy"), float("nan"), -1
    else:
        p, v, ep = train(seed)
    preds.append(p)
    r = evaluate(p); r.update({"src_val": v, "epochs": ep, "seconds": round(time.time() - t0)}); results[f"seed{seed}"] = r
    print(f"  seed {seed}: MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f} MAPE>=5={r['MAPE_ge5']:.1f}% MAE_share={r['MAE_share']:.5f} "
          f"peak={r['peak_ratio']:.2f} ({r['seconds']}s, {ep} ep)", flush=True)
    np.save(f"{out}_seed{seed}.npy", p)
ens = np.mean(preds, axis=0); ens /= ens.sum(axis=1, keepdims=True)
results["ensemble"] = evaluate(ens)
r = results["ensemble"]
print(f"  ENSEMBLE of {len(preds)}: MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f} MAPE>=5={r['MAPE_ge5']:.1f}% "
      f"MAE_share={r['MAE_share']:.5f} peak={r['peak_ratio']:.2f}", flush=True)
json.dump(results, open(f"{out}.json", "w"), indent=1)
print("done", flush=True)
