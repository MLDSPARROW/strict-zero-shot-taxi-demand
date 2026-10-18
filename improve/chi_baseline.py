"""PREDICTION step of the Chi, Xu and Chen (2026) allocation baseline (PROTOCOL_CHI_BASELINE.md, written before any run).
Trains the POI temporal-activity MLP on the helper cities only and writes the target's spatial shares on its calendar
slots. The target demand file is never opened. Output: out_cal/chi_<T>_seed<s>.npy (slots x regions).
Usage (from improve/): python chi_baseline.py <target> <helpers,comma> <seeds,comma>"""
import os, sys, numpy as np, torch, torch.nn as nn
from datetime import date, timedelta
from slots import calendar_slots
HERE = os.path.dirname(os.path.abspath(__file__)); B, MC = f"{HERE}/../bench77/data", f"{HERE}/../multicity/data"
torch.set_num_threads(2)
CITY = {"chicago": (f"{B}/chicago_2021_ca_30min.npy", f"{B}/region_static_chicago.npz", date(2021, 1, 1)),
        "nyc": (f"{B}/nyc_2021_zone_30min.npy", f"{B}/region_static_nyc.npz", date(2021, 1, 1)),
        "sf": (f"{MC}/sf_2023_tract_30min.npy", f"{MC}/region_static_sf.npz", date(2023, 1, 1)),
        "dc": (f"{MC}/dc_2021_tract_30min.npy", f"{MC}/region_static_dc.npz", date(2021, 1, 1)),
        "sf_zcta": (None, f"{MC}/region_static_sf_zcta.npz", date(2023, 1, 1)),
        "dc_cluster": (None, f"{MC}/region_static_dc_cluster.npz", date(2021, 1, 1)),
        "austin": (None, f"{MC}/region_static_austin.npz", date(2016, 6, 4)),
        "porto": (None, f"{MC}/region_static_porto.npz", date(2013, 7, 1))}
TGT, HEL, SEEDS = sys.argv[1], sys.argv[2].split(","), [int(s) for s in sys.argv[3].split(",")]
assert TGT not in HEL


def counts(c):                                     # 8 OSM POI groups (counts) + LODES resident workers (population)
    S = np.load(CITY[c][1]); n = np.expm1(S["X"][:, :8].astype(np.float64)) * S["area_km2"][:, None]
    return np.concatenate([n, np.load(f"{HERE}/data/extra_{c}.npz")["res"][:, None]], 1)


def template_obs(c):                               # weekly 48-slot observed allocation of a helper (Eq. 6 target)
    Y = np.load(CITY[c][0]).astype(np.float64); st = CITY[c][2]
    we = np.array([(st + timedelta(days=d)).weekday() >= 5 for d in range(Y.shape[0])])
    H = Y.reshape(Y.shape[0], 24, 2, -1).sum(2)    # day x hour x region
    obs = np.concatenate([H[~we].mean(0) * 5, H[we].mean(0) * 2], 0)   # 48 x R
    return obs / obs.sum()


ENC = torch.tensor([[float(t >= 24), np.cos(2 * np.pi * (t % 24) / 24), np.sin(2 * np.pi * (t % 24) / 24)] for t in range(48)], dtype=torch.float32)
ALPHA = torch.tensor([5.0] * 24 + [2.0] * 24)[:, None]
NC = {c: torch.tensor(counts(c), dtype=torch.float32) for c in HEL + [TGT]}
OBS = {c: torch.tensor(template_obs(c), dtype=torch.float32) for c in HEL}


def alloc(mlp, n):
    a = torch.relu(mlp(ENC))                                   # 48 x C activity levels (Eq. 4)
    h = a @ n.T                                                # 48 x R regional heat (Eq. 2)
    d = ALPHA * h; return d / d.sum().clamp(min=1e-12)         # Eq. 5


tgt_slots = calendar_slots(TGT, CITY[TGT][2]); st = CITY[TGT][2]
hour = np.array([s // 2 for _, s in tgt_slots]); wk = np.array([(st + timedelta(days=d)).weekday() >= 5 for d, _ in tgt_slots])
row = hour + 24 * wk
os.makedirs(f"{HERE}/out_cal", exist_ok=True)
for seed in SEEDS:
    torch.manual_seed(seed)
    C = NC[HEL[0]].shape[1]
    mlp = nn.Sequential(nn.Linear(3, 48), nn.LeakyReLU(), nn.Linear(48, 64), nn.LeakyReLU(), nn.Linear(64, 48), nn.LeakyReLU(), nn.Linear(48, C))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for ep in range(5000):
        loss = sum(torch.sum(OBS[c] * (torch.log(OBS[c].clamp(min=1e-12)) - torch.log(alloc(mlp, NC[c]).clamp(min=1e-12)))) for c in HEL) / len(HEL)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        d = alloc(mlp, NC[TGT]).numpy(); sh = d / np.maximum(d.sum(1, keepdims=True), 1e-12)
    np.save(f"{HERE}/out_cal/chi_{TGT}_seed{seed}.npy", sh[row].astype(np.float32))
    print(f"{TGT} <- {HEL} seed {seed}: final KL {loss.item():.4f}, active categories {int((torch.relu(mlp(ENC)).sum(0) > 0).sum())}/{C}", flush=True)
