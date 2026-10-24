"""PROTOCOL_SYDNEY.md: from the 20 allocator predictions (heads eh and eo, runs 1..10, calendar 2023) to the demand estimate per Sydney suburb. No Sydney demand
exists or is read. Steps, all from helper cities, public data and the calendar:
  shares  : equal average of the eh and eo ten-run ensembles (each run normalised per slot), as the final method
  rhythm  : mean weekday x slot profile of the share helpers Chicago, NYC and SF (predict_final.py); no month factor (southern hemisphere, no same-season helper)
  level   : the strict rule of the paper: unit chosen by leave-one-helper-out among the helper cities, then the log-pooled ratio of trips per unit over Chicago, NYC, SF and DC,
            applied to the Sydney value of that unit (ABS Census 2021 count for Greater Sydney, see PROTOCOL_SYDNEY.md); reported as the strict value AND flagged as unreliable
  display : for every suburb, weekday and half-hour: demand = (trips per day) x rhythm[w, s] x 7 / ... see below x share[suburb | w, s]
Outputs (results/): sydney_suburbs.csv, sydney_week.json (inputs of the interactive page), sydney_checks.txt.
Usage (from improve/): python sydney_results.py"""
import os, sys, json, numpy as np, pandas as pd, geopandas as gpd
from datetime import date, timedelta
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); RES = f"{HERE}/../results"
from runner_sydney import sydney_srcs
from slots import calendar_slots
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("def totals")], ns); YP, U, days = ns["YP"], ns["U"], ns["days"]
SYD_TRANSIT = 136614                                   # ABS Census 2021 QuickStats, Greater Sydney: employed people who travelled to work by train, bus, ferry or tram on census day (5.6%)
START = date(2023, 1, 1); norm = lambda p: p / p.sum(1, keepdims=True)
log = []; say = lambda s: (print(s, flush=True), log.append(s))

# ---- shares ----
sl = calendar_slots("sydney", START); di, si = np.array([a for a, _ in sl]), np.array([b for _, b in sl])
reg = gpd.read_file(f"{HERE}/../multicity/data/sydney/sydney_regions.geojson").sort_values("rid").reset_index(drop=True); R = len(reg)
PARTIAL = "--partial" in sys.argv                      # page test only: use the runs finished so far (both heads), never for the reported estimate
RUNS = [r for r in range(1, 11) if all(os.path.exists(f"{HERE}/out_cal/multi_sydney_from_{'+'.join(sydney_srcs(r))}_v{h}_seed{r - 1}.npy") for h in ("eh", "eo"))] if PARTIAL else list(range(1, 11))
H = {}
for h in ("eh", "eo"):
    runs = [norm(np.load(f"{HERE}/out_cal/multi_sydney_from_{'+'.join(sydney_srcs(r))}_v{h}_seed{r - 1}.npy").astype(np.float64)) for r in RUNS]
    assert all(p.shape == (len(sl), R) for p in runs), "prediction rows differ from the calendar slots or the suburbs"
    H[h] = (np.mean(runs, 0), np.std([p.mean(0) for p in runs], 0) if len(runs) > 1 else np.zeros(R))
if PARTIAL: RES = f"{RES}/partial"; os.makedirs(RES, exist_ok=True); print(f"PARTIAL TEST: runs {RUNS} only", flush=True)
P = norm(0.5 * H["eh"][0] + 0.5 * H["eo"][0])
say(f"shares: {len(sl)} calendar slots of 2023, {R} suburbs, heads eh and eo, ten runs each")

# ---- rhythm and level (predict_final.py) ----
Y = {c: np.load(YP[c]).astype(np.float64) for c in YP}; N = {c: Y[c].sum() for c in YP}
loo = {u: np.mean([abs(np.exp(np.mean([np.log(N[c] / U[c][u]) for c in YP if c != h])) * U[h][u] / N[h] - 1) for h in YP]) for u in U["chicago"]}
unit = min(loo, key=loo.get); assert unit == "transit commuters", f"the unit chosen by the helpers is {unit}; the Sydney value is only available for transit commuters"
rate = float(np.exp(np.mean([np.log(N[c] / U[c][unit]) for c in YP]))); lev_year = SYD_TRANSIT * rate
say(f"level: unit '{unit}' (leave-one-helper-out error {100 * loo[unit]:.1f}%), trips per commuter {rate:.2f} (log-pooled over {', '.join(YP)}), Greater Sydney value {SYD_TRANSIT:,} "
    f"-> strict yearly total {lev_year:,.0f} = {lev_year / 365:,.0f} per day (UNRELIABLE: census day 10 August 2021 was in the Sydney lockdown)")
DOW_h = {c: np.array([d.weekday() for d in days(c)]) for c in YP}


def profile(c):
    tot = Y[c].sum(2); r = np.array([tot[DOW_h[c] == k].sum(0) for k in range(7)]); return r / r.sum()


rh = np.mean([profile(c) for c in ("chicago", "nyc", "sf")], 0)                    # (7, 48), sums to 1 over the week
dow = np.array([(START + timedelta(days=int(d))).weekday() for d in range(365)]); nd = np.bincount(dow, minlength=7); w_t = dow[di]
Tt = lev_year * rh[w_t, si] / nd[w_t]; assert abs(Tt.sum() / lev_year - 1) < 1e-9
# ---- per suburb ----
ann = (P * Tt[:, None]).sum(0); share = ann / ann.sum()
S_ws = np.zeros((7, 48, R))                                                          # typical share of every suburb by weekday and half-hour (mean over the days of 2023)
for w in range(7):
    for s in range(48):
        m = (w_t == w) & (si == s); S_ws[w, s] = P[m].mean(0)
jobs, res = reg["jobs"].values, reg["res"].values
df = pd.DataFrame({"rank": 0, "suburb": reg["LOC_NAME"], "loc_pid": reg["LOC_PID"], "area_km2": reg["area_km2"].round(3), "jobs": jobs.round(0), "residents": res.round(0),
                   "share_pct": (100 * share).round(4), "trips_per_day_strict": (ann / 365).round(2), "run_to_run_sd_of_slot_share_pct": (100 * np.sqrt(H["eh"][1] ** 2 / 2 + H["eo"][1] ** 2 / 2)).round(4)})
df["rank"] = df["share_pct"].rank(ascending=False, method="first").astype(int); df = df.sort_values("rank")
df.to_csv(f"{RES}/sydney_suburbs.csv", index=False)
json.dump({"names": reg["LOC_NAME"].tolist(), "pid": reg["LOC_PID"].tolist(), "area": reg["area_km2"].round(3).tolist(), "jobs": jobs.round(0).tolist(), "res": res.round(0).tolist(),
           "rhythm": rh.round(6).tolist(), "nd": nd.tolist(), "level_year_strict": lev_year, "share_year": share.round(8).tolist(),
           "share_ws": np.round(S_ws * 1e6).astype(int).transpose(2, 0, 1).tolist()},
          open(f"{RES}/sydney_week.json", "w"), separators=(",", ":"))
# ---- checks (descriptive; there is nothing to validate against) ----
from scipy.stats import spearmanr
jsh, rsh = jobs / jobs.sum(), res / res.sum()
say(f"checks: Spearman of the predicted yearly share with the share of jobs {spearmanr(share, jsh)[0]:.2f}, of residents {spearmanr(share, rsh)[0]:.2f}; "
    f"top-10 suburbs hold {100 * np.sort(share)[-10:].sum():.1f}% of the demand (jobs-proportional: {100 * np.sort(jsh)[-10:].sum():.1f}%)")
say("top 15 suburbs by predicted share: " + "; ".join(f"{r.suburb} {r.share_pct:.2f}%" for r in df.head(15).itertuples()))
say("top 10 suburbs by jobs: " + "; ".join(f"{n} {100 * j:.2f}%" for n, j in sorted(zip(reg['LOC_NAME'], jsh), key=lambda t: -t[1])[:10]))
ap = int(np.argmax(np.load(f"{HERE}/data/enpl_sydney.npy"))); say(f"airport suburb {reg['LOC_NAME'][ap]}: predicted share {100 * share[ap]:.2f}% (rank {int(df.loc[df['suburb'] == reg['LOC_NAME'][ap], 'rank'].iloc[0])}), jobs share {100 * jsh[ap]:.2f}%")
cbd = [i for i, n in enumerate(reg["LOC_NAME"]) if n in ("Sydney", "Haymarket", "Barangaroo", "Millers Point", "The Rocks", "Dawes Point")]
say(f"CBD suburbs ({', '.join(reg['LOC_NAME'][cbd])}): {100 * share[cbd].sum():.1f}% of the demand against {100 * jsh[cbd].sum():.1f}% of the jobs")
for w, nm in ((1, "Tuesday"), (5, "Saturday")):
    pk = np.argsort(-S_ws[w, 18])[:5]; say(f"{nm} 09:00 top suburbs: " + "; ".join(f"{reg['LOC_NAME'][i]} {100 * S_ws[w, 18, i]:.2f}%" for i in pk))
    pk = np.argsort(-S_ws[w, 44])[:5]; say(f"{nm} 22:00 top suburbs: " + "; ".join(f"{reg['LOC_NAME'][i]} {100 * S_ws[w, 44, i]:.2f}%" for i in pk))
say(f"run-to-run: mean sd of a suburb's yearly share across the ten runs {100 * np.mean(np.sqrt(H['eh'][1] ** 2 / 2 + H['eo'][1] ** 2 / 2)):.4f} percentage points per slot-average share (mean share {100 / R:.3f}%)")
open(f"{RES}/sydney_checks.txt", "w").write("\n".join(log) + "\n"); print("done", flush=True)
