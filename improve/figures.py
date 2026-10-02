"""Figures for paper v2 (2026-10-02). All values come from saved predictions / data (same pipeline as paper_numbers.py).
Output: Desktop/UrbanMind_paper_v2/figs/*.pdf (+ .png previews)."""
import os, io, contextlib, numpy as np, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import date, timedelta
from sklearn.cluster import KMeans
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
OUT = "C:/Users/user/Desktop/UrbanMind_paper_v2/figs"; os.makedirs(OUT, exist_ok=True)
C = {"blue": "#1f5aa6", "orange": "#e07b39", "green": "#2a9d8f", "red": "#c0392b", "grey": "#8c8c8c", "purple": "#7b4fa0", "gold": "#d4a017"}
src = open("idea_avg_rhythm.py", encoding="utf-8").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(src[:src.rindex("for tgt, (srcs, S) in st.CFG.items():")], ns)
st, strict_T = ns["st"], ns["strict_T"]
B, M = "../bench77/data", "../multicity"
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
def regions(c):
    if c == "chicago":
        g = gpd.read_file(f"{B}/chicago_community_areas.geojson"); g["rid"] = g["area_numbe"].astype(int)
    elif c == "nyc":
        g = gpd.read_file("../retune_out/taxi_zones_extracted/taxi_zones/taxi_zones.shp"); g["rid"] = g["LocationID"].astype(int); g = g.dissolve(by="rid", as_index=False)
    else:
        g = gpd.read_file(f"{M}/data/sf_tracts_2020.geojson").to_crs(epsg=4326)
        cen = g.to_crs(g.estimate_utm_crs()).geometry.centroid.to_crs(epsg=4326)
        g = g[cen.x.values > -122.6].dissolve(by="name", as_index=False).sort_values("name").reset_index(drop=True); g["rid"] = np.arange(len(g))
    g = g.to_crs(epsg=4326).sort_values("rid").reset_index(drop=True)
    return g.to_crs(g.estimate_utm_crs())
def strict_pred(tgt):
    srcs, S = st.CFG[tgt]; Y = st.Y[tgt]; tot = Y.sum(2)
    idx = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]; di, si = np.array([a for a, _ in idx]), np.array([b for _, b in idx])
    T = strict_T(tgt, S, di, si); p = np.load(f"out/FINAL_{tgt}_shares.npy"); P = np.zeros_like(Y); P[di, si] = p * T[:, None]
    pja = np.mean([np.load(f"{M}/multi_{tgt}_from_{srcs}_seed{s}.npy") for s in range(10)], 0); pja /= pja.sum(1, keepdims=True)
    Pb = np.zeros_like(Y); Pb[di, si] = pja * T[:, None]
    return Y, P, Pb

# ---------------- Fig. 2: maps of annual demand, true vs strictly zero-shot prediction ----------------
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.6), gridspec_kw={"wspace": 0.05, "hspace": 0.18})
for row, tgt in enumerate(["chicago", "nyc"]):
    g = regions(tgt); Y, P, Pb = strict_pred(tgt); yt = Y.sum((0, 1)); yp = P.sum((0, 1)); yb = Pb.sum((0, 1))
    vmin, vmax = max(min(yt[yt > 0].min(), yp.min()), 50), max(yt.max(), yp.max())
    for col, (vals, title) in enumerate([(yt, "Observed"), (yp, "This work"), (yb, "Base allocator")]):
        ax = axs[row, col]; g.assign(v=np.maximum(vals, vmin)).plot(column="v", ax=ax, cmap="magma_r", norm=LogNorm(vmin, vmax), edgecolor="white", linewidth=0.15)
        ax.set_axis_off()
        if row == 0: ax.set_title(title, fontsize=10)
        if col == 0: ax.text(-0.08, 0.5, "Chicago" if tgt == "chicago" else "New York City", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=LogNorm(vmin, vmax)); cb = fig.colorbar(sm, ax=axs[row, :], shrink=0.8, pad=0.01); cb.set_label("Pickups in 2021 (log scale)")
save(fig, "fig_maps")

# ---------------- Fig. 3: scale matching illustration (NYC and SF re-aggregated for the Chicago target) ----------------
fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.8), gridspec_kw={"wspace": 0.12})
for k, (c, tag) in enumerate([("nyc", "nyc_for_chicago_p1"), ("sf", "sf_for_chicago_p1")]):
    g = regions(c); S = np.load(f"{B}/region_static_{c}.npz") if c == "nyc" else np.load(f"{M}/data/region_static_{c}.npz")
    K = len(np.load(f"data/agg/{tag}_static.npz")["area_km2"])
    lab = KMeans(K, n_init=4, random_state=1).fit(S["centroid_km"], sample_weight=S["area_km2"]).labels_
    blocks = g.assign(b=lab).dissolve(by="b", as_index=False)
    a2 = np.load(f"data/agg/{tag}_static.npz")["area_km2"]
    rng = np.random.default_rng(3)
    g.plot(ax=axs[2 * k], color=plt.cm.tab20(rng.integers(0, 20, len(g))), edgecolor="white", linewidth=0.15)
    blocks.plot(ax=axs[2 * k + 1], color=plt.cm.tab20(rng.integers(0, 20, len(blocks))), edgecolor="white", linewidth=0.4)
    name = "New York City" if c == "nyc" else "San Francisco"
    axs[2 * k].set_title(f"{name}, original\n{len(g)} regions\nmedian {np.median(S['area_km2']):.2f} km$^2$", fontsize=8.5)
    axs[2 * k + 1].set_title(f"{name}, re-aggregated\n{len(blocks)} blocks\nmedian {np.median(a2):.2f} km$^2$", fontsize=8.5)
for ax in axs: ax.set_axis_off()
save(fig, "fig_scale")

# ---------------- Fig. 4: one week of demand ----------------
d0 = (date(2021, 10, 4) - date(2021, 1, 1)).days                      # Monday 4 Oct 2021 .. Sunday 10 Oct 2021
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True)
x = np.arange(7 * 48) / 48
for col, (tgt, rid, rname) in enumerate([("chicago", 32, "Loop (community area 32)"), ("nyc", 161, "Midtown Center (zone 161)")]):
    Y, P, _ = strict_pred(tgt); rids = np.load(f"{B}/region_static_{tgt}.npz")["rid"]; k = int(np.where(rids == rid)[0][0])
    city_t, city_p = Y[d0:d0 + 7].sum(2).ravel(), P[d0:d0 + 7].sum(2).ravel()
    reg_t, reg_p = Y[d0:d0 + 7, :, k].ravel(), P[d0:d0 + 7, :, k].ravel()
    cname = "Chicago" if tgt == "chicago" else "New York City"
    for row, (yt, yp, title) in enumerate([(city_t, city_p, f"{cname}: whole city"), (reg_t, reg_p, f"{cname}: {rname}")]):
        ax = axs[row, col]; ax.plot(x, yt, color=C["grey"], lw=1.1, label="Observed"); ax.plot(x, yp, color=C["blue"], lw=1.1, label="This work (strict)")
        ax.set_title(title); ax.set_ylabel("Pickups per 30 min")
        if row == 1: ax.set_xticks(np.arange(7) + 0.5); ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
axs[0, 0].legend(frameon=False, loc="upper left")
fig.text(0.5, -0.01, "Week of 4–10 October 2021", ha="center")
fig.tight_layout(); save(fig, "fig_week")

# ---------------- Fig. 5: step-by-step improvements (full year, strict) ----------------
vals = {"chicago": [(2.68, 7.13), (2.40, 6.02), (2.02, 6.00), (2.01, 6.33), (1.80, 4.83), (1.83, 5.21)],
        "nyc": [(7.84, 17.96), (7.60, 18.16), (6.96, 17.43), (6.96, 17.28), (8.17, 18.72), (7.45, 17.59)]}
labels = ["Base\nallocator", "+ airport\nbranch (eh)", "eh + scale\nmatching", "eh + 10\npartitions", "eo + 10\npartitions", "Final\n(average)"]
cols = [C["grey"], C["orange"], C["green"], C["green"], C["purple"], C["blue"]]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6))
for col, tgt in enumerate(["chicago", "nyc"]):
    for row, (m, nm) in enumerate([(0, "MAE"), (1, "RMSE")]):
        ax = axs[row, col]; v = [a[m] for a in vals[tgt]]; bars = ax.bar(range(6), v, color=cols, width=0.7)
        for b, y in zip(bars, v): ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(0, max(v) * 1.12); ax.set_ylabel(nm)
        ax.set_title(f"{'Chicago' if tgt == 'chicago' else 'New York City'}: {nm}"); ax.set_xticks(range(6))
        ax.set_xticklabels(labels if row == 1 else [""] * 6, fontsize=7.2)
fig.tight_layout(); save(fig, "fig_steps")

# ---------------- Fig. 6: cost of strictness (December 2021) ----------------
cost = {"chicago": [(2.18, 6.31), (1.68, 5.22), (1.05, 2.60)], "nyc": [(9.44, 21.02), (2.93, 9.84), (1.56, 4.76)]}
lab = ["A. Strict\n(no target data)", "B. + 11 months\nof target data", "C. + recent target\nobservations"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
for k, tgt in enumerate(["chicago", "nyc"]):
    ax = axs[k]; w = 0.36; xs = np.arange(3)
    for j, (nm, c) in enumerate([("MAE", C["blue"]), ("RMSE", C["orange"])]):
        v = [a[j] for a in cost[tgt]]; bb = ax.bar(xs + (j - 0.5) * w, v, w, color=c, label=nm)
        for b, y in zip(bb, v): ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7.4); ax.set_title(f"{'Chicago' if tgt == 'chicago' else 'New York City'}, December 2021")
    ax.set_ylabel("Error (trips per region per 30 min)")
axs[0].legend(frameon=False)
fig.tight_layout(); save(fig, "fig_cost")
print("figures written to", OUT)
