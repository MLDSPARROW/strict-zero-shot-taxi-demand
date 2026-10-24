"""Fig: cost of strictness, December 2021, versions A to D. Values: A, B, C as in figures.py (cost_final.txt); D = Chronos-Bolt-small with the previous
seven days of each region (results/fm_chronos.txt, PROTOCOL_V6). Writes only fig_cost (figures.py writes the other figures)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
C = {"blue": "#1f5aa6", "orange": "#e07b39"}
OUT = "C:/Users/user/Desktop/TAXI DEMAND/UrbanMind_paper_v2/figs"
cost = {"chicago": [(2.18, 6.31), (1.68, 5.22), (1.05, 2.60), (1.08, 2.89)], "nyc": [(9.44, 21.02), (2.93, 9.84), (1.56, 4.76), (1.51, 4.69)]}
lab = ["A. Strict,\nno target data", "B. With 11 months\nof target data", "C. With recent\nobservations", "D. Foundation\nmodel, 7 days"]
fig, axs = plt.subplots(1, 2, figsize=(8.0, 2.8))
for k, tgt in enumerate(["chicago", "nyc"]):
    ax = axs[k]; w = 0.36; xs = np.arange(4)
    for j, (nm, c) in enumerate([("MAE", C["blue"]), ("RMSE", C["orange"])]):
        v = [a[j] for a in cost[tgt]]; bb = ax.bar(xs + (j - 0.5) * w, v, w, color=c, label=nm)
        for b, y in zip(bb, v): ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=6.8); ax.set_title(f"{'Chicago' if tgt == 'chicago' else 'New York City'}, December 2021")
    ax.set_ylabel("Error, trips per region per 30 min")
axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_cost.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_cost.png", dpi=170, bbox_inches="tight")
print("ok")
