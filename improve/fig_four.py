"""Fig: four-city comparison (FULLY STRICT). Errors relative to the uniform split (lower is better). Values from score_all.py (uniform, base,
final; full year; DC hourly) and from results/xgb_scale.txt (Chicago, NYC) and results/v10_heldout.txt (SF, DC) for the XGBoost allocator trained on
scale-matched helpers. Development targets: Chicago, NYC. Held-out targets: SF, DC."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
V = {  # (MAE, RMSE): uniform, base allocator, XGBoost allocator (scale-matched helpers), final method
 "Chicago\ndevelopment": [(4.06, 10.53), (2.68, 7.13), (2.45, 6.99), (1.83, 5.21)],
 "New York City\ndevelopment": [(10.03, 20.29), (7.84, 17.96), (6.29, 16.95), (7.45, 17.59)],
 "San Francisco\nheld-out": [(0.56, 1.14), (0.52, 1.18), (0.505, 1.170), (0.485, 1.082)],
 "Washington, DC\nheld-out, hourly": [(0.94, 2.69), (0.86, 2.57), (0.923, 2.780), (0.856, 2.658)]}
lab = ["Base allocator", "XGBoost allocator, scale-matched", "Final method (neural allocator)"]
col = ["#2a9d8f", "#e07b39", "#1f5aa6"]
OUT = "C:/Users/user/Desktop/TAXI DEMAND/UrbanMind_paper_v2/figs"
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
x = np.arange(4); w = 0.26
for m, ax in enumerate(axs):
    for j in range(3):
        v = [V[c][j + 1][m] / V[c][0][m] for c in V]
        b = ax.bar(x + (j - 1) * w, v, w, color=col[j], label=lab[j])
    ax.axhline(1, color="black", lw=0.6, ls="--"); ax.set_xticks(x); ax.set_xticklabels(list(V), fontsize=7.3)
    ax.set_title(["MAE relative to uniform split", "RMSE relative to uniform split"][m]); ax.set_ylim(0, 1.15)
axs[0].set_ylabel("Relative error, lower is better"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_four.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_four.png", dpi=170, bbox_inches="tight")
print("ok")
