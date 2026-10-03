"""Fig: four-city comparison (FULLY STRICT). Errors relative to the uniform split (lower is better). Values from score_all.py
(full year; DC hourly). Development targets: Chicago, NYC. Held-out targets: SF, DC."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
V = {  # (MAE, RMSE): uniform, jobs, jobs+res, base, final
 "Chicago\ndevelopment": [(4.06, 10.53), (2.52, 8.71), (2.59, 7.41), (2.68, 7.13), (1.83, 5.21)],
 "New York City\ndevelopment": [(10.03, 20.29), (7.49, 17.51), (8.64, 18.57), (7.84, 17.96), (7.45, 17.59)],
 "San Francisco\nheld-out": [(0.56, 1.14), (0.48, 1.15), (0.49, 1.01), (0.52, 1.18), (0.48, 1.08)],
 "Washington, DC\nheld-out, hourly": [(0.94, 2.69), (0.83, 2.69), (0.80, 2.40), (0.86, 2.57), (0.85, 2.66)]}
lab = ["Jobs-proportional", "Jobs + residents", "Base allocator", "This work"]
col = ["#8c8c8c", "#e07b39", "#2a9d8f", "#1f5aa6"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
x = np.arange(4); w = 0.2
for m, ax in enumerate(axs):
    for j in range(4):
        v = [V[c][j + 1][m] / V[c][0][m] for c in V]
        b = ax.bar(x + (j - 1.5) * w, v, w, color=col[j], label=lab[j])
    ax.axhline(1, color="black", lw=0.6, ls="--"); ax.set_xticks(x); ax.set_xticklabels(list(V), fontsize=7.3)
    ax.set_title(["MAE relative to uniform split", "RMSE relative to uniform split"][m]); ax.set_ylim(0, 1.15)
axs[0].set_ylabel("Relative error, lower is better"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); fig.savefig("C:/Users/user/Desktop/UrbanMind_paper_v2/figs/fig_four.pdf", bbox_inches="tight"); fig.savefig("C:/Users/user/Desktop/UrbanMind_paper_v2/figs/fig_four.png", dpi=170, bbox_inches="tight")
print("ok")
