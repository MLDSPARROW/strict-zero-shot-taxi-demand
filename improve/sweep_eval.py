"""EVALUATION of the block-size sweep (PROTOCOL_SCALE_SWEEP.md). Trip errors only. All seven ratios were run on Kaggle CPU
workers (out_kaggle/). Prints a table and writes figs/fig_sweep.pdf (MAE and RMSE of the final method against q)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); B = f"{HERE}/../bench77/data"; KG = f"{HERE}/out_kaggle"
OUT = "C:/Users/user/Desktop/UrbanMind_paper_v2/figs"
YP = {"chicago": f"{B}/chicago_2021_ca_30min.npy", "nyc": f"{B}/nyc_2021_zone_30min.npy"}
NAT = {"chicago": "nyc+sf", "nyc": "chicago+sf"}
HEL = {"chicago": lambda c, r: f"nyc_for_chicago_{c}{r}+sf_for_chicago_{c}{r}", "nyc": lambda c, r: f"chicago+sf_for_nyc_{c}{r}"}
Q = [(0.25, "x025p"), (0.5, "H"), (0.75, "x075p"), (1.0, "p"), (1.5, "x150p"), (2.0, "D"), (4.0, "x400p")]
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def boot(x, rng):
    nb = len(x) // 7
    return [x[np.concatenate([np.arange(k, k + 7) for k in rng.integers(0, len(x) - 7, nb)])].mean() for _ in range(2000)]


fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.8))
for row, T in enumerate(("chicago", "nyc")):
    rng = np.random.default_rng(0); P = np.load(f"{HERE}/out_cal/pred_{T}.npz"); di, Tt = P["di"], P["total"]
    Yc = np.load(YP[T]).astype(np.float64)[di, P["si"]]
    norm = lambda p: p / p.sum(1, keepdims=True)
    def err(p):
        E = norm(p) * Tt[:, None] - Yc; A = np.abs(E); return A.mean(), np.sqrt((E ** 2).mean()), np.bincount(di, A.mean(1)) / np.bincount(di)
    def final(fe, fo):
        F = {"eh": [np.load(f) for f in fe], "eo": [np.load(f) for f in fo]}
        fin = 0.5 * norm(np.mean(F["eh"], 0)) + 0.5 * norm(np.mean(F["eo"], 0))
        single = np.array([err(0.5 * norm(a) + 0.5 * norm(b))[:2] for a, b in zip(F["eh"], F["eo"])]); return err(fin), single
    nat, _ = final(*[[f"{HERE}/out/multi_{T}_from_{NAT[T]}_v{h}_seed{s}.npy" for s in range(10)] for h in ("eh", "eo")])
    res = {q: final(*[[f"{KG}/multi_{T}_from_{HEL[T](c, r)}_v{h}_seed{r-1}.npy" for r in range(1, 11)] for h in ("eh", "eo")]) for q, c in Q}
    print(f"\n===== {T.upper()}: final method, MAE / RMSE; single runs mean +- SD; mean daily MAE difference to q = 1 [95% interval] =====")
    print(f"   native   {nat[0]:.2f} / {nat[1]:.2f}")
    for q, _ in Q:
        (a, b, d), sg = res[q]; dd = d - res[1.0][0][2]; bs = boot(dd, rng) if q != 1.0 else [0, 0]
        print(f"   q={q:<5} {a:.2f} / {b:.2f} | single runs MAE {sg[:,0].mean():.2f} +- {sg[:,0].std(ddof=1):.2f}, RMSE {sg[:,1].mean():.2f} +- {sg[:,1].std(ddof=1):.2f}"
              + ("" if q == 1.0 else f" | diff {dd.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"))
    qs = [q for q, _ in Q]
    for col, (k, nm) in enumerate(((0, "MAE"), (1, "RMSE"))):
        ax = axs[row, col]; v = [res[q][0][k] for q in qs]; m = np.array([res[q][1][:, k].mean() for q in qs]); sd = np.array([res[q][1][:, k].std(ddof=1) for q in qs])
        ax.fill_between(qs, m - sd, m + sd, color="#1f5aa6", alpha=0.15, lw=0, label="Single runs, mean $\\pm$ SD")
        ax.plot(qs, m, color="#1f5aa6", lw=0.8, ls=":")
        ax.plot(qs, v, "o-", color="#1f5aa6", lw=1.4, ms=4, label="Ten-run ensemble")
        ax.axhline(nat[k], color="#8c8c8c", ls="--", lw=0.9, label="Native helpers")
        ax.axvline(1, color="#e07b39", lw=0.8, alpha=0.7)
        ax.set_xscale("log"); ax.set_xticks(qs); ax.set_xticklabels(["0.25", "0.5", "0.75", "1", "1.5", "2", "4"])
        ax.set_title(f"{'Chicago' if T == 'chicago' else 'New York City'}: {nm}"); ax.set_ylabel(nm)
        if row == 1: ax.set_xlabel("Block area relative to the target median area, q")
axs[0, 0].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sweep.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_sweep.png", dpi=170, bbox_inches="tight")
print("figure written")
