"""Gain of scale matching against the size gap, for every target that has been run with native (N) and scale-matched (M) helpers.
All numbers are parsed from the saved result files (results/control_eval.txt, coarse_eval.txt, austin_eval.txt); region areas are
read from the feature files. Gain = mean daily MAE difference (N - M) / MAE of N, with the 95% 7-day block-bootstrap interval of the
difference divided by the same MAE. San Francisco has no helper finer than its regions, so N = M by construction (gain 0, 'not
applicable'). Output: results/scale_gain.txt and ../figs/fig_scalegain.{pdf,png}. Usage (from improve/): python scale_gain_plot.py"""
import os, re, itertools, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = f"{HERE}/../results"; FEAT = f"{HERE}/../features"; MC = f"{HERE}/../multicity/data"
FIGS = f"{HERE}/../../figs"; os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, INK = "#1f5aa6", "#8c8c8c", "#333333"
NL = chr(10)
NUM = r"([+-]?\d+\.\d+)"
DIFF = rf"N\s+minus M: mean daily MAE difference {NUM} \[{NUM}, {NUM}\]"


def area(path): return float(np.median(np.load(path)["area_km2"]))


AREA = {"chicago": area(f"{FEAT}/region_static_chicago.npz"), "nyc": area(f"{FEAT}/region_static_nyc.npz"), "sf": area(f"{FEAT}/region_static_sf.npz"),
        "dc": area(f"{FEAT}/region_static_dc.npz"), "sf_zcta": area(f"{MC}/region_static_sf_zcta.npz"),
        "dc_cluster": area(f"{MC}/region_static_dc_cluster.npz"), "austin": area(f"{MC}/region_static_austin.npz")}
HELP = {"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"], "sf": ["chicago", "nyc"], "dc": ["chicago", "nyc", "sf"],
        "sf_zcta": ["chicago", "nyc"], "dc_cluster": ["chicago", "nyc", "sf"], "austin": ["chicago", "nyc", "sf"]}
NAME = {"chicago": "Chicago", "nyc": "NYC", "sf": "SF", "dc": "DC", "sf_zcta": "SF zip-code areas", "dc_cluster": "DC clusters", "austin": "Austin"}

rows = {}                                                                    # case -> (MAE_N, MAE_M, diff, lo, hi)
txt = open(f"{R}/control_eval.txt", encoding="utf-8").read()
for key, head in (("chicago", "CHICAGO"), ("nyc", "NYC")):                    # development targets: Kaggle runs of both conditions
    sec = txt[txt.index(f"===== {head} "):]; e = sec.find(NL + "=====", 5); sec = sec if e < 0 else sec[:e]
    n = float(re.search(r"^\s+N\s+final (\S+) /", sec, re.M).group(1)); m = float(re.search(r"^\s+M\s+final (\S+) /", sec, re.M).group(1))
    d = re.search(rf"^\s+{DIFF}", sec, re.M); rows[key] = (n, m, *map(float, d.groups()))
txt = open(f"{R}/coarse_eval.txt", encoding="utf-8").read()
for key in ("dc", "sf_zcta", "dc_cluster"):
    sec = txt[txt.index(f"===== {key} ("):]; e = sec.find(NL + "=====", 5); sec = sec if e < 0 else sec[:e]
    n = float(re.search(r"N final \(native helpers\)\s+MAE (\S+)", sec).group(1)); m = float(re.search(r"M final \(scale-matched\)\s+MAE (\S+)", sec).group(1))
    rows[key] = (n, m, *map(float, re.search(DIFF, sec).groups()))
txt = open(f"{R}/austin_eval.txt", encoding="utf-8").read()
n = float(re.search(r"Final, native helpers \(N\)\s+(\S+) /", txt).group(1)); m = float(re.search(r"Final, scale-matched \(M\)\s+(\S+) /", txt).group(1))
rows["austin"] = (n, m, *map(float, re.search(DIFF, txt).groups()))

table = []
for c, (n, m, d, lo, hi) in rows.items():
    a = AREA[c]; ratios = [a / AREA[h] for h in HELP[c] if AREA[h] < a]; fin = a / min(AREA[h] for h in HELP[c])
    table.append(dict(case=c, area=a, finest=fin, gm=float(np.exp(np.mean(np.log(ratios)))), merged=len(ratios), n=n, m=m, g=100 * d / n, lo=100 * lo / n, hi=100 * hi / n))
sf_fin = AREA["sf"] / min(AREA[h] for h in HELP["sf"])


def exact_spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); rho = float(np.corrcoef(rx, ry)[0, 1])
    perms = [abs(float(np.corrcoef(rx, np.array(p))[0, 1])) for p in itertools.permutations(ry)]
    return rho, float(np.mean(np.array(perms) >= abs(rho) - 1e-12))


lines = ["Gain of scale-matched (M) over native (N) helpers, fully strict mean-forecast MAE (final method, ten runs per head)",
         f"{'case':18s} {'target area':>11s} {'finest ratio':>12s} {'geo-mean ratio':>14s} {'merged':>6s} {'MAE N':>7s} {'MAE M':>7s} {'gain %':>7s} {'95% interval':>16s}"]
for t in sorted(table, key=lambda r: -r["finest"]):
    lines.append(f"{NAME[t['case']]:18s} {t['area']:11.2f} {t['finest']:12.2f} {t['gm']:14.2f} {t['merged']:6d} {t['n']:7.3f} {t['m']:7.3f} {t['g']:7.2f} [{t['lo']:5.2f}, {t['hi']:5.2f}]")
lines.append(f"{'SF':18s} {AREA['sf']:11.2f} {sf_fin:12.2f} {'-':>14s} {0:6d}   N = M by construction (no helper finer than the target): gain 0, not applicable")
for lab, key in (("finest-helper ratio", "finest"), ("geometric-mean ratio over merged helpers", "gm")):
    rho, p = exact_spearman([t[key] for t in table], [t["g"] for t in table])
    lines.append(f"Spearman rank correlation, gain vs {lab}: rho = {rho:+.2f}, exact two-sided permutation p = {p:.2f} (n = {len(table)} cases; not a test of a law)")
lines.append("MAE of N and M for Chicago and NYC are from the Kaggle runs of the control experiment (both conditions on the same workers); the others are laptop runs.")
os.makedirs(R, exist_ok=True); open(f"{R}/scale_gain.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.28}, sharey=True)
OFF = {"chicago": (-6, 6, "right"), "nyc": (6, 4, "left"), "dc": (6, 5, "left"), "sf_zcta": (-6, 7, "right"), "dc_cluster": (6, 6, "left"), "austin": (6, -10, "left")}
for ax, key, xl, title in ((axs[0], "finest", "Target median region area ÷ area of the finest share helper", "(a) finest helper"),
                           (axs[1], "gm", "Same ratio, geometric mean over the merged helpers", "(b) all merged helpers")):
    ax.axhline(0, color=GREY, lw=0.6, zorder=0)
    if key == "finest": ax.axvline(1, color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=0)
    for t in table:
        x = t[key]
        ax.plot([x, x], [t["lo"], t["hi"]], color=BLUE, lw=1.0, zorder=2); ax.plot(x, t["g"], "o", ms=6, color=BLUE, mec="white", mew=0.8, zorder=3)
        dx, dy, ha = OFF[t["case"]]; ax.annotate(NAME[t["case"]], (x, t["g"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8, color=INK)
    if key == "finest":
        ax.plot(sf_fin, 0, "o", ms=6, mfc="white", mec=GREY, mew=1.0, zorder=3); ax.annotate("SF (no finer helper)", (sf_fin, 0), xytext=(-2, -13), textcoords="offset points", ha="left", fontsize=8, color=INK)
        ax.text(1.06, ax.get_ylim()[1] * 0.0 + 14.6, "helpers merged →", fontsize=7.5, color=GREY, va="top")
    ax.set_xscale("log"); ax.set_xlabel(xl, fontsize=8.5); ax.set_title(title, loc="left", fontsize=9)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"]); ax.minorticks_off()
axs[0].set_ylabel("Relative MAE gain of scale matching (%)"); axs[0].set_ylim(-2.8, 16.5); axs[0].set_xlim(0.11, 38); axs[1].set_xlim(1.2, 14)
fig.savefig(f"{FIGS}/fig_scalegain.pdf", bbox_inches="tight"); fig.savefig(f"{FIGS}/fig_scalegain.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print("figure saved")
