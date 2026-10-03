"""Block-size sweep (PROTOCOL_SCALE_SWEEP.md): helpers re-aggregated with goal median block area q times the target
median region area, q in 0.25, 0.75, 1.5, 4 (q = 0.5, 1, 2 come from aggregate_control.py / aggregate_multi*.py).
10 partitions each. A helper whose native median area is already at least the goal is used at native scale for that q.
Output tags <H>_for_<T>_x<q code>p<r>. No target demand is read."""
import numpy as np
src = open("aggregate_control.py").read(); ns = {}; exec(src[:src.index("for T, H in")], ns)
choose_k, write, SP, YP, G = ns["choose_k"], ns["write"], ns["SP"], ns["YP"], ns["G"]
from sklearn.cluster import KMeans
Q = {"025": 0.25, "075": 0.75, "150": 1.5, "400": 4.0}
for T, H in [("chicago", "nyc"), ("chicago", "sf"), ("nyc", "sf")]:
    aT = np.median(np.load(SP[T])["area_km2"]); S = np.load(SP[H]); area = S["area_km2"]
    g = np.load(f"{G}/grid_{H}.npz"); Y = np.load(YP[H]).astype(np.float32); E = np.load(f"data/extra_{H}.npz")
    for code, q in Q.items():
        goal = q * aT
        if np.median(area) >= goal:
            print(f"{H} for {T}, q={q}: native median {np.median(area):.2f} >= goal {goal:.2f} km2 -> native scale", flush=True)
            for r in range(1, 11): write(H, T, f"{H}_for_{T}_x{code}p{r}", np.arange(len(area)), S, area, g, Y, E)
            continue
        K = ns["choose_k"](S, goal); meds = []
        for r in range(1, 11):
            lab = KMeans(K, n_init=4, random_state=r).fit(S["centroid_km"], sample_weight=area).labels_
            n, med, _ = write(H, T, f"{H}_for_{T}_x{code}p{r}", lab, S, area, g, Y, E); meds.append(med)
        print(f"{H} for {T}, q={q}: goal {goal:.2f} km2, K {K}, median block area {np.mean(meds):.2f} (range {min(meds):.2f} to {max(meds):.2f})", flush=True)
