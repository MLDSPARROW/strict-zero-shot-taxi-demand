"""PROTOCOL_REVISION.md part A, check 4: the retrained held-out models must reproduce the earlier predictions on every slot
that existed before (max abs share difference below 1e-5). Reads target demand only to rebuild the old slot list."""
import os, numpy as np
from datetime import date
from slots import calendar_slots
HERE = os.path.dirname(os.path.abspath(__file__)); M = f"{HERE}/../multicity"
CFG = {"sf": (f"{M}/data/sf_2023_tract_30min.npy", date(2023, 1, 1)), "dc": (f"{M}/data/dc_2021_tract_30min.npy", date(2021, 1, 1))}
MODEL_H = {"sf": lambda r: "chicago+nyc", "dc": lambda r: f"chicago+nyc+sf_for_dc_p{r}"}
SHARE = {"sf": "chicago+nyc", "dc": "chicago+nyc+sf"}
for T, (yp, start) in CFG.items():
    Y = np.load(yp); tot = Y.sum(2); old = [(a, b) for a in range(Y.shape[0]) for b in range(48) if tot[a, b] > 0]
    new = calendar_slots(T, start); pos = {x: i for i, x in enumerate(new)}; rows = np.array([pos[x] for x in old])
    pairs = [(f"{M}/multi_{T}_from_{SHARE[T]}_seed{s}.npy", f"{M}/out_cal/multi_{T}_from_{SHARE[T]}_seed{s}.npy") for s in range(10)]
    for h in ("eh", "eo"):
        pairs += [(f"{HERE}/out/multi_{T}_from_{MODEL_H[T](r)}_v{h}_seed{r-1}.npy", f"{HERE}/out_cal/multi_{T}_from_{MODEL_H[T](r)}_v{h}_seed{r-1}.npy") for r in range(1, 11)]
    d = [np.abs(np.load(a) - np.load(b)[rows]).max() for a, b in pairs]
    print(f"{T}: {len(pairs)} runs, {len(old)} old slots, {len(new)} calendar slots; max abs share difference on old slots "
          f"{max(d):.2e} (runs above 1e-5: {sum(x > 1e-5 for x in d)})")
