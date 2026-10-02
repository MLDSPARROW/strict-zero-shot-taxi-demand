"""FULLY STRICT pipeline v2 (declared 2026-10-01 before running the rhythm part):
  LEVEL : car-free households (ACS) x geometric mean helper trips per car-free household   [post-hoc choice, flagged]
  RHYTHM: v1 = mean helper (day-of-week x slot) profile
          v2 = v1 x month factor, month factor = helper cities with the SAME calendar year (2021) only, normalised to mean 1
               (Chicago target -> NYC 2021; NYC target -> Chicago 2021). Uses no target taxi data.
  SHARE : our 10-seed models (eh, eho) and the simple baselines
Also stores the per-slot city-total error."""
import json, numpy as np, importlib.util, io, contextlib
from datetime import date, timedelta
spec = importlib.util.spec_from_file_location("st", "strict_total.py"); st = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(st)
d = json.load(open("raw/acs_city.json"))["data"]; G = {"chicago": "16000US1714000", "nyc": "16000US3651000", "sf": "16000US0667000"}
nocar = {c: d[g]["B25044"]["estimate"]["B25044003"] + d[g]["B25044"]["estimate"]["B25044010"] for c, g in G.items()}
YEAR = {"chicago": 2021, "nyc": 2021, "sf": 2023}
def month_of(c): return np.array([(st.Ypath[c][1] + timedelta(days=k)).month for k in range(st.Y[c].shape[0])])
def strict_T(tgt, S, di, si, v):
    level = nocar[tgt] * np.exp(np.mean([np.log(st.Y[c].sum() / nocar[c]) for c in S]))
    rh = np.mean([st.rhythm(c) for c in S], 0); nd = np.bincount(st.dow[tgt], minlength=7); w = st.dow[tgt][di]
    T = level * rh[w, si] / nd[w]
    if v == 2:
        same = [c for c in S if YEAR[c] == YEAR[tgt]]
        if same:
            mf = []
            for c in same:
                tot = st.Y[c].sum((1, 2)); mo = month_of(c); m = np.array([tot[mo == k].mean() for k in range(1, 13)]); mf.append(m / m.mean())
            mf = np.mean(mf, 0); mt = month_of(tgt)[di]
            daymean = np.array([mf[month_of(tgt)[k] - 1] for k in range(st.Y[tgt].shape[0])]).mean()
            T = T * mf[mt - 1] / daymean
    return T
if __name__ == "__main__":
    import baselines as BL  # noqa  (re-runs baselines quietly is too slow; we recompute simple shares here)
