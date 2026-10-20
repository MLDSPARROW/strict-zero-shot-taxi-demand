"""PROTOCOL_V2.md addendum 6: pseudo-target runs for the development targets (T = chicago, nyc). The learned allocator (heads eh and eo,
runs r = 1..5: helper partition r, seed r - 1, demand-reading mode, the pseudo-target's own data used for scoring only) and the Chi et al.
allocation (5 seeds, native scale) are trained on the OTHER share helper of T only, so that no taxi data of T enter. Restartable.
Run from improve/."""
import subprocess, sys
PY = sys.executable
LEARNED = [("chicago", "nyc_for_chicago_p1", lambda r: f"sf_for_chicago_p{r}"), ("chicago", "sf_for_chicago_p1", lambda r: f"nyc_for_chicago_p{r}"),
           ("nyc", "chicago", lambda r: f"sf_for_chicago_p{r}"), ("nyc", "sf_for_nyc_p1", lambda r: "chicago")]
CHI = [("chicago", "nyc", "sf"), ("chicago", "sf", "nyc"), ("nyc", "chicago", "sf"), ("nyc", "sf", "chicago")]       # (T, pseudo-target c, helper)
JOBS = [[PY, "run_x.py", tg, srcs(r), str(r - 1), h] for T, tg, srcs in LEARNED for h in ("eh", "eo") for r in range(1, 6)]
JOBS += [[PY, "chi_baseline.py", c, hel, "0,1,2,3,4", f"_pseudoT_{T}"] for T, c, hel in CHI]
for k, j in enumerate(JOBS, 1):
    print(f"[{k}/{len(JOBS)}] {' '.join(j[1:])}", flush=True)
    subprocess.run(j, check=True)
print("all dev pseudo-target runs done", flush=True)
