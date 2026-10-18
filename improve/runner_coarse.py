"""Runs of PROTOCOL_COARSE_TARGETS.md, one at a time, on calendar slots (target demand never opened).
run_x.py reuses finished seeds, so this runner can be stopped and restarted. Run from improve/."""
import subprocess, sys
PY = sys.executable
JOBS = []
for T, N, M in [("sf_zcta", "chicago,nyc", lambda r: f"chicago,nyc_for_sf_zcta_p{r}"),
                ("dc_cluster", "chicago,nyc,sf", lambda r: f"chicago,nyc_for_dc_cluster_p{r},sf_for_dc_cluster_p{r}"),
                ("dc", "chicago,nyc,sf", lambda r: f"chicago,nyc,sf_for_dc_p{r}")]:
    for h in ("eh", "eo"):
        JOBS.append([T, N, ",".join(map(str, range(10))), h])
        JOBS += [[T, M(r), str(r - 1), h] for r in range(1, 11)]
for k, j in enumerate(JOBS, 1):
    print(f"[{k}/{len(JOBS)}] run_x.py {' '.join(j)} --calendar", flush=True)
    subprocess.run([PY, "run_x.py", *j, "--calendar"], check=True)
print("all coarse-target runs done", flush=True)
