"""PROTOCOL_V2.md: calendar-only (fully strict) predictions for the v2 evaluation, run once after v2 was adopted on the
development targets. Head ep for every target (condition M helpers, run r = partition r, seed r - 1), and the v1 heads eh / eo
for the targets whose v1 predictions were lost with the local data (Chicago, NYC, SF). Restartable (finished seeds reused).
Run from improve/."""
import subprocess, sys
PY = sys.executable
M = {"sf": lambda r: "chicago,nyc", "dc": lambda r: f"chicago,nyc,sf_for_dc_p{r}",
     "sf_zcta": lambda r: f"chicago,nyc_for_sf_zcta_p{r}", "dc_cluster": lambda r: f"chicago,nyc_for_dc_cluster_p{r},sf_for_dc_cluster_p{r}",
     "austin": lambda r: f"chicago,nyc,sf_for_austin_p{r}",
     "chicago": lambda r: f"nyc_for_chicago_p{r},sf_for_chicago_p{r}", "nyc": lambda r: f"chicago,sf_for_nyc_p{r}"}
JOBS = [(T, v) for T in ("sf", "dc", "sf_zcta", "dc_cluster", "austin") for v in ("ep",)]
JOBS += [("sf", "eh"), ("sf", "eo")] + [(T, v) for T in ("chicago", "nyc") for v in ("ep", "eh", "eo")]
JOBS = [(T, v, r) for T, v in JOBS for r in range(1, 11)]
for k, (T, v, r) in enumerate(JOBS, 1):
    print(f"[{k}/{len(JOBS)}] {T} {v} r={r}", flush=True)
    subprocess.run([PY, "run_x.py", T, M[T](r), str(r - 1), v, "--calendar"], check=True)
print("all v2 held-out runs done", flush=True)
