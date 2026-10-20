"""PROTOCOL_V2.md addendum 3: recompute the v1 calendar-only predictions lost with the local data (Chicago, NYC, SF), heads eh
and eo, runs r = 1..10 (helper partition r, seed r - 1), exactly as the paper's final method, so that they can be scored with
the I1 median forecast. Restartable. Run from improve/."""
import subprocess, sys
PY = sys.executable
M = {"chicago": lambda r: f"nyc_for_chicago_p{r},sf_for_chicago_p{r}", "nyc": lambda r: f"chicago,sf_for_nyc_p{r}", "sf": lambda r: "chicago,nyc"}
JOBS = [(T, v, r) for T in ("chicago", "nyc", "sf") for v in ("eh", "eo") for r in range(1, 11)]
for k, (T, v, r) in enumerate(JOBS, 1):
    print(f"[{k}/{len(JOBS)}] {T} {v} r={r}", flush=True)
    subprocess.run([PY, "run_x.py", T, M[T](r), str(r - 1), v, "--calendar"], check=True)
print("all v1 restore runs done", flush=True)
