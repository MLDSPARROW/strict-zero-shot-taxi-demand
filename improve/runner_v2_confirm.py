"""PROTOCOL_V2.md addendum 2: confirmation on the development targets, runs r = 4..10 (helper partition r, seed r - 1) of the
heads eh, eo and ep (runs 1-3 come from the screen). Demand-reading mode, target demand used for scoring only. Run from improve/."""
import subprocess, sys
PY = sys.executable
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r},sf_for_chicago_p{r}", "nyc": lambda r: f"chicago,sf_for_nyc_p{r}"}
JOBS = [(T, r, v) for r in range(4, 11) for T in ("chicago", "nyc") for v in ("ep", "eo", "eh")]
for k, (T, r, v) in enumerate(JOBS, 1):
    print(f"[{k}/{len(JOBS)}] {T} r={r} {v}", flush=True)
    subprocess.run([PY, "run_x.py", T, SRC[T](r), str(r - 1), v], check=True)
print("all confirmation runs done", flush=True)
