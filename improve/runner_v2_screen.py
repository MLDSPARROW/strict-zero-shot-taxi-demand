"""PROTOCOL_V2.md screen: development targets Chicago and NYC only, runs r = 1..3 (helper partition r, seed r - 1), demand-
reading mode (target demand used for scoring only). New variants ep, epa (vs eo) and eha (vs eh); eo and eh re-run under
identical conditions. One run at a time; finished runs are reused (run_x.py resumes from saved seeds). Run from improve/."""
import subprocess, sys
PY = sys.executable
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r},sf_for_chicago_p{r}", "nyc": lambda r: f"chicago,sf_for_nyc_p{r}"}
JOBS = [(T, r, v) for r in (1, 2, 3) for T in ("chicago", "nyc") for v in ("epa", "ep", "eha", "eo", "eh")]
for k, (T, r, v) in enumerate(JOBS, 1):
    print(f"[{k}/{len(JOBS)}] {T} r={r} {v}", flush=True)
    subprocess.run([PY, "run_x.py", T, SRC[T](r), str(r - 1), v], check=True)
print("all screen runs done", flush=True)
