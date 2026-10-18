"""Austin runs (PROTOCOL_MORE_CITIES.md, Austin addendum), one at a time, calendar slots only. Restartable. Run from improve/."""
import subprocess, sys
PY = sys.executable
S10 = ",".join(map(str, range(10)))
JOBS = [(["chi_baseline.py", "austin", "chicago,nyc,sf", S10], "."),
        (["run_multi.py", "austin", "chicago,nyc,sf", S10, "", "--calendar"], "../multicity")]
for h in ("eh", "eo"):
    JOBS += [(["run_x.py", "austin", "chicago,nyc,sf", str(r - 1), h, "--calendar"], ".") for r in range(1, 11)]
    JOBS += [(["run_x.py", "austin", f"chicago,nyc,sf_for_austin_p{r}", str(r - 1), h, "--calendar"], ".") for r in range(1, 11)]
for k, (j, cwd) in enumerate(JOBS, 1):
    print(f"[{k}/{len(JOBS)}] {' '.join(j)}", flush=True)
    subprocess.run([PY, *j], check=True, cwd=cwd)
print("all Austin runs done", flush=True)
