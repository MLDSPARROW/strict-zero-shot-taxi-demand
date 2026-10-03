"""PROTOCOL_REVISION.md part A: retrain the held-out targets with calendar slots (--calendar), on the same CPU as before.
Usage: python runner_revision.py sf|dc"""
import sys, subprocess, os
HERE = os.path.dirname(os.path.abspath(__file__)); T = sys.argv[1]
JOBS = []
if T == "sf":
    for h in ("eh", "eo"): JOBS.append(["run_x.py", "sf", "chicago,nyc", ",".join(map(str, range(10))), h])
    JOBS.append(["../multicity/run_multi.py", "sf", "chicago,nyc", ",".join(map(str, range(10))), ""])
else:
    for h in ("eh", "eo"):
        for r in range(1, 11): JOBS.append(["run_x.py", "dc", f"chicago,nyc,sf_for_dc_p{r}", str(r - 1), h])
    JOBS.append(["../multicity/run_multi.py", "dc", "chicago,nyc,sf", ",".join(map(str, range(10))), ""])
for j in JOBS:
    print(">>", " ".join(j), flush=True)
    subprocess.run([sys.executable] + j + ["--calendar"], cwd=HERE, check=True)
print("ALL DONE", flush=True)
