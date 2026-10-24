"""PROTOCOL_V9.md: the allocator of the final method with the five bike-share cities as extra teachers (switch k; heads ehk and eok), development targets
Chicago and New York City, helper partition r and model seed r - 1. Demand-reading mode (run_x.py reads the target demand for scoring only), like the
screen of PROTOCOL_V4. Restartable (finished seeds are reused by run_x.py). Jobs are split between workers by index.
Usage (from improve/): python runner_v9.py <worker 0..n-1> <n workers> [screen|confirm]     screen: runs r = 1..3; confirm: runs r = 4..10"""
import subprocess, sys
from v9_common import srcs
PY = sys.executable
w, nw = int(sys.argv[1]), int(sys.argv[2]); stage = sys.argv[3] if len(sys.argv) > 3 else "screen"
RUNS = range(1, 4) if stage == "screen" else range(4, 11)
ALL = [(T, r, h) for r in RUNS for T in ("chicago", "nyc") for h in ("eh", "eo")]
MINE = [j for k, j in enumerate(ALL) if k % nw == w]
for k, (T, r, h) in enumerate(MINE, 1):
    print(f"[worker {w}: {k}/{len(MINE)}] {T} r={r} {h}k", flush=True)
    subprocess.run([PY, "run_x.py", T, ",".join(srcs(T, r)), str(r - 1), h + "k"], check=True)
print(f"worker {w} done", flush=True)
