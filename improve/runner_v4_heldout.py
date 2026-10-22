"""PROTOCOL_V4.md addendum 2: held-out evaluation of Way 2 (switch d), fully strict calendar-slot predictions (the target demand is never
opened), heads ehd and eod, runs r = 1..10 (helper partition r, model seed r - 1), with the helper configuration of the final method of the
paper for each target (the same as runner_v2_heldout.py / runner_coarse.py / runner_austin.py). 100 runs, in the order DC, SF, sf_zcta,
dc_cluster, Austin; for each target run r of both heads before run r + 1, so that partial results are balanced. Restartable (finished
seeds are reused by run_x.py). Jobs are split between workers by index: python runner_v4_heldout.py <worker 0..n-1> <n workers>.
Run from improve/."""
import subprocess, sys
PY = sys.executable
w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
M = {"dc": lambda r: f"chicago,nyc,sf_for_dc_p{r}", "sf": lambda r: "chicago,nyc",
     "sf_zcta": lambda r: f"chicago,nyc_for_sf_zcta_p{r}", "dc_cluster": lambda r: f"chicago,nyc_for_dc_cluster_p{r},sf_for_dc_cluster_p{r}",
     "austin": lambda r: f"chicago,nyc,sf_for_austin_p{r}"}
ALL = [(T, r, h) for T in M for r in range(1, 11) for h in ("ehd", "eod")]
MINE = [j for k, j in enumerate(ALL) if k % nw == w]
for k, (T, r, h) in enumerate(MINE, 1):
    print(f"[worker {w}: {k}/{len(MINE)}] {T} r={r} {h}", flush=True)
    subprocess.run([PY, "run_x.py", T, M[T](r), str(r - 1), h, "--calendar"], check=True)
print(f"worker {w} done", flush=True)
