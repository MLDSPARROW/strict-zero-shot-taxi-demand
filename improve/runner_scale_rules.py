"""PROTOCOL_V3.md runs, restartable (finished seeds are reused by run_x.py), calendar-slot mode (the target demand is never opened).
Jobs are in priority order and are split between workers by index: python runner_scale_rules.py <worker 0..n-1> <n workers>.
Run from improve/."""
import subprocess, sys
PY = sys.executable
w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
JOBS = [("sf_zcta", lambda r: f"chicago_for_sf_zcta_mp{r},nyc_for_sf_zcta_mp{r}", range(1, 11))]
JOBS += [("dc", lambda r, c=c: f"chicago,nyc,sf_for_dc_x{c}p{r}", range(1, 6)) for c in ("mean", "150", "200", "075", "400")]
ALL = [(tgt, srcs(r), str(r - 1), h) for tgt, srcs, rr in JOBS for r in rr for h in ("eh", "eo")]
MINE = [j for k, j in enumerate(ALL) if k % nw == w]
for k, (tgt, srcs, seed, h) in enumerate(MINE, 1):
    print(f"[worker {w}: {k}/{len(MINE)}] {tgt} {srcs} seed {seed} {h}", flush=True)
    subprocess.run([PY, "run_x.py", tgt, srcs, seed, h, "--calendar"], check=True)
print(f"worker {w} done", flush=True)
