"""PROTOCOL_V4.md runs on the development targets, restartable: run_x.py reuses finished seeds and mapmae.py is skipped when its output
exists. Demand-reading mode as in the PROTOCOL_V2 screen (the target demand is used for scoring only, by v4_eval.py).
Stage screen: map pre-training of Chicago and NYC with seeds 0-2, then runs r = 1..3 (helper partition r, model seed r - 1) of the
heads eh and eo with the switches m, d and md. Stage confirm: runs r = 4..10 of the variants given (comma-separated).
Jobs are split between workers by index: python runner_v4.py <screen|confirm> <worker> <n workers> [variants]. Run from improve/."""
import os, subprocess, sys, time
PY = sys.executable
stage, w, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r},sf_for_chicago_p{r}", "nyc": lambda r: f"chicago,sf_for_nyc_p{r}"}
RR = range(1, 4) if stage == "screen" else range(4, 11)
VAR = ["d", "m", "md"] if stage == "screen" else sys.argv[4].split(",")
PRE = [(T, r - 1) for r in RR for T in ("chicago", "nyc")] if any("m" in v for v in VAR) else []
done = lambda T, s: os.path.exists(f"data/mapmae/_done__{T}__s{s}")
for k, (T, s) in enumerate(PRE):
    if k % nw != w or done(T, s): continue
    print(f"[worker {w}] map pre-training {T} seed {s}", flush=True)
    subprocess.run([PY, "mapmae.py", T, str(s)], check=True)
ALL = [(T, r, h + v) for r in RR for v in VAR for T in ("chicago", "nyc") for h in ("eh", "eo")]
MINE = [j for k, j in enumerate(ALL) if k % nw == w]
for k, (T, r, fl) in enumerate(MINE, 1):
    while "m" in fl and not done(T, r - 1): time.sleep(30)           # pre-training of the other worker still running
    print(f"[worker {w}: {k}/{len(MINE)}] {T} r={r} {fl}", flush=True)
    subprocess.run([PY, "run_x.py", T, SRC[T](r), str(r - 1), fl], check=True)
print(f"worker {w} done", flush=True)
