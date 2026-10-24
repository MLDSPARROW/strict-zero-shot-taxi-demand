"""PROTOCOL_SYDNEY.md: the final method for Sydney, heads eh and eo, runs r = 1..10 (helper partition r, model seed r - 1), calendar mode (--calendar): the
demand of the target is never opened (none exists). Restartable (finished seeds are reused by run_x.py). Jobs are split between workers by index, and the
trainer runs at below-normal priority so that the laptop stays usable.
Usage (from improve/): python runner_sydney.py <worker 0..n-1> <n workers>       helper names: sydney_srcs(r) below, shared with sydney_results.py"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable


def sydney_srcs(r):                                    # chicago, nyc, sf: the re-aggregated version of partition r when build_sydney_partitions.py made one, else native
    return [f"{H}_for_sydney_p{r}" if os.path.exists(f"{HERE}/data/agg/{H}_for_sydney_p{r}_Y.npy") else H for H in ("chicago", "nyc", "sf")]


if __name__ == "__main__":
    w, nw = int(sys.argv[1]), int(sys.argv[2])
    JOBS = [(r, h) for r in range(1, 11) for h in ("eh", "eo")]
    MINE = [j for k, j in enumerate(JOBS) if k % nw == w]
    for k, (r, h) in enumerate(MINE, 1):
        print(f"[worker {w}: {k}/{len(MINE)}] sydney r={r} {h} <- {','.join(sydney_srcs(r))}", flush=True)
        subprocess.run([PY, "run_x.py", "sydney", ",".join(sydney_srcs(r)), str(r - 1), h, "--calendar"], check=True, creationflags=0x4000 if os.name == "nt" else 0, cwd=HERE)
    print(f"worker {w} done", flush=True)
