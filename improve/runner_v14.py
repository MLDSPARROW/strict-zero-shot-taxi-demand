"""PROTOCOL_V14.md: pseudo-target runs of the final method v1 (heads eh and eo, runs r = 1..5: helper partition r, seed r - 1, the pseudo-target's own demand used for scoring only,
exactly as runner_pseudo_dev.py of PROTOCOL_V2 addendum 6) for the five held-out targets. Each share helper of a held-out target is a pseudo-target predicted by models trained on the
OTHER share helpers only, so no taxi datum of the held-out target enters. Restartable (run_x.py reuses finished seeds). Usage (from improve/): python runner_v14.py WORKER NWORKERS"""
import subprocess, sys
PY = sys.executable
S = lambda s: (lambda r: s)
# (pseudo-target tag, sources as a function of the partition r)
CONFIGS = [("chicago", S("nyc")), ("nyc", S("chicago")),                                                                                           # sf (native helpers)
           ("chicago", lambda r: f"nyc_for_sf_zcta_p{r}"), ("nyc_for_sf_zcta_p1", S("chicago")),                                                      # sf_zcta
           ("chicago", lambda r: f"nyc,sf_for_dc_p{r}"), ("nyc", lambda r: f"chicago,sf_for_dc_p{r}"), ("sf_for_dc_p1", S("chicago,nyc")),             # dc
           ("chicago", lambda r: f"nyc_for_dc_cluster_p{r},sf_for_dc_cluster_p{r}"), ("nyc_for_dc_cluster_p1", lambda r: f"chicago,sf_for_dc_cluster_p{r}"),
           ("sf_for_dc_cluster_p1", lambda r: f"chicago,nyc_for_dc_cluster_p{r}"),                                                                    # dc_cluster
           ("chicago", lambda r: f"nyc,sf_for_austin_p{r}"), ("nyc", lambda r: f"chicago,sf_for_austin_p{r}"), ("sf_for_austin_p1", S("chicago,nyc"))]  # austin
JOBS = [[tg, srcs(r), str(r - 1), h] for tg, srcs in CONFIGS for h in ("eh", "eo") for r in range(1, 6)]
if __name__ == "__main__":
    w, n = int(sys.argv[1]), int(sys.argv[2]); mine = JOBS[w::n]
    for k, j in enumerate(mine, 1):
        print(f"[{k}/{len(mine)}] run_x.py {' '.join(j)}", flush=True)
        subprocess.run([PY, "run_x.py", *j], check=True)
    print(f"worker {w}: all pseudo-target runs done", flush=True)
