"""PROTOCOL_V5.md, comparison 1: the allocation of Chi et al. trained on the scale-matched helpers (Chi-M). Run r uses partition r of every
merged helper with model seed r - 1, output tag _M (out_cal/chi_<target>_M_seed<s>.npy). Sequential, restartable (finished seeds are
skipped). Run from improve/."""
import os, subprocess, sys
PY = sys.executable
M = {"chicago": lambda r: f"nyc_for_chicago_p{r},sf_for_chicago_p{r}", "nyc": lambda r: f"chicago,sf_for_nyc_p{r}",
     "dc": lambda r: f"chicago,nyc,sf_for_dc_p{r}", "sf_zcta": lambda r: f"chicago,nyc_for_sf_zcta_p{r}",
     "dc_cluster": lambda r: f"chicago,nyc_for_dc_cluster_p{r},sf_for_dc_cluster_p{r}", "austin": lambda r: f"chicago,nyc,sf_for_austin_p{r}"}
JOBS = [(T, r) for T in M for r in range(1, 11) if not os.path.exists(f"out_cal/chi_{T}_M_seed{r - 1}.npy")]
for k, (T, r) in enumerate(JOBS, 1):
    print(f"[{k}/{len(JOBS)}] chi_baseline.py {T} {M[T](r)} {r - 1} _M", flush=True)
    subprocess.run([PY, "chi_baseline.py", T, M[T](r), str(r - 1), "_M"], check=True)
print("all Chi-M runs done", flush=True)
