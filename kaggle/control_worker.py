# PROTOCOL_REVISION.md part B: aggregation control runs, Kaggle worker K of 5 (K = 1..5 set below) (CPU). Target demand of NYC is not in the dataset.
import os, sys, glob, tarfile, subprocess, time
from concurrent.futures import ThreadPoolExecutor
K = 1
W = "/tmp/w"
if not os.path.exists(W):
    tars = glob.glob("/kaggle/input/**/urbanmind_control.tar", recursive=True)
    if tars: tarfile.open(tars[0]).extractall(W)
    else:
        import shutil; root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/run_x.py", recursive=True)[0])); shutil.copytree(root, W)
os.makedirs("/kaggle/working/out_cal", exist_ok=True); os.makedirs("/kaggle/working/logs", exist_ok=True)
if not os.path.exists(f"{W}/improve/out_cal"): os.symlink("/kaggle/working/out_cal", f"{W}/improve/out_cal")
JOBS = []
for cond in ("p", "R", "H", "D"):
    for r in range(1, 11):
        for tgt, hel in (("chicago", f"nyc_for_chicago_{cond}{r},sf_for_chicago_{cond}{r}"), ("nyc", f"chicago,sf_for_nyc_{cond}{r}")):
            for h in ("eh", "eo"): JOBS.append((tgt, hel, str(r - 1), h))
JOBS = JOBS[K - 1::5]
print(len(JOBS), "jobs", flush=True)
def go(j):
    t0 = time.time(); name = "_".join(j).replace(",", "+")
    with open(f"/kaggle/working/logs/{name}.txt", "w") as f:
        rc = subprocess.run([sys.executable, "run_x.py", *j, "--calendar"], cwd=f"{W}/improve", stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"{name} rc={rc} {time.time()-t0:.0f}s", flush=True); return rc
with ThreadPoolExecutor(2) as ex: rcs = list(ex.map(go, JOBS))
print("ALL DONE", "failures:", sum(r != 0 for r in rcs), flush=True)
