"""Held-out targets per PROTOCOL_HELDOUT.md (frozen 2026-10-02 21:26)."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); tgt = sys.argv[1]
with open(f"{HERE}/runner_heldout_{tgt}_log.txt", "a") as log:
    for fl in ["eh", "eo"]:
        for r in range(1, 11):
            src = "chicago,nyc" if tgt == "sf" else f"chicago,nyc,sf_for_dc_p{r}"
            log.write(f"=== {tgt} <- {src} variant {fl} seed {r-1}\n"); log.flush()
            subprocess.run([sys.executable, f"{HERE}/run_x.py", tgt, src, str(r - 1), fl], stdout=log, stderr=subprocess.STDOUT)
    base_src = "chicago,nyc" if tgt == "sf" else "chicago,nyc,sf"
    log.write(f"=== BASE joint allocator {tgt} <- {base_src}\n"); log.flush()
    subprocess.run([sys.executable, f"{HERE}/../multicity/run_multi.py", tgt, base_src, "0,1,2,3,4,5,6,7,8,9", ""], stdout=log, stderr=subprocess.STDOUT)
open(f"{HERE}/runner_heldout_{tgt}_done.txt", "w").write("done")
