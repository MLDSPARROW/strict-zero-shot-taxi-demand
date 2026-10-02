"""Partition-diverse runs: run r (1..5) uses partition r of every merged helper and model seed r-1 (so it pairs with seed r-1
of the single-partition runs)."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); tgt = sys.argv[1]
H = {"chicago": ["nyc_for_chicago", "sf_for_chicago"], "nyc": ["chicago", "sf_for_nyc"]}[tgt]
with open(f"{HERE}/runner15_{tgt}_log.txt", "a") as log:
    for fl in ["eh", "eo"]:
        for r in range(1, 6):
            src = ",".join(h if h == "chicago" else f"{h}_p{r}" for h in H)
            log.write(f"=== {tgt} <- {src} variant {fl} seed {r-1}\n"); log.flush()
            subprocess.run([sys.executable, f"{HERE}/run_x.py", tgt, src, str(r - 1), fl], stdout=log, stderr=subprocess.STDOUT)
open(f"{HERE}/runner15_{tgt}_done.txt", "w").write("done")
